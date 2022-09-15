use aho_corasick::AhoCorasick;
use rustc_hash::FxHashMap;

use crate::{
	ast,
	ast::{ExprKind, GlobalDeclKind, Ident, StmtKind, VarDecl},
	diagnostic::{Diagnostics, Span},
	resolve::{
		inbuilt::{
			reserved_matcher,
			AccessMode,
			AddressSpace,
			AttributeType,
			Builtin,
			DepthTextureType,
			InterpolationSample,
			InterpolationType,
			MatType,
			PrimitiveType,
			SampledTextureType,
			SamplerType,
			StorageTextureType,
			TexelFormat,
			VecType,
		},
		inbuilt_functions::{InbuiltFunction, InbuiltFunctionKws},
		index::Index,
		ir::{FloatType, FnTarget, InbuiltType, LocalId, SampleType},
	},
	text::{Interner, Text},
};

mod features;
mod inbuilt;
mod inbuilt_functions;
mod index;
pub mod ir;

pub fn resolve(tu: ast::TranslationUnit, intern: &mut Interner, diagnostics: &mut Diagnostics) -> ir::TranslationUnit {
	let index = index::generate_index(&tu, diagnostics);

	let mut out = ir::TranslationUnit::default();

	for enable in tu.enables {
		out.features.enable(enable, intern, diagnostics);
	}

	let mut resolver = Resolver {
		ty_kws: TypeKws::init(intern),
		attrib_kws: AttributeKws::init(intern),
		fn_kws: InbuiltFunctionKws::init(intern),
		tu: &mut out,
		index,
		diagnostics,
		intern,
		reserved_matcher: reserved_matcher(),
		locals: 0,
		in_loop: false,
		in_continuing: false,
		scopes: Vec::new(),
	};

	for decl in tu.decls {
		resolver.decl(decl);
	}

	out
}

struct Resolver<'a> {
	index: Index,
	tu: &'a mut ir::TranslationUnit,
	diagnostics: &'a mut Diagnostics,
	intern: &'a mut Interner,
	reserved_matcher: AhoCorasick,
	ty_kws: TypeKws,
	attrib_kws: AttributeKws,
	fn_kws: InbuiltFunctionKws,
	locals: u32,
	in_loop: bool,
	in_continuing: bool,
	scopes: Vec<FxHashMap<Text, (LocalId, Span)>>,
}

impl<'a> Resolver<'a> {
	fn decl(&mut self, decl: ast::GlobalDecl) {
		self.locals = 0;

		let kind = match decl.kind {
			GlobalDeclKind::Fn(f) => ir::DeclKind::Fn(self.fn_(f)),
			GlobalDeclKind::Override(ov) => ir::DeclKind::Override(self.ov(ov)),
			GlobalDeclKind::Var(v) => ir::DeclKind::Var(ir::Var {
				attribs: v.attribs.into_iter().filter_map(|x| self.attrib(x)).collect(),
				inner: self.var(v.inner),
			}),
			GlobalDeclKind::Let(l) => ir::DeclKind::Let(self.let_(l)),
			GlobalDeclKind::Const(c) => ir::DeclKind::Const(self.let_(c)),
			GlobalDeclKind::StaticAssert(s) => ir::DeclKind::StaticAssert(self.expr(s.expr)),
			GlobalDeclKind::Struct(s) => {
				self.verify_ident(s.name);
				ir::DeclKind::Struct(ir::Struct {
					name: s.name,
					fields: s.fields.into_iter().map(|f| self.arg(f)).collect(),
				})
			},
			GlobalDeclKind::Type(ty) => {
				self.verify_ident(ty.name);
				ir::DeclKind::Type(ir::TypeDecl {
					name: ty.name,
					ty: self.ty(ty.ty),
				})
			},
		};

		let decl = ir::Decl { kind, span: decl.span };

		self.tu.decls.push(decl);
	}

	fn fn_(&mut self, fn_: ast::Fn) -> ir::Fn {
		self.verify_ident(fn_.name);

		self.locals += fn_.args.len() as u32;
		let mut args = FxHashMap::default();

		for (id, arg) in fn_.args.iter().enumerate() {
			let old = args.insert(arg.name.name, (LocalId(id as u32), arg.span));
			if let Some((_, span)) = old {
				self.diagnostics
					.push(arg.name.span.error("duplicate argument name") + span.label("previous argument declaration"));
			}
		}

		self.scopes.push(args);
		let block = self.block(fn_.block);
		self.scopes.pop();

		ir::Fn {
			attribs: fn_.attribs.into_iter().filter_map(|x| self.attrib(x)).collect(),
			name: fn_.name,
			args: fn_.args.into_iter().map(|x| self.arg(x)).collect(),
			ret_attribs: fn_.ret_attribs.into_iter().filter_map(|x| self.attrib(x)).collect(),
			ret: fn_.ret.map(|x| self.ty(x)),
			block,
		}
	}

	fn ov(&mut self, o: ast::Override) -> ir::Override {
		self.verify_ident(o.name);
		ir::Override {
			attribs: o.attribs.into_iter().filter_map(|x| self.attrib(x)).collect(),
			name: o.name,
			ty: o.ty.map(|x| self.ty(x)),
			val: o.val.map(|x| self.expr(x)),
		}
	}

	fn arg(&mut self, arg: ast::Arg) -> ir::Arg {
		self.verify_ident(arg.name);
		ir::Arg {
			attribs: arg.attribs.into_iter().filter_map(|x| self.attrib(x)).collect(),
			name: arg.name,
			ty: self.ty(arg.ty),
			span: arg.span,
		}
	}

	fn let_(&mut self, l: ast::Let) -> ir::Let {
		self.verify_ident(l.name);
		ir::Let {
			name: l.name,
			ty: l.ty.map(|x| self.ty(x)),
			val: self.expr(l.val),
		}
	}

	fn var(&mut self, v: ast::VarNoAttribs) -> ir::VarNoAttribs {
		self.verify_ident(v.name);
		ir::VarNoAttribs {
			address_space: v.address_space.and_then(|x| self.address_space(x)),
			access_mode: v.access_mode.and_then(|x| self.access_mode(x)),
			name: v.name,
			ty: v.ty.map(|x| self.ty(x)),
			val: v.val.map(|x| self.expr(x)),
		}
	}

	fn ty(&mut self, ty: ast::Type) -> ir::Type {
		let span = ty.span;
		let t = match &ty.kind {
			ast::TypeKind::Ident(ident, generics) => Some((*ident, generics.len() == 0)),
			_ => None,
		};
		let kind = if let Some(inbuilt) = self.inbuilt(ty) {
			ir::TypeKind::Inbuilt(inbuilt)
		} else {
			let (ident, no_generics) = t.unwrap();
			if !no_generics {
				self.diagnostics
					.push(ident.span.error("unexpected generics on type") + ident.span.marker());
			}

			if let Some(user) = self.index.get(ident.name) {
				ir::TypeKind::User(user)
			} else {
				self.diagnostics
					.push(ident.span.error("undefined type") + ident.span.marker());
				ir::TypeKind::Inbuilt(InbuiltType::AbstractInt)
			}
		};

		ir::Type { kind, span }
	}

	fn block(&mut self, block: ast::Block) -> ir::Block {
		let mut stmts = Vec::with_capacity(block.stmts.len());
		let last = block.stmts.len().wrapping_sub(1);
		for (i, stmt) in block.stmts.into_iter().enumerate() {
			if let Some(stmt) = self.stmt(stmt) {
				if i != last {
					if matches!(stmt.kind, ir::StmtKind::Continuing(_)) && self.in_loop {
						self.diagnostics.push(
							stmt.span.error("`continuing` must be the last statement in `loop`") + stmt.span.marker(),
						);
					} else if matches!(stmt.kind, ir::StmtKind::BreakIf(_)) && self.in_continuing {
						self.diagnostics.push(
							stmt.span.error("`break if` must be the last statement in `continuing`")
								+ stmt.span.marker(),
						);
					}
				} else {
					stmts.push(stmt);
				}
			}
		}

		ir::Block { stmts }
	}

	fn stmt(&mut self, stmt: ast::Stmt) -> Option<ir::Stmt> {
		let kind = match stmt.kind {
			StmtKind::Block(block) => {
				self.scopes.push(FxHashMap::default());
				let ret = ir::StmtKind::Block(self.block(block));
				self.scopes.pop();
				ret
			},
			StmtKind::Expr(expr) => ir::StmtKind::Expr(self.expr_statement(expr)?.kind),
			StmtKind::Break => ir::StmtKind::Break,
			StmtKind::Continue => ir::StmtKind::Continue,
			StmtKind::Discard => ir::StmtKind::Discard,
			StmtKind::For(for_) => {
				self.scopes.push(FxHashMap::default());
				let init = for_.init.and_then(|x| self.expr_statement(x));
				let cond = for_.cond.map(|x| self.expr(x));
				let update = for_.update.and_then(|x| self.expr_statement(x)).and_then(|x| {
					if matches!(x.kind, ir::ExprStatementKind::VarDecl(_)) {
						self.diagnostics
							.push(x.span.error("variable declaration not allowed here") + x.span.marker());
						None
					} else {
						Some(x)
					}
				});
				self.scopes.push(FxHashMap::default());
				let block = self.block(for_.block);
				self.scopes.pop();
				self.scopes.pop();

				ir::StmtKind::For(ir::For {
					init,
					cond,
					update,
					block,
				})
			},
			StmtKind::If(if_) => {
				let cond = self.expr(if_.cond);
				self.scopes.push(FxHashMap::default());
				let block = self.block(if_.block);
				self.scopes.pop();
				let mut else_ = if_.else_.and_then(|x| self.stmt(*x)).map(Box::new);

				if !matches!(
					else_.as_ref().map(|x| &x.kind),
					Some(ir::StmtKind::If(_) | ir::StmtKind::Block(_)) | None
				) {
					self.diagnostics
						.push(stmt.span.error("`else` must be followed by `if` or block") + stmt.span.marker());
					else_ = None;
				}

				ir::StmtKind::If(ir::If { cond, block, else_ })
			},
			StmtKind::Loop(block) => {
				self.in_loop = true;
				self.scopes.push(FxHashMap::default());
				let block = self.block(block);
				self.scopes.pop();
				self.in_loop = false;
				ir::StmtKind::Loop(block)
			},
			StmtKind::Return(expr) => ir::StmtKind::Return(expr.map(|x| self.expr(x))),
			StmtKind::StaticAssert(assert) => ir::StmtKind::StaticAssert(self.expr(assert.expr)),
			StmtKind::Switch(switch) => ir::StmtKind::Switch(ir::Switch {
				expr: self.expr(switch.expr),
				cases: switch
					.cases
					.into_iter()
					.map(|case| ir::Case {
						selectors: case
							.selectors
							.into_iter()
							.map(|sel| match sel {
								ast::CaseSelector::Expr(expr) => ir::CaseSelector::Expr(self.expr(expr)),
								ast::CaseSelector::Default => ir::CaseSelector::Default,
							})
							.collect(),
						block: {
							self.scopes.push(FxHashMap::default());
							let ret = self.block(case.block);
							self.scopes.pop();
							ret
						},
						span: case.span,
					})
					.collect(),
			}),
			StmtKind::While(while_) => ir::StmtKind::While(ir::While {
				cond: self.expr(while_.cond),
				block: {
					self.scopes.push(FxHashMap::default());
					let ret = self.block(while_.block);
					self.scopes.pop();
					ret
				},
			}),
			StmtKind::Continuing(c) => {
				if !self.in_loop {
					self.diagnostics
						.push(stmt.span.error("`continuing` must be inside a `loop`") + stmt.span.marker());
				}
				self.in_loop = false;
				self.in_continuing = true;
				self.scopes.push(FxHashMap::default());
				let block = self.block(c);
				self.scopes.pop();
				self.in_continuing = false;
				ir::StmtKind::Continuing(block)
			},
			StmtKind::BreakIf(expr) => {
				if !self.in_continuing {
					self.diagnostics
						.push(stmt.span.error("`break if` must be inside a `continuing`") + stmt.span.marker());
				}
				ir::StmtKind::BreakIf(self.expr(expr))
			},
			StmtKind::Empty => return None,
		};

		Some(ir::Stmt { kind, span: stmt.span })
	}

	fn expr(&mut self, expr: ast::Expr) -> ir::Expr {
		let kind = match expr.kind {
			ExprKind::Underscore => {
				self.diagnostics
					.push(expr.span.error("cannot use `_` as an expression") + expr.span.marker());
				ir::ExprKind::Error
			},
			ExprKind::VarDecl(_) => {
				self.diagnostics
					.push(expr.span.error("cannot use variable declaration as an expression") + expr.span.marker());
				ir::ExprKind::Error
			},
			ExprKind::Literal(l) => ir::ExprKind::Literal(l),
			ExprKind::Ident(ident) => {
				self.verify_ident(ident.name);
				if ident.generics.len() != 0 {
					self.diagnostics
						.push(expr.span.error("generics not allowed here") + expr.span.marker());
				}
				self.resolve_access(ident.name)
			},
			ExprKind::Unary(u) => ir::ExprKind::Unary(ir::UnaryExpr {
				op: u.op,
				expr: Box::new(self.expr(*u.expr)),
			}),
			ExprKind::Binary(b) => ir::ExprKind::Binary(ir::BinaryExpr {
				op: b.op,
				lhs: Box::new(self.expr(*b.lhs)),
				rhs: Box::new(self.expr(*b.rhs)),
			}),
			ExprKind::Assign(_) => {
				self.diagnostics
					.push(expr.span.error("cannot use assignment as an expression") + expr.span.marker());
				ir::ExprKind::Error
			},
			ExprKind::Call(call) => {
				let target = self.call_target(*call.target);
				let args = call.args.into_iter().map(|x| self.expr(x)).collect();
				ir::ExprKind::Call(ir::CallExpr { target, args })
			},
			ExprKind::Index(on, with) => ir::ExprKind::Index(Box::new(self.expr(*on)), Box::new(self.expr(*with))),
			ExprKind::Member(on, member) => ir::ExprKind::Member(Box::new(self.expr(*on)), member),
			ExprKind::Postfix(_) => {
				self.diagnostics
					.push(expr.span.error("cannot use postfix statement as an expression") + expr.span.marker());
				ir::ExprKind::Error
			},
		};

		ir::Expr { kind, span: expr.span }
	}

	fn expr_statement(&mut self, expr: ast::Expr) -> Option<ir::ExprStatement> {
		let kind = match expr.kind {
			ExprKind::VarDecl(decl) => {
				let (name, kind) = match *decl {
					VarDecl::Var(v) => (v.name, ir::VarDeclKind::Var(self.var(v))),
					VarDecl::Const(c) => (c.name, ir::VarDeclKind::Const(self.let_(c))),
					VarDecl::Let(l) => (l.name, ir::VarDeclKind::Let(self.let_(l))),
				};

				ir::ExprStatementKind::VarDecl(ir::VarDecl {
					kind,
					local: {
						let id = LocalId(self.locals);
						self.locals += 1;
						let old = self
							.scopes
							.last_mut()
							.expect("no scopes")
							.insert(name.name, (id, name.span));

						if let Some((_, span)) = old {
							self.diagnostics.push(
								name.span.error("shadowing is not allowed in the same scope")
									+ span.label("previous declaration"),
							);
						}

						id
					},
				})
			},
			ExprKind::Call(call) => {
				let target = self.call_target(*call.target);
				let args = call.args.into_iter().map(|x| self.expr(x)).collect();
				ir::ExprStatementKind::Call(ir::CallExpr { target, args })
			},
			ExprKind::Assign(assign) => {
				if let ExprKind::Underscore = &assign.lhs.kind {
					if let ast::AssignOp::Assign = assign.op {
						ir::ExprStatementKind::IgnoreExpr(self.expr(*assign.rhs))
					} else {
						self.diagnostics
							.push(assign.lhs.span.error("`_` is not allowed here") + assign.lhs.span.marker());
						return None;
					}
				} else {
					let lhs = Box::new(self.expr(*assign.lhs));
					let rhs = Box::new(self.expr(*assign.rhs));
					ir::ExprStatementKind::Assign(ir::AssignExpr {
						lhs,
						rhs,
						op: assign.op,
					})
				}
			},
			ExprKind::Postfix(postfix) => {
				let expr = Box::new(self.expr(*postfix.expr));
				ir::ExprStatementKind::Postfix(ir::PostfixExpr { expr, op: postfix.op })
			},
			_ => {
				self.diagnostics
					.push(expr.span.error("this expression is not allowed here") + expr.span.marker());
				return None;
			},
		};

		Some(ir::ExprStatement { kind, span: expr.span })
	}

	fn call_target(&mut self, target: ast::Expr) -> FnTarget {
		match target.kind {
			ExprKind::Ident(ident) => {
				let name = ident.name;

				if let Some(decl) = self.index.get(name.name) {
					FnTarget::Decl(decl)
				} else if let Some(ty) = self.constructible_inbuilt(ident) {
					FnTarget::InbuiltType(Box::new(ty))
				} else if let Some(inbuilt) = self.inbuilt_function(name) {
					FnTarget::InbuiltFunction(inbuilt)
				} else {
					self.diagnostics
						.push(name.span.error("undefined function") + name.span.marker());
					FnTarget::Error
				}
			},
			_ => {
				self.diagnostics
					.push(target.span.error("invalid function call target") + target.span.marker());
				FnTarget::Error
			},
		}
	}

	fn constructible_inbuilt(&mut self, ident: ast::IdentExpr) -> Option<InbuiltType> {
		let vec = |this: &mut Self, name: &str, comp: VecType, generics: Vec<ast::Type>| {
			if generics.len() > 1 {
				this.diagnostics.push(
					ident.name.span.error(format!("too many generics for `{}`", name)) + ident.name.span.marker(),
				);
			}
			let ty = generics.into_iter().next();
			let of = ty.map(|x| (x.span, this.inbuilt(x)));
			match of {
				Some((_, Some(InbuiltType::Primitive(ty)))) => InbuiltType::Vec { ty, comp },
				Some((span, _)) => {
					this.diagnostics
						.push(span.error(format!("`{}` requires primitive type", name)) + span.marker());
					InbuiltType::Vec {
						ty: PrimitiveType::Infer,
						comp,
					}
				},
				None => InbuiltType::Vec {
					ty: PrimitiveType::Infer,
					comp,
				},
			}
		};

		let mat = |this: &mut Self, name: &str, comp: MatType, generics: Vec<ast::Type>| {
			if generics.len() > 1 {
				this.diagnostics.push(
					ident.name.span.error(format!("too many generics for `{}`", name)) + ident.name.span.marker(),
				);
			}
			let ty = generics.into_iter().next();
			let of = ty.map(|x| (x.span, this.inbuilt(x)));
			match of {
				Some((_, Some(InbuiltType::Primitive(ty)))) => {
					let ty = if let PrimitiveType::F16 = ty {
						FloatType::F16
					} else if let PrimitiveType::F32 = ty {
						FloatType::F32
					} else {
						this.diagnostics.push(
							ident
								.name
								.span
								.error(format!("`{}` requires floating point type", name))
								+ ident.name.span.marker(),
						);
						FloatType::Infer
					};
					InbuiltType::Mat { ty, comp }
				},
				Some((span, _)) => {
					this.diagnostics
						.push(span.error(format!("`{}` requires primitive type", name)) + span.marker());
					InbuiltType::Mat {
						ty: FloatType::Infer,
						comp,
					}
				},
				None => InbuiltType::Mat {
					ty: FloatType::Infer,
					comp,
				},
			}
		};

		Some(match ident.name.name {
			x if x == self.ty_kws.bool => InbuiltType::Primitive(PrimitiveType::Bool),
			x if x == self.ty_kws.f64 => {
				if !self.tu.features.float64 {
					self.diagnostics
						.push(ident.name.span.error("feature `f64` not enabled") + ident.name.span.marker());
				}
				InbuiltType::Primitive(PrimitiveType::F64)
			},
			x if x == self.ty_kws.f32 => InbuiltType::Primitive(PrimitiveType::F32),
			x if x == self.ty_kws.f16 => {
				if !self.tu.features.float16 {
					self.diagnostics
						.push(ident.name.span.error("feature `f16` not enabled") + ident.name.span.marker());
				}
				InbuiltType::Primitive(PrimitiveType::F16)
			},
			x if x == self.ty_kws.i32 => InbuiltType::Primitive(PrimitiveType::I32),
			x if x == self.ty_kws.u32 => InbuiltType::Primitive(PrimitiveType::U32),
			x if x == self.ty_kws.array => {
				if ident.generics.len() > 1 {
					self.diagnostics
						.push(ident.name.span.error("too many generics for `array`") + ident.name.span.marker());
				}
				let of = ident
					.generics
					.into_iter()
					.next()
					.map(|x| self.ty(x))
					.unwrap_or(ir::Type {
						kind: ir::TypeKind::Inbuilt(InbuiltType::Primitive(PrimitiveType::Infer)),
						span: ident.name.span,
					});
				let len = ident.array_len.map(|x| self.expr(*x));
				InbuiltType::Array { of: Box::new(of), len }
			},
			x if x == self.ty_kws.vec2 => vec(self, "vec2", VecType::Vec2, ident.generics),
			x if x == self.ty_kws.vec3 => vec(self, "vec3", VecType::Vec3, ident.generics),
			x if x == self.ty_kws.vec4 => vec(self, "vec4", VecType::Vec4, ident.generics),
			x if x == self.ty_kws.mat2x2 => mat(self, "mat2x2", MatType::Mat2x2, ident.generics),
			x if x == self.ty_kws.mat2x3 => mat(self, "mat2x3", MatType::Mat2x3, ident.generics),
			x if x == self.ty_kws.mat2x4 => mat(self, "mat2x4", MatType::Mat2x4, ident.generics),
			x if x == self.ty_kws.mat3x2 => mat(self, "mat3x2", MatType::Mat3x2, ident.generics),
			x if x == self.ty_kws.mat3x3 => mat(self, "mat3x3", MatType::Mat3x3, ident.generics),
			x if x == self.ty_kws.mat3x4 => mat(self, "mat3x4", MatType::Mat3x4, ident.generics),
			x if x == self.ty_kws.mat4x2 => mat(self, "mat4x2", MatType::Mat4x2, ident.generics),
			x if x == self.ty_kws.mat4x3 => mat(self, "mat4x3", MatType::Mat4x3, ident.generics),
			x if x == self.ty_kws.mat4x4 => mat(self, "mat4x4", MatType::Mat4x4, ident.generics),
			_ => return None,
		})
	}

	fn inbuilt(&mut self, ty: ast::Type) -> Option<InbuiltType> {
		let no_generics = |this: &mut Self, generics: Vec<ast::Type>, name: &str| {
			if generics.len() != 0 {
				this.diagnostics
					.push(ty.span.error(format!("`{}` cannot have generic parameters", name)) + ty.span.marker());
			}
		};
		let mat = |this: &mut Self, generics: Vec<ast::Type>, name: &str, comp: MatType| {
			if generics.len() != 1 {
				this.diagnostics.push(
					ty.span
						.error(format!("`{}` must have exactly 1 generic parameter", name))
						+ ty.span.marker(),
				);
			}

			if let Some(InbuiltType::Primitive(x)) = generics.into_iter().next().map(|x| this.inbuilt(x)).flatten() {
				let ty = match x {
					PrimitiveType::F16 => FloatType::F16,
					PrimitiveType::F32 => FloatType::F32,
					PrimitiveType::F64 => FloatType::F64,
					_ => {
						this.diagnostics.push(
							ty.span.error(format!(
								"`{}` must have a floating-point type as its generic parameter",
								name
							)) + ty.span.marker(),
						);
						FloatType::F32
					},
				};

				InbuiltType::Mat { comp, ty }
			} else {
				this.diagnostics.push(
					ty.span.error(format!(
						"`{}` must have a floating-point type as its generic parameter",
						name
					)) + ty.span.marker(),
				);
				InbuiltType::Mat {
					comp,
					ty: FloatType::F32,
				}
			}
		};
		let vec = |this: &mut Self, generics: Vec<ast::Type>, name: &str, comp: VecType| {
			if generics.len() != 1 {
				this.diagnostics.push(
					ty.span
						.error(format!("`{}` must have exactly 1 generic parameter", name))
						+ ty.span.marker(),
				);
			}

			if let Some(InbuiltType::Primitive(ty)) = generics.into_iter().next().map(|x| this.inbuilt(x)).flatten() {
				InbuiltType::Vec { comp, ty }
			} else {
				this.diagnostics.push(
					ty.span
						.error(format!("`{}` must have a scalar type as its generic parameter", name))
						+ ty.span.marker(),
				);
				InbuiltType::Vec {
					comp,
					ty: PrimitiveType::F32,
				}
			}
		};
		let sampled_texture = |this: &mut Self, generics: Vec<ast::Type>, name: &str, s: SampledTextureType| {
			if generics.len() != 1 {
				this.diagnostics.push(
					ty.span
						.error(format!("`{}` must have exactly 1 generic parameter", name))
						+ ty.span.marker(),
				);
			}

			let sample_type = generics.into_iter().next().and_then(|x| this.inbuilt(x));
			let sample_type = match sample_type {
				Some(InbuiltType::Primitive(PrimitiveType::F32)) => SampleType::F32,
				Some(InbuiltType::Primitive(PrimitiveType::U32)) => SampleType::U32,
				Some(InbuiltType::Primitive(PrimitiveType::I32)) => SampleType::I32,
				Some(_) => {
					this.diagnostics.push(
						ty.span.error(format!(
							"`{}` must have either `f32`, `i32`, or `u32` as its generic parameter",
							name
						)) + ty.span.marker(),
					);
					SampleType::F32
				},
				None => SampleType::F32,
			};

			InbuiltType::SampledTexture(s, sample_type)
		};
		let storage_texture = |this: &mut Self, generics: Vec<ast::Type>, name: &str, s: StorageTextureType| {
			if generics.len() != 2 {
				this.diagnostics.push(
					ty.span
						.error(format!("`{}` must have exactly 2 generic parameters", name))
						+ ty.span.marker(),
				);
			}

			let mut generics = generics.into_iter();
			let texel_format = generics
				.next()
				.and_then(|x| this.ty_to_ident(x, "texel format"))
				.and_then(|x| this.texel_format(x))
				.unwrap_or(TexelFormat::Rgba8Unorm);
			let access = generics
				.next()
				.and_then(|x| this.ty_to_ident(x, "access mode"))
				.map(|x| (this.access_mode(x), x.span));

			let access = if let Some((access, span)) = access {
				let access = access.unwrap_or(AccessMode::Write);
				if access != AccessMode::Write && !this.tu.features.storage_image_other_access {
					this.diagnostics
						.push(span.error("feature `storage_image_other_access` is not enabled") + span.marker());
				}
				access
			} else {
				AccessMode::Write
			};

			InbuiltType::StorageTexture(s, texel_format, access)
		};

		let ty = match ty.kind {
			ast::TypeKind::Ident(ident, generics) => match ident.name {
				x if x == self.ty_kws.bool => {
					no_generics(self, generics, "bool");
					InbuiltType::Primitive(PrimitiveType::Bool)
				},
				x if x == self.ty_kws.f64 => {
					if !self.tu.features.float64 {
						self.diagnostics
							.push(ident.span.error("feature `f64` is not enabled") + ident.span.marker());
					}

					no_generics(self, generics, "f64");
					InbuiltType::Primitive(PrimitiveType::F64)
				},
				x if x == self.ty_kws.f32 => {
					no_generics(self, generics, "f32");
					InbuiltType::Primitive(PrimitiveType::F32)
				},
				x if x == self.ty_kws.f16 => {
					if !self.tu.features.float16 {
						self.diagnostics
							.push(ident.span.error("feature `float16` is not enabled") + ident.span.marker());
					}
					no_generics(self, generics, "f16");
					InbuiltType::Primitive(PrimitiveType::F16)
				},
				x if x == self.ty_kws.i32 => {
					no_generics(self, generics, "i32");
					InbuiltType::Primitive(PrimitiveType::I32)
				},
				x if x == self.ty_kws.u32 => {
					no_generics(self, generics, "u32");
					InbuiltType::Primitive(PrimitiveType::U32)
				},
				x if x == self.ty_kws.mat2x2 => mat(self, generics, "mat2x2", MatType::Mat2x2),
				x if x == self.ty_kws.mat2x3 => mat(self, generics, "mat2x3", MatType::Mat2x3),
				x if x == self.ty_kws.mat2x4 => mat(self, generics, "mat2x4", MatType::Mat2x4),
				x if x == self.ty_kws.mat3x2 => mat(self, generics, "mat3x2", MatType::Mat3x2),
				x if x == self.ty_kws.mat3x3 => mat(self, generics, "mat3x3", MatType::Mat3x3),
				x if x == self.ty_kws.mat3x4 => mat(self, generics, "mat3x4", MatType::Mat3x4),
				x if x == self.ty_kws.mat4x2 => mat(self, generics, "mat4x2", MatType::Mat4x2),
				x if x == self.ty_kws.mat4x3 => mat(self, generics, "mat4x3", MatType::Mat4x3),
				x if x == self.ty_kws.mat4x4 => mat(self, generics, "mat4x4", MatType::Mat4x4),
				x if x == self.ty_kws.vec2 => vec(self, generics, "vec2", VecType::Vec2),
				x if x == self.ty_kws.vec3 => vec(self, generics, "vec3", VecType::Vec3),
				x if x == self.ty_kws.vec4 => vec(self, generics, "vec4", VecType::Vec4),
				x if x == self.ty_kws.atomic => {
					if generics.len() != 1 {
						self.diagnostics.push(
							ty.span
								.error(format!("`atomic` must have exactly one generic parameter"))
								+ ty.span.marker(),
						);
					}

					if let Some(InbuiltType::Primitive(p)) =
						generics.into_iter().next().map(|x| self.inbuilt(x)).flatten()
					{
						if let PrimitiveType::U32 = p {
							InbuiltType::Atomic { signed: false }
						} else if let PrimitiveType::I32 = p {
							InbuiltType::Atomic { signed: true }
						} else {
							self.diagnostics.push(
								ty.span
									.error(format!("`atomic` must have an integer type as its generic parameter",))
									+ ty.span.marker(),
							);
							InbuiltType::Atomic { signed: true }
						}
					} else {
						self.diagnostics.push(
							ty.span
								.error(format!("`atomic` must have an integer type as its generic parameter",))
								+ ty.span.marker(),
						);
						InbuiltType::Atomic { signed: true }
					}
				},
				x if x == self.ty_kws.ptr => {
					let mut generics = generics.into_iter();
					let address_space = generics.next();
					let to = generics.next().map(|x| self.ty(x));
					let access_mode = generics.next();

					let address_space = if let Some(address_space) = address_space {
						self.ty_to_ident(address_space, "address space")
							.and_then(|x| self.address_space(x))
							.unwrap_or(AddressSpace::Private)
					} else {
						AddressSpace::Private
					};
					let to = if let Some(to) = to {
						to
					} else {
						self.diagnostics
							.push(ty.span.error(format!("expected type")) + ty.span.marker());
						ir::Type {
							kind: ir::TypeKind::Inbuilt(InbuiltType::Primitive(PrimitiveType::I32)),
							span: ty.span,
						}
					};
					let access_mode = access_mode
						.and_then(|access_mode| {
							if address_space == AddressSpace::Storage {
								self.ty_to_ident(access_mode, "access mode")
									.and_then(|x| self.access_mode(x))
							} else {
								self.diagnostics.push(
									access_mode
										.span
										.error("access mode is not allowed for this address space")
										+ access_mode.span.marker(),
								);
								None
							}
						})
						.unwrap_or(match address_space {
							AddressSpace::Function => AccessMode::ReadWrite,
							AddressSpace::Private => AccessMode::ReadWrite,
							AddressSpace::Storage => AccessMode::Read,
							AddressSpace::Uniform => AccessMode::Read,
							AddressSpace::Workgroup => AccessMode::ReadWrite,
							AddressSpace::PushConstant => AccessMode::Read,
						});

					InbuiltType::Ptr {
						address_space,
						to: Box::new(to),
						access_mode,
					}
				},
				x if x == self.ty_kws.texture_1d => {
					sampled_texture(self, generics, "texture_1d", SampledTextureType::Texture1d)
				},
				x if x == self.ty_kws.texture_1d_array => {
					sampled_texture(self, generics, "texture_1d_array", SampledTextureType::Texture1dArray)
				},
				x if x == self.ty_kws.texture_2d => {
					sampled_texture(self, generics, "texture_2d", SampledTextureType::Texture2d)
				},
				x if x == self.ty_kws.texture_multisampled_2d => sampled_texture(
					self,
					generics,
					"texture_multisampled_2d",
					SampledTextureType::TextureMultisampled2d,
				),
				x if x == self.ty_kws.texture_2d_array => {
					sampled_texture(self, generics, "texture_2d_array", SampledTextureType::Texture2dArray)
				},
				x if x == self.ty_kws.texture_3d => {
					sampled_texture(self, generics, "texture_3d", SampledTextureType::Texture3d)
				},
				x if x == self.ty_kws.texture_cube => {
					sampled_texture(self, generics, "texture_cube", SampledTextureType::TextureCube)
				},
				x if x == self.ty_kws.texture_cube_array => sampled_texture(
					self,
					generics,
					"texture_cube_array",
					SampledTextureType::TextureCubeArray,
				),
				x if x == self.ty_kws.texture_depth_2d => {
					no_generics(self, generics, "texture_depth_2d");
					InbuiltType::DepthTexture(DepthTextureType::Depth2d)
				},
				x if x == self.ty_kws.texture_depth_2d_array => {
					no_generics(self, generics, "texture_depth_2d_array");
					InbuiltType::DepthTexture(DepthTextureType::Depth2dArray)
				},
				x if x == self.ty_kws.texture_depth_cube => {
					no_generics(self, generics, "texture_depth_cube");
					InbuiltType::DepthTexture(DepthTextureType::DepthCube)
				},
				x if x == self.ty_kws.texture_depth_cube_array => {
					no_generics(self, generics, "texture_depth_cube_array");
					InbuiltType::DepthTexture(DepthTextureType::DepthCubeArray)
				},
				x if x == self.ty_kws.texture_depth_multisampled_2d => {
					no_generics(self, generics, "texture_depth_multisampled_2d");
					InbuiltType::DepthTexture(DepthTextureType::DepthMultisampled2d)
				},
				x if x == self.ty_kws.texture_storage_1d => {
					storage_texture(self, generics, "texture_storage_1d", StorageTextureType::Storage1d)
				},
				x if x == self.ty_kws.texture_storage_1d_array => storage_texture(
					self,
					generics,
					"texture_storage_1d_array",
					StorageTextureType::Storage1dArray,
				),
				x if x == self.ty_kws.texture_storage_2d => {
					storage_texture(self, generics, "texture_storage_2d", StorageTextureType::Storage2d)
				},
				x if x == self.ty_kws.texture_storage_2d_array => storage_texture(
					self,
					generics,
					"texture_storage_2d_array",
					StorageTextureType::Storage2dArray,
				),
				x if x == self.ty_kws.texture_storage_3d => {
					storage_texture(self, generics, "texture_storage_3d", StorageTextureType::Storage3d)
				},
				x if x == self.ty_kws.sampler => {
					no_generics(self, generics, "sampler");
					InbuiltType::Sampler(SamplerType::Sampler)
				},
				x if x == self.ty_kws.sampler_comparison => {
					no_generics(self, generics, "sampler_comparison");
					InbuiltType::Sampler(SamplerType::SamplerComparison)
				},
				_ => return None,
			},
			ast::TypeKind::Array(array, of, len) => {
				if array.name == self.ty_kws.array {
					InbuiltType::Array {
						of: Box::new(self.ty(*of)),
						len: len.map(|x| self.expr(x)),
					}
				} else {
					// Is `binding_array`
					if !self.tu.features.binding_array {
						self.diagnostics
							.push(array.span.error("feature `binding_array` is not enabled") + array.span.marker());
					}

					InbuiltType::BindingArray {
						of: Box::new(self.ty(*of)),
						len: len.map(|x| self.expr(x)),
					}
				}
			},
		};

		Some(ty)
	}

	fn attrib(&mut self, attrib: ast::Attribute) -> Option<ir::Attribute> {
		let args = |this: &mut Self, args: usize| {
			if attrib.exprs.len() != args {
				this.diagnostics
					.push(attrib.span.error("expected 1 argument") + attrib.span.marker());
				None
			} else {
				Some(())
			}
		};
		let expr_as_ident = |this: &mut Self, expr: &ast::Expr| match &expr.kind {
			ExprKind::Ident(i) => {
				if i.generics.len() != 0 || i.array_len.is_some() {
					this.diagnostics
						.push(expr.span.error("expected identifier") + expr.span.marker());
				}
				Some(i.name)
			},
			_ => None,
		};

		let ty = match attrib.name.name {
			x if x == self.attrib_kws.align => args(self, 1).map(|_| AttributeType::Align(attrib.exprs[0].clone())),
			x if x == self.attrib_kws.binding => args(self, 1).map(|_| AttributeType::Binding(attrib.exprs[0].clone())),
			x if x == self.attrib_kws.builtin => args(self, 1).and_then(|_| {
				expr_as_ident(self, &attrib.exprs[0])
					.and_then(|ident| self.builtin(ident).map(|x| AttributeType::Builtin(x)))
			}),
			x if x == self.attrib_kws.compute => args(self, 0).map(|_| AttributeType::Compute),
			x if x == self.attrib_kws.const_ => args(self, 0).map(|_| AttributeType::Const),
			x if x == self.attrib_kws.fragment => args(self, 0).map(|_| AttributeType::Fragment),
			x if x == self.attrib_kws.group => args(self, 1).map(|_| AttributeType::Group(attrib.exprs[0].clone())),
			x if x == self.attrib_kws.id => args(self, 1).map(|_| AttributeType::Id(attrib.exprs[0].clone())),
			x if x == self.attrib_kws.interpolate => {
				if attrib.exprs.len() < 1 || attrib.exprs.len() > 2 {
					self.diagnostics
						.push(attrib.span.error("expected 1 or 2 arguments") + attrib.span.marker());
					None
				} else {
					let ty = expr_as_ident(self, &attrib.exprs[0])
						.and_then(|x| self.interpolation_type(x))
						.unwrap_or(InterpolationType::Perspective);
					let sample = attrib
						.exprs
						.get(1)
						.and_then(|x| expr_as_ident(self, x))
						.and_then(|x| self.interpolation_sample(x));

					if ty == InterpolationType::Flat && sample.is_some() {
						let span = attrib.exprs[1].span;
						self.diagnostics
							.push(span.error("flat interpolation must not have a sample type") + span.marker());
					}

					Some(AttributeType::Interpolate(
						ty,
						sample.unwrap_or(InterpolationSample::Center),
					))
				}
			},
			x if x == self.attrib_kws.invariant => args(self, 0).map(|_| AttributeType::Invariant),
			x if x == self.attrib_kws.location => {
				args(self, 1).map(|_| AttributeType::Location(attrib.exprs[0].clone()))
			},
			x if x == self.attrib_kws.size => args(self, 1).map(|_| AttributeType::Size(attrib.exprs[0].clone())),
			x if x == self.attrib_kws.vertex => args(self, 0).map(|_| AttributeType::Vertex),
			x if x == self.attrib_kws.workgroup_size => {
				if attrib.exprs.len() < 1 || attrib.exprs.len() > 3 {
					self.diagnostics
						.push(attrib.span.error("expected 1, 2, or 3 arguments") + attrib.span.marker());
					None
				} else {
					let x = attrib.exprs[0].clone();
					let y = attrib.exprs.get(1).cloned();
					let z = attrib.exprs.get(2).cloned();
					Some(AttributeType::WorkgroupSize(x, y, z))
				}
			},
			_ => {
				self.diagnostics
					.push(attrib.name.span.error("unknown attribute") + attrib.name.span.marker());
				None
			},
		};

		ty.map(|ty| ir::Attribute { span: attrib.span, ty })
	}

	fn builtin(&mut self, ident: Ident) -> Option<Builtin> {
		Some(match ident.name {
			x if x == self.attrib_kws.frag_depth => Builtin::FragDepth,
			x if x == self.attrib_kws.front_facing => Builtin::FrontFacing,
			x if x == self.attrib_kws.global_invocation_id => Builtin::GlobalInvocationId,
			x if x == self.attrib_kws.instance_index => Builtin::InstanceIndex,
			x if x == self.attrib_kws.local_invocation_id => Builtin::LocalInvocationId,
			x if x == self.attrib_kws.local_invocation_index => Builtin::LocalInvocationIndex,
			x if x == self.attrib_kws.num_workgroups => Builtin::NumWorkgroups,
			x if x == self.attrib_kws.position => Builtin::Position,
			x if x == self.attrib_kws.sample_index => Builtin::SampleIndex,
			x if x == self.attrib_kws.sample_mask => Builtin::SampleMask,
			x if x == self.attrib_kws.vertex_index => Builtin::VertexIndex,
			x if x == self.attrib_kws.workgroup_id => Builtin::WorkgroupId,
			x if x == self.attrib_kws.primitive_index => {
				if !self.tu.features.primitive_index {
					self.diagnostics
						.push(ident.span.error("feature `primitive_index` is not enabled") + ident.span.marker());
				}
				Builtin::PrimitiveIndex
			},
			x if x == self.attrib_kws.view_index => {
				if !self.tu.features.multiview {
					self.diagnostics
						.push(ident.span.error("feature `multiview` is not enabled") + ident.span.marker());
				}
				Builtin::ViewIndex
			},
			_ => {
				self.diagnostics
					.push(ident.span.error("unknown builtin") + ident.span.marker());
				return None;
			},
		})
	}

	fn interpolation_type(&mut self, ident: Ident) -> Option<InterpolationType> {
		Some(match ident.name {
			x if x == self.attrib_kws.flat => InterpolationType::Flat,
			x if x == self.attrib_kws.linear => InterpolationType::Linear,
			x if x == self.attrib_kws.perspective => InterpolationType::Perspective,
			_ => {
				self.diagnostics
					.push(ident.span.error("unknown interpolation type") + ident.span.marker());
				return None;
			},
		})
	}

	fn interpolation_sample(&mut self, ident: Ident) -> Option<InterpolationSample> {
		Some(match ident.name {
			x if x == self.attrib_kws.center => InterpolationSample::Center,
			x if x == self.attrib_kws.centroid => InterpolationSample::Centroid,
			x if x == self.attrib_kws.sample => InterpolationSample::Sample,
			_ => {
				self.diagnostics
					.push(ident.span.error("unknown interpolation sample") + ident.span.marker());
				return None;
			},
		})
	}

	fn address_space(&mut self, ident: Ident) -> Option<AddressSpace> {
		Some(match ident.name {
			x if x == self.ty_kws.uniform => AddressSpace::Uniform,
			x if x == self.ty_kws.storage => AddressSpace::Storage,
			x if x == self.ty_kws.private => AddressSpace::Private,
			x if x == self.ty_kws.workgroup => AddressSpace::Workgroup,
			x if x == self.ty_kws.function => AddressSpace::Function,
			x if x == self.ty_kws.push_constant => {
				if !self.tu.features.push_constant {
					self.diagnostics
						.push(ident.span.error("feature `push_constant` is not enabled") + ident.span.marker());
				}
				AddressSpace::PushConstant
			},
			_ => {
				self.diagnostics
					.push(ident.span.error(format!("expected address space")) + ident.span.marker());
				return None;
			},
		})
	}

	fn access_mode(&mut self, ident: Ident) -> Option<AccessMode> {
		Some(match ident.name {
			x if x == self.ty_kws.read => AccessMode::Read,
			x if x == self.ty_kws.write => AccessMode::Write,
			x if x == self.ty_kws.read_write => AccessMode::ReadWrite,
			_ => {
				self.diagnostics
					.push(ident.span.error(format!("expected access mode")) + ident.span.marker());
				return None;
			},
		})
	}

	fn ty_to_ident(&mut self, ty: ast::Type, expected: &str) -> Option<Ident> {
		match &ty.kind {
			ast::TypeKind::Ident(ident, generics) => {
				if generics.len() != 0 {
					self.diagnostics
						.push(ty.span.error(format!("expected {}", expected)) + ty.span.marker());
				}
				Some(*ident)
			},
			ast::TypeKind::Array(..) => {
				self.diagnostics
					.push(ty.span.error(format!("expected {}", expected)) + ty.span.marker());
				None
			},
		}
	}

	fn texel_format(&mut self, ident: Ident) -> Option<TexelFormat> {
		Some(match ident.name {
			x if x == self.ty_kws.r32float => TexelFormat::R32Float,
			x if x == self.ty_kws.r32sint => TexelFormat::R32Sint,
			x if x == self.ty_kws.r32uint => TexelFormat::R32Uint,
			x if x == self.ty_kws.rg32float => TexelFormat::Rg32Float,
			x if x == self.ty_kws.rg32sint => TexelFormat::Rg32Sint,
			x if x == self.ty_kws.rg32uint => TexelFormat::Rg32Uint,
			x if x == self.ty_kws.rgba16float => TexelFormat::Rgba16Float,
			x if x == self.ty_kws.rgba16sint => TexelFormat::Rgba16Sint,
			x if x == self.ty_kws.rgba16uint => TexelFormat::Rgba16Uint,
			x if x == self.ty_kws.rgba32float => TexelFormat::Rgba32Float,
			x if x == self.ty_kws.rgba32sint => TexelFormat::Rgba32Sint,
			x if x == self.ty_kws.rgba32uint => TexelFormat::Rgba32Uint,
			x if x == self.ty_kws.rgba8sint => TexelFormat::Rgba8Sint,
			x if x == self.ty_kws.rgba8uint => TexelFormat::Rgba8Uint,
			x if x == self.ty_kws.rgba8unorm => TexelFormat::Rgba8Unorm,
			x if x == self.ty_kws.rgba8snorm => TexelFormat::Rgba8Snorm,
			_ => {
				self.diagnostics
					.push(ident.span.error("expected texel format") + ident.span.marker());
				return None;
			},
		})
	}

	fn inbuilt_function(&mut self, ident: Ident) -> Option<InbuiltFunction> {
		use InbuiltFunction::*;

		Some(match ident.name {
			x if x == self.fn_kws.bitcast => Bitcast,
			x if x == self.fn_kws.all => All,
			x if x == self.fn_kws.any => Any,
			x if x == self.fn_kws.select => Select,
			x if x == self.fn_kws.array_length => ArrayLength,
			x if x == self.fn_kws.abs => Abs,
			x if x == self.fn_kws.acos => Acos,
			x if x == self.fn_kws.acosh => Acosh,
			x if x == self.fn_kws.asin => Asin,
			x if x == self.fn_kws.asinh => Asinh,
			x if x == self.fn_kws.atan => Atan,
			x if x == self.fn_kws.atanh => Atanh,
			x if x == self.fn_kws.atan2 => Atan2,
			x if x == self.fn_kws.ceil => Ceil,
			x if x == self.fn_kws.clamp => Clamp,
			x if x == self.fn_kws.cos => Cos,
			x if x == self.fn_kws.cosh => Cosh,
			x if x == self.fn_kws.count_leading_zeros => CountLeadingZeros,
			x if x == self.fn_kws.count_one_bits => CountOneBits,
			x if x == self.fn_kws.count_trailing_zeros => CountTrailingZeros,
			x if x == self.fn_kws.cross => Cross,
			x if x == self.fn_kws.degrees => Degrees,
			x if x == self.fn_kws.determinant => Determinant,
			x if x == self.fn_kws.distance => Distance,
			x if x == self.fn_kws.dot => Dot,
			x if x == self.fn_kws.exp => Exp,
			x if x == self.fn_kws.exp2 => Exp2,
			x if x == self.fn_kws.extract_bits => ExtractBits,
			x if x == self.fn_kws.face_forward => FaceForward,
			x if x == self.fn_kws.first_leading_bit => FirstLeadingBit,
			x if x == self.fn_kws.first_trailing_bit => FirstTrailingBit,
			x if x == self.fn_kws.floor => Floor,
			x if x == self.fn_kws.fma => Fma,
			x if x == self.fn_kws.fract => Fract,
			x if x == self.fn_kws.frexp => Frexp,
			x if x == self.fn_kws.insert_bits => InsertBits,
			x if x == self.fn_kws.inverse_sqrt => InverseSqrt,
			x if x == self.fn_kws.ldexp => Ldexp,
			x if x == self.fn_kws.length => Length,
			x if x == self.fn_kws.log => Log,
			x if x == self.fn_kws.log2 => Log2,
			x if x == self.fn_kws.max => Max,
			x if x == self.fn_kws.min => Min,
			x if x == self.fn_kws.mix => Mix,
			x if x == self.fn_kws.modf => Modf,
			x if x == self.fn_kws.normalize => Normalize,
			x if x == self.fn_kws.pow => Pow,
			x if x == self.fn_kws.quantize_to_f16 => QuantizeToF16,
			x if x == self.fn_kws.radians => Radians,
			x if x == self.fn_kws.reflect => Reflect,
			x if x == self.fn_kws.refract => Refract,
			x if x == self.fn_kws.reverse_bits => ReverseBits,
			x if x == self.fn_kws.round => Round,
			x if x == self.fn_kws.saturate => Saturate,
			x if x == self.fn_kws.sign => Sign,
			x if x == self.fn_kws.sin => Sin,
			x if x == self.fn_kws.sinh => Sinh,
			x if x == self.fn_kws.smooth_step => Smoothstep,
			x if x == self.fn_kws.sqrt => Sqrt,
			x if x == self.fn_kws.step => Step,
			x if x == self.fn_kws.tan => Tan,
			x if x == self.fn_kws.tanh => Tanh,
			x if x == self.fn_kws.transpose => Transpose,
			x if x == self.fn_kws.trunc => Trunc,
			x if x == self.fn_kws.texture_sample => Dpdx,
			x if x == self.fn_kws.texture_sample_bias => DpdxCoarse,
			x if x == self.fn_kws.texture_sample_compare => DpdxFine,
			x if x == self.fn_kws.texture_sample_grad => Dpdy,
			x if x == self.fn_kws.texture_sample_level => DpdyCoarse,
			x if x == self.fn_kws.dpdx => DpdyFine,
			x if x == self.fn_kws.dpdx_coarse => Fwidth,
			x if x == self.fn_kws.dpdx_fine => FwidthCoarse,
			x if x == self.fn_kws.dpdy => FwidthFine,
			x if x == self.fn_kws.dpdy_coarse => TextureDimensions,
			x if x == self.fn_kws.dpdy_fine => TextureGather,
			x if x == self.fn_kws.fwidth => TextureGatherCompare,
			x if x == self.fn_kws.fwidth_coarse => TextureLoad,
			x if x == self.fn_kws.fwidth_fine => TextureNumLayers,
			x if x == self.fn_kws.texture_dimensions => TextuerNumLevels,
			x if x == self.fn_kws.texture_gather => TextureNumSamples,
			x if x == self.fn_kws.texture_gather_compare => TextureSample,
			x if x == self.fn_kws.texture_load => TextureSampleBias,
			x if x == self.fn_kws.texture_num_layers => TextureSampleCompare,
			x if x == self.fn_kws.texture_num_levels => TextureSampleCompareLevel,
			x if x == self.fn_kws.texture_num_samples => TextureSampleGrad,
			x if x == self.fn_kws.texture_sample_compare_level => TextureSampleLevel,
			x if x == self.fn_kws.texture_sample_base_clamp_to_edge => TextureSampleBaseClampToEdge,
			x if x == self.fn_kws.texture_store => TextureStore,
			x if x == self.fn_kws.atomic_load => AtomicLoad,
			x if x == self.fn_kws.atomic_store => AtomicStore,
			x if x == self.fn_kws.atomic_add => AtomicAdd,
			x if x == self.fn_kws.atomic_sub => AtomicSub,
			x if x == self.fn_kws.atomic_max => AtomicMax,
			x if x == self.fn_kws.atomic_min => AtomicMin,
			x if x == self.fn_kws.atomic_and => AtomicAnd,
			x if x == self.fn_kws.atomic_or => AtomicOr,
			x if x == self.fn_kws.atomic_xor => AtomicXor,
			x if x == self.fn_kws.atomic_exchange => AtomicExchange,
			x if x == self.fn_kws.atomic_compare_exchange_weak => AtomicCompareExchangeWeak,
			x if x == self.fn_kws.pack4x8snorm => Pack4x8Snorm,
			x if x == self.fn_kws.pack4x8unorm => Pack4x8Unorm,
			x if x == self.fn_kws.pack2x16snorm => Pack2x16Snorm,
			x if x == self.fn_kws.pack2x16unorm => Pack2x16Unorm,
			x if x == self.fn_kws.pack2x16float => Pack2x16Float,
			x if x == self.fn_kws.unpack4x8snorm => Unpack4x8Snorm,
			x if x == self.fn_kws.unpack4x8unorm => Unpack4x8Unorm,
			x if x == self.fn_kws.unpack2x16snorm => Unpack2x16Snorm,
			x if x == self.fn_kws.unpack2x16unorm => Unpack2x16Unorm,
			x if x == self.fn_kws.unpack2x16float => Unpack2x16Float,
			x if x == self.fn_kws.storage_barrier => StorageBarrier,
			x if x == self.fn_kws.workgroup_barrier => WorkgroupBarrier,
			_ => return None,
		})
	}

	fn verify_ident(&mut self, ident: Ident) {
		let text = self.intern.resolve(ident.name);
		if let Some(m) = self.reserved_matcher.earliest_find(text) {
			if m.len() == text.len() {
				self.diagnostics
					.push(ident.span.error("usage of reserved identifier") + ident.span.marker());
			}
		}
	}

	fn resolve_access(&mut self, ident: Ident) -> ir::ExprKind {
		for scope in self.scopes.iter().rev() {
			if let Some((id, _)) = scope.get(&ident.name) {
				return ir::ExprKind::Local(*id);
			}
		}

		if let Some(global) = self.index.get(ident.name) {
			ir::ExprKind::Global(global)
		} else {
			self.diagnostics
				.push(ident.span.error("undefined identifier") + ident.span.marker());
			ir::ExprKind::Error
		}
	}
}

struct TypeKws {
	array: Text,
	f64: Text,
	f32: Text,
	f16: Text,
	i32: Text,
	u32: Text,
	bool: Text,
	mat2x2: Text,
	mat2x3: Text,
	mat2x4: Text,
	mat3x2: Text,
	mat3x3: Text,
	mat3x4: Text,
	mat4x2: Text,
	mat4x3: Text,
	mat4x4: Text,
	vec2: Text,
	vec3: Text,
	vec4: Text,
	atomic: Text,
	ptr: Text,
	read: Text,
	write: Text,
	read_write: Text,
	function: Text,
	private: Text,
	storage: Text,
	uniform: Text,
	workgroup: Text,
	push_constant: Text,
	texture_1d: Text,
	texture_1d_array: Text,
	texture_2d: Text,
	texture_multisampled_2d: Text,
	texture_2d_array: Text,
	texture_3d: Text,
	texture_cube: Text,
	texture_cube_array: Text,
	texture_depth_2d: Text,
	texture_depth_2d_array: Text,
	texture_depth_cube: Text,
	texture_depth_cube_array: Text,
	texture_depth_multisampled_2d: Text,
	texture_storage_1d: Text,
	texture_storage_1d_array: Text,
	texture_storage_2d: Text,
	texture_storage_2d_array: Text,
	texture_storage_3d: Text,
	sampler: Text,
	sampler_comparison: Text,
	r32float: Text,
	r32sint: Text,
	r32uint: Text,
	rg32float: Text,
	rg32sint: Text,
	rg32uint: Text,
	rgba16float: Text,
	rgba16sint: Text,
	rgba16uint: Text,
	rgba32float: Text,
	rgba32sint: Text,
	rgba32uint: Text,
	rgba8sint: Text,
	rgba8uint: Text,
	rgba8unorm: Text,
	rgba8snorm: Text,
}

impl TypeKws {
	fn init(intern: &mut Interner) -> Self {
		Self {
			array: intern.get_static("array"),
			f64: intern.get_static("f64"),
			f32: intern.get_static("f32"),
			f16: intern.get_static("f16"),
			i32: intern.get_static("i32"),
			u32: intern.get_static("u32"),
			bool: intern.get_static("bool"),
			mat2x2: intern.get_static("mat2x2"),
			mat2x3: intern.get_static("mat2x3"),
			mat2x4: intern.get_static("mat2x4"),
			mat3x2: intern.get_static("mat3x2"),
			mat3x3: intern.get_static("mat3x3"),
			mat3x4: intern.get_static("mat3x4"),
			mat4x2: intern.get_static("mat4x2"),
			mat4x3: intern.get_static("mat4x3"),
			mat4x4: intern.get_static("mat4x4"),
			vec2: intern.get_static("vec2"),
			vec3: intern.get_static("vec3"),
			vec4: intern.get_static("vec4"),
			atomic: intern.get_static("atomic"),
			ptr: intern.get_static("ptr"),
			read: intern.get_static("read"),
			write: intern.get_static("write"),
			read_write: intern.get_static("read_write"),
			function: intern.get_static("function"),
			private: intern.get_static("private"),
			storage: intern.get_static("storage"),
			uniform: intern.get_static("uniform"),
			workgroup: intern.get_static("workgroup"),
			push_constant: intern.get_static("push_constant"),
			texture_1d: intern.get_static("texture_1d"),
			texture_1d_array: intern.get_static("texture_1d_array"),
			texture_2d: intern.get_static("texture_2d"),
			texture_multisampled_2d: intern.get_static("texture_multisampled_2d"),
			texture_2d_array: intern.get_static("texture_2d_array"),
			texture_3d: intern.get_static("texture_3d"),
			texture_cube: intern.get_static("texture_cube"),
			texture_cube_array: intern.get_static("texture_cube_array"),
			texture_depth_2d: intern.get_static("texture_depth_2d"),
			texture_depth_2d_array: intern.get_static("texture_depth_2d_array"),
			texture_depth_cube: intern.get_static("texture_depth_cube"),
			texture_depth_cube_array: intern.get_static("texture_depth_cube_array"),
			texture_depth_multisampled_2d: intern.get_static("texture_depth_multisampled_2d"),
			texture_storage_1d: intern.get_static("texture_storage_1d"),
			texture_storage_1d_array: intern.get_static("texture_storage_1d_array"),
			texture_storage_2d: intern.get_static("texture_storage_2d"),
			texture_storage_2d_array: intern.get_static("texture_storage_2d_array"),
			texture_storage_3d: intern.get_static("texture_storage_3d"),
			sampler: intern.get_static("sampler"),
			sampler_comparison: intern.get_static("sampler_comparison"),
			r32float: intern.get_static("r32float"),
			r32sint: intern.get_static("r32sint"),
			r32uint: intern.get_static("r32uint"),
			rg32float: intern.get_static("rg32float"),
			rg32sint: intern.get_static("rg32sint"),
			rg32uint: intern.get_static("rg32uint"),
			rgba16float: intern.get_static("rgba16float"),
			rgba16sint: intern.get_static("rgba16sint"),
			rgba16uint: intern.get_static("rgba16uint"),
			rgba32float: intern.get_static("rgba32float"),
			rgba32sint: intern.get_static("rgba32sint"),
			rgba32uint: intern.get_static("rgba32uint"),
			rgba8sint: intern.get_static("rgba8sint"),
			rgba8uint: intern.get_static("rgba8uint"),
			rgba8unorm: intern.get_static("rgba8unorm"),
			rgba8snorm: intern.get_static("rgba8snorm"),
		}
	}
}

struct AttributeKws {
	align: Text,
	binding: Text,
	builtin: Text,
	compute: Text,
	const_: Text,
	fragment: Text,
	group: Text,
	id: Text,
	interpolate: Text,
	invariant: Text,
	location: Text,
	size: Text,
	vertex: Text,
	workgroup_size: Text,
	frag_depth: Text,
	front_facing: Text,
	global_invocation_id: Text,
	instance_index: Text,
	local_invocation_id: Text,
	local_invocation_index: Text,
	num_workgroups: Text,
	position: Text,
	sample_index: Text,
	sample_mask: Text,
	vertex_index: Text,
	workgroup_id: Text,
	primitive_index: Text,
	view_index: Text,
	center: Text,
	centroid: Text,
	sample: Text,
	flat: Text,
	linear: Text,
	perspective: Text,
}

impl AttributeKws {
	fn init(intern: &mut Interner) -> Self {
		Self {
			align: intern.get_static("align"),
			binding: intern.get_static("binding"),
			builtin: intern.get_static("builtin"),
			compute: intern.get_static("compute"),
			const_: intern.get_static("const"),
			fragment: intern.get_static("fragment"),
			group: intern.get_static("group"),
			id: intern.get_static("id"),
			interpolate: intern.get_static("interpolate"),
			invariant: intern.get_static("invariant"),
			location: intern.get_static("location"),
			size: intern.get_static("size"),
			vertex: intern.get_static("vertex"),
			workgroup_size: intern.get_static("workgroup_size"),
			frag_depth: intern.get_static("frag_depth"),
			front_facing: intern.get_static("front_facing"),
			global_invocation_id: intern.get_static("global_invocation_id"),
			instance_index: intern.get_static("instance_index"),
			local_invocation_id: intern.get_static("local_invocation_id"),
			local_invocation_index: intern.get_static("local_invocation_index"),
			num_workgroups: intern.get_static("num_workgroups"),
			position: intern.get_static("position"),
			sample_index: intern.get_static("sample_index"),
			sample_mask: intern.get_static("sample_mask"),
			vertex_index: intern.get_static("vertex_index"),
			workgroup_id: intern.get_static("workgroup_id"),
			primitive_index: intern.get_static("primitive_index"),
			view_index: intern.get_static("view_index"),
			center: intern.get_static("center"),
			centroid: intern.get_static("centroid"),
			sample: intern.get_static("sample"),
			flat: intern.get_static("flat"),
			linear: intern.get_static("linear"),
			perspective: intern.get_static("perspective"),
		}
	}
}
