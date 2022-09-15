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
			Matcher,
			PrimitiveType,
			SampledTextureType,
			SamplerType,
			StorageTextureType,
			TexelFormat,
			ToStaticString,
			VecType,
		},
		inbuilt_functions::InbuiltFunction,
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
		kws: Box::new(Kws::init(intern)),
		access_mode: Matcher::new(intern),
		address_space: Matcher::new(intern),
		builtin: Matcher::new(intern),
		interpolation_sample: Matcher::new(intern),
		interpolation_type: Matcher::new(intern),
		primitive: Matcher::new(intern),
		vec: Matcher::new(intern),
		mat: Matcher::new(intern),
		sampled_texture: Matcher::new(intern),
		depth_texture: Matcher::new(intern),
		sampler: Matcher::new(intern),
		storage_texture: Matcher::new(intern),
		texel_format: Matcher::new(intern),
		inbuilt_function: Matcher::new(intern),
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
	access_mode: Matcher<AccessMode>,
	address_space: Matcher<AddressSpace>,
	builtin: Matcher<Builtin>,
	interpolation_sample: Matcher<InterpolationSample>,
	interpolation_type: Matcher<InterpolationType>,
	primitive: Matcher<PrimitiveType>,
	vec: Matcher<VecType>,
	mat: Matcher<MatType>,
	sampled_texture: Matcher<SampledTextureType>,
	depth_texture: Matcher<DepthTextureType>,
	sampler: Matcher<SamplerType>,
	storage_texture: Matcher<StorageTextureType>,
	texel_format: Matcher<TexelFormat>,
	inbuilt_function: Matcher<InbuiltFunction>,
	kws: Box<Kws>,
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
			address_space: v.address_space.and_then(|x| self.address_space.get(x.name)),
			access_mode: v.access_mode.and_then(|x| self.access_mode.get(x.name)),
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
				} else if let Some(inbuilt) = self.inbuilt_function.get(name.name) {
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
		let name = ident.name.name;
		Some(if name == self.kws.array {
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
		} else if let Some(prim) = self.primitive.get(name) {
			match prim {
				PrimitiveType::F64 => {
					if !self.tu.features.float64 {
						self.diagnostics
							.push(ident.name.span.error("feature `f64` not enabled") + ident.name.span.marker());
					}
				},
				PrimitiveType::F16 => {
					if !self.tu.features.float16 {
						self.diagnostics
							.push(ident.name.span.error("feature `f16` not enabled") + ident.name.span.marker());
					}
				},
				_ => {},
			}

			InbuiltType::Primitive(prim)
		} else if let Some(comp) = self.vec.get(name) {
			let name = comp.to_static_str();
			if ident.generics.len() > 1 {
				self.diagnostics.push(
					ident.name.span.error(format!("too many generics for `{}`", name)) + ident.name.span.marker(),
				);
			}
			let ty = ident.generics.into_iter().next();
			let of = ty.map(|x| (x.span, self.inbuilt(x)));
			match of {
				Some((_, Some(InbuiltType::Primitive(ty)))) => InbuiltType::Vec { ty, comp },
				Some((span, _)) => {
					self.diagnostics
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
		} else if let Some(comp) = self.mat.get(name) {
			let name = comp.to_static_str();
			if ident.generics.len() > 1 {
				self.diagnostics.push(
					ident.name.span.error(format!("too many generics for `{}`", name)) + ident.name.span.marker(),
				);
			}
			let ty = ident.generics.into_iter().next();
			let of = ty.map(|x| (x.span, self.inbuilt(x)));
			match of {
				Some((_, Some(InbuiltType::Primitive(ty)))) => {
					let ty = if let PrimitiveType::F16 = ty {
						FloatType::F16
					} else if let PrimitiveType::F32 = ty {
						FloatType::F32
					} else {
						self.diagnostics.push(
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
					self.diagnostics
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
		} else {
			return None;
		})
	}

	fn inbuilt(&mut self, ty: ast::Type) -> Option<InbuiltType> {
		let no_generics = |this: &mut Self, generics: Vec<ast::Type>, name: &str| {
			if generics.len() != 0 {
				this.diagnostics
					.push(ty.span.error(format!("`{}` cannot have generic parameters", name)) + ty.span.marker());
			}
		};

		let ty = match ty.kind {
			ast::TypeKind::Ident(ident, generics) => {
				if let Some(prim) = self.primitive.get(ident.name) {
					no_generics(self, generics, prim.to_static_str());

					match prim {
						PrimitiveType::F64 => {
							if !self.tu.features.float64 {
								self.diagnostics
									.push(ident.span.error("feature `f64` is not enabled") + ident.span.marker());
							}
						},
						PrimitiveType::F16 => {
							if !self.tu.features.float16 {
								self.diagnostics
									.push(ident.span.error("feature `float16` is not enabled") + ident.span.marker());
							}
						},
						_ => {},
					}

					InbuiltType::Primitive(prim)
				} else if let Some(comp) = self.vec.get(ident.name) {
					let name = comp.to_static_str();

					if generics.len() != 1 {
						self.diagnostics.push(
							ty.span
								.error(format!("`{}` must have exactly 1 generic parameter", name))
								+ ty.span.marker(),
						);
					}

					if let Some(InbuiltType::Primitive(ty)) =
						generics.into_iter().next().map(|x| self.inbuilt(x)).flatten()
					{
						InbuiltType::Vec { comp, ty }
					} else {
						self.diagnostics.push(
							ty.span
								.error(format!("`{}` must have a scalar type as its generic parameter", name))
								+ ty.span.marker(),
						);
						InbuiltType::Vec {
							comp,
							ty: PrimitiveType::F32,
						}
					}
				} else if let Some(comp) = self.mat.get(ident.name) {
					let name = comp.to_static_str();

					if generics.len() != 1 {
						self.diagnostics.push(
							ty.span
								.error(format!("`{}` must have exactly 1 generic parameter", name))
								+ ty.span.marker(),
						);
					}

					if let Some(InbuiltType::Primitive(x)) =
						generics.into_iter().next().map(|x| self.inbuilt(x)).flatten()
					{
						let ty = match x {
							PrimitiveType::F16 => FloatType::F16,
							PrimitiveType::F32 => FloatType::F32,
							PrimitiveType::F64 => FloatType::F64,
							_ => {
								self.diagnostics.push(
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
						self.diagnostics.push(
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
				} else if ident.name == self.kws.atomic {
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
				} else if ident.name == self.kws.ptr {
					let mut generics = generics.into_iter();
					let address_space = generics.next();
					let to = generics.next().map(|x| self.ty(x));
					let access_mode = generics.next();

					let address_space = if let Some(address_space) = address_space {
						self.ty_to_ident(address_space, "address space")
							.and_then(|x| self.address_space.get(x.name))
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
									.and_then(|x| self.access_mode.get(x.name))
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
				} else if let Some(s) = self.sampled_texture.get(ident.name) {
					let name = s.to_static_str();

					if generics.len() != 1 {
						self.diagnostics.push(
							ty.span
								.error(format!("`{}` must have exactly 1 generic parameter", name))
								+ ty.span.marker(),
						);
					}

					let sample_type = generics.into_iter().next().and_then(|x| self.inbuilt(x));
					let sample_type = match sample_type {
						Some(InbuiltType::Primitive(PrimitiveType::F32)) => SampleType::F32,
						Some(InbuiltType::Primitive(PrimitiveType::U32)) => SampleType::U32,
						Some(InbuiltType::Primitive(PrimitiveType::I32)) => SampleType::I32,
						Some(_) => {
							self.diagnostics.push(
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
				} else if let Some(depth) = self.depth_texture.get(ident.name) {
					no_generics(self, generics, depth.to_static_str());
					InbuiltType::DepthTexture(depth)
				} else if let Some(s) = self.storage_texture.get(ident.name) {
					let name = s.to_static_str();

					if generics.len() != 2 {
						self.diagnostics.push(
							ty.span
								.error(format!("`{}` must have exactly 2 generic parameters", name))
								+ ty.span.marker(),
						);
					}

					let mut generics = generics.into_iter();
					let texel_format = generics
						.next()
						.and_then(|x| self.ty_to_ident(x, "texel format"))
						.and_then(|x| self.texel_format.get(x.name))
						.unwrap_or(TexelFormat::Rgba8Unorm);
					let access = generics
						.next()
						.and_then(|x| self.ty_to_ident(x, "access mode"))
						.map(|x| (self.access_mode.get(x.name), x.span));

					let access = if let Some((access, span)) = access {
						let access = access.unwrap_or(AccessMode::Write);
						if access != AccessMode::Write && !self.tu.features.storage_image_other_access {
							self.diagnostics.push(
								span.error("feature `storage_image_other_access` is not enabled") + span.marker(),
							);
						}
						access
					} else {
						AccessMode::Write
					};

					InbuiltType::StorageTexture(s, texel_format, access)
				} else if let Some(s) = self.sampler.get(ident.name) {
					no_generics(self, generics, s.to_static_str());
					InbuiltType::Sampler(s)
				} else {
					return None;
				}
			},
			ast::TypeKind::Array(array, of, len) => {
				if array.name == self.kws.array {
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
			x if x == self.kws.align => args(self, 1).map(|_| AttributeType::Align(attrib.exprs[0].clone())),
			x if x == self.kws.binding => args(self, 1).map(|_| AttributeType::Binding(attrib.exprs[0].clone())),
			x if x == self.kws.builtin => args(self, 1).and_then(|_| {
				expr_as_ident(self, &attrib.exprs[0])
					.and_then(|ident| self.builtin.get(ident.name).map(|x| AttributeType::Builtin(x)))
			}),
			x if x == self.kws.compute => args(self, 0).map(|_| AttributeType::Compute),
			x if x == self.kws.const_ => args(self, 0).map(|_| AttributeType::Const),
			x if x == self.kws.fragment => args(self, 0).map(|_| AttributeType::Fragment),
			x if x == self.kws.group => args(self, 1).map(|_| AttributeType::Group(attrib.exprs[0].clone())),
			x if x == self.kws.id => args(self, 1).map(|_| AttributeType::Id(attrib.exprs[0].clone())),
			x if x == self.kws.interpolate => {
				if attrib.exprs.len() < 1 || attrib.exprs.len() > 2 {
					self.diagnostics
						.push(attrib.span.error("expected 1 or 2 arguments") + attrib.span.marker());
					None
				} else {
					let ty = expr_as_ident(self, &attrib.exprs[0])
						.and_then(|x| self.interpolation_type.get(x.name))
						.unwrap_or(InterpolationType::Perspective);
					let sample = attrib
						.exprs
						.get(1)
						.and_then(|x| expr_as_ident(self, x))
						.and_then(|x| self.interpolation_sample.get(x.name));

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
			x if x == self.kws.invariant => args(self, 0).map(|_| AttributeType::Invariant),
			x if x == self.kws.location => args(self, 1).map(|_| AttributeType::Location(attrib.exprs[0].clone())),
			x if x == self.kws.size => args(self, 1).map(|_| AttributeType::Size(attrib.exprs[0].clone())),
			x if x == self.kws.vertex => args(self, 0).map(|_| AttributeType::Vertex),
			x if x == self.kws.workgroup_size => {
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

struct Kws {
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
	array: Text,
	atomic: Text,
	ptr: Text,
}

impl Kws {
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
			array: intern.get_static("array"),
			atomic: intern.get_static("atomic"),
			ptr: intern.get_static("ptr"),
		}
	}
}