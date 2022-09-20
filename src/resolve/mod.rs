use aho_corasick::AhoCorasick;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
	ast,
	ast::{ExprKind, GlobalDeclKind, Ident, StmtKind, VarDecl},
	diagnostic::{Diagnostics, Span},
	resolve::{
		features::{EnabledFeatures, Feature},
		inbuilt::{
			reserved_matcher,
			AccessMode,
			AddressSpace,
			AttributeType,
			Builtin,
			ConservativeDepth,
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
		ir::{DeclDependency, DeclDependencyKind, DeclId, FloatType, FnTarget, InbuiltType, LocalId, SampleType},
	},
	text::{Interner, Text},
};

mod cycle;
pub mod features;
pub mod inbuilt;
pub mod inbuilt_functions;
mod index;
pub mod ir;

pub fn resolve(tu: ast::TranslationUnit, intern: &mut Interner, diagnostics: &mut Diagnostics) -> ir::TranslationUnit {
	let index = index::generate_index(&tu, diagnostics);

	let mut out = ir::TranslationUnit::new(EnabledFeatures::new(intern));

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
		conservative_depth: Matcher::new(intern),
		inbuilt_function: Matcher::new(intern),
		tu: &mut out,
		index,
		diagnostics,
		intern,
		reserved_matcher: reserved_matcher(),
		locals: 0,
		in_loop: false,
		in_continuing: false,
		in_function: false,
		scopes: Vec::new(),
		dependencies: FxHashSet::default(),
	};

	for decl in tu.decls {
		resolver.decl(decl);
	}

	cycle::resolve_all_dependencies(&mut out, diagnostics);

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
	conservative_depth: Matcher<ConservativeDepth>,
	inbuilt_function: Matcher<InbuiltFunction>,
	kws: Box<Kws>,
	locals: u32,
	in_loop: bool,
	in_continuing: bool,
	in_function: bool,
	scopes: Vec<FxHashMap<Text, (LocalId, Span, bool)>>,
	dependencies: FxHashSet<DeclDependency>,
}

impl<'a> Resolver<'a> {
	fn decl(&mut self, decl: ast::GlobalDecl) {
		self.locals = 0;

		let kind = match decl.kind {
			GlobalDeclKind::Fn(f) => {
				let f = self.fn_(f);

				if !matches!(f.attribs, ir::FnAttribs::None) {
					self.tu.roots.push(DeclId(self.tu.decls.len() as _));
				}

				ir::DeclKind::Fn(f)
			},
			GlobalDeclKind::Override(ov) => ir::DeclKind::Override(self.ov(ov)),
			GlobalDeclKind::Var(v) => ir::DeclKind::Var(ir::Var {
				attribs: self.var_attribs(v.attribs),
				inner: self.var(v.inner),
			}),
			GlobalDeclKind::Let(l) => ir::DeclKind::Const(self.let_(l)),
			GlobalDeclKind::Const(c) => ir::DeclKind::Const(self.let_(c)),
			GlobalDeclKind::StaticAssert(s) => ir::DeclKind::StaticAssert(self.expr(s.expr)),
			GlobalDeclKind::Struct(s) => {
				self.verify_ident(s.name);
				ir::DeclKind::Struct(ir::Struct {
					name: s.name,
					fields: s.fields.into_iter().map(|f| self.field(f)).collect(),
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

		let decl = ir::Decl {
			kind,
			span: decl.span,
			dependencies: std::mem::take(&mut self.dependencies),
		};

		self.tu.decls.push(decl);
	}

	fn fn_(&mut self, fn_: ast::Fn) -> ir::Fn {
		self.verify_ident(fn_.name);

		self.in_function = true;

		self.scopes.push(FxHashMap::default());
		let args = fn_.args.into_iter().map(|x| self.arg(x)).collect();
		let block = self.block_inner(fn_.block);
		self.pop_scope();

		self.in_function = false;

		ir::Fn {
			attribs: self.fn_attribs(fn_.attribs),
			name: fn_.name,
			args,
			ret_attribs: self.arg_attribs(fn_.ret_attribs),
			ret: fn_.ret.map(|x| self.ty(x)),
			block,
		}
	}

	fn ov(&mut self, o: ast::Override) -> ir::Override {
		self.verify_ident(o.name);

		let mut id = None;
		let a: Vec<_> = o.attribs.into_iter().filter_map(|x| self.attrib(x)).collect();
		for attrib in a {
			match attrib.ty {
				AttributeType::Id(expr) => {
					if id.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						id = Some(self.expr(expr));
					}
				},
				_ => {
					self.diagnostics
						.push(attrib.span.error("this attribute is not allowed here") + attrib.span.marker());
				},
			}
		}

		ir::Override {
			id,
			name: o.name,
			ty: o.ty.map(|x| self.ty(x)),
			val: o.val.map(|x| self.expr(x)),
		}
	}

	fn arg(&mut self, arg: ast::Arg) -> ir::Arg {
		self.verify_ident(arg.name);

		let args = self.scopes.last_mut().expect("no scope");
		let id = LocalId(self.locals);
		self.locals += 1;
		let old = args.insert(arg.name.name, (id, arg.span, false));
		if let Some((_, span, _)) = old {
			self.diagnostics.push(
				arg.name.span.error("duplicate argument name")
					+ arg.name.span.label("redeclared here")
					+ span.label("previously declared here"),
			);
		}

		ir::Arg {
			attribs: self.arg_attribs(arg.attribs),
			name: arg.name,
			ty: self.ty(arg.ty),
			span: arg.span,
			id,
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

	fn var_attribs(&mut self, attribs: Vec<ast::Attribute>) -> ir::VarAttribs {
		let mut out = ir::VarAttribs {
			group: None,
			binding: None,
		};

		let a: Vec<_> = attribs.into_iter().filter_map(|x| self.attrib(x)).collect();
		for attrib in a {
			match attrib.ty {
				AttributeType::Group(g) => {
					if out.group.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						out.group = Some(self.expr(g));
					}
				},
				AttributeType::Binding(b) => {
					if out.binding.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						out.binding = Some(self.expr(b));
					}
				},
				_ => {
					self.diagnostics
						.push(attrib.span.error("this attribute is not allowed here") + attrib.span.marker());
				},
			}
		}

		out
	}

	fn arg_attribs(&mut self, attribs: Vec<ast::Attribute>) -> ir::ArgAttribs {
		let mut out = ir::ArgAttribs {
			builtin: None,
			location: None,
			interpolate: None,
			invariant: false,
		};

		let a: Vec<_> = attribs.into_iter().filter_map(|x| self.attrib(x)).collect();
		for attrib in a {
			match attrib.ty {
				AttributeType::Builtin(b) => {
					if out.builtin.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						out.builtin = Some(b);
					}
				},
				AttributeType::Location(l) => {
					if out.location.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						out.location = Some(self.expr(l));
					}
				},
				AttributeType::Interpolate(i, s) => {
					if out.interpolate.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						out.interpolate = Some((i, s));
					}
				},
				AttributeType::Invariant => {
					if out.invariant {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						out.invariant = true;
					}
				},
				_ => {
					self.diagnostics
						.push(attrib.span.error("this attribute is not allowed here") + attrib.span.marker());
				},
			}
		}

		out
	}

	fn fn_attribs(&mut self, attribs: Vec<ast::Attribute>) -> ir::FnAttribs {
		let mut out = ir::FnAttribs::None;
		let mut expect_compute = None;

		let a: Vec<_> = attribs.into_iter().filter_map(|x| self.attrib(x)).collect();
		for attrib in a {
			match attrib.ty {
				AttributeType::Const => self
					.diagnostics
					.push(attrib.span.error("user defined `const` functions are not allowed") + attrib.span.marker()),
				AttributeType::Vertex => {
					if let ir::FnAttribs::None = out {
						out = ir::FnAttribs::Vertex;
					} else {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					}
				},
				AttributeType::Fragment => {
					if let ir::FnAttribs::None = out {
						out = ir::FnAttribs::Fragment(None);
					} else {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					}
				},
				AttributeType::Compute => {
					if let ir::FnAttribs::None = out {
						expect_compute = Some(attrib.span);
						out = ir::FnAttribs::Compute(None, None, None);
					} else if expect_compute.is_some() {
						expect_compute = None;
					} else {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					}
				},
				AttributeType::WorkgroupSize(x, y, z) => {
					if let ir::FnAttribs::None = out {
						expect_compute = Some(attrib.span);
						out = ir::FnAttribs::Compute(
							Some(self.expr(x)),
							y.map(|x| self.expr(x)),
							z.map(|x| self.expr(x)),
						);
					} else if expect_compute.is_some() {
						expect_compute = None;
					} else {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					}
				},
				AttributeType::ConservativeDepth(depth) => {
					if let ir::FnAttribs::Fragment(_) = out {
						out = ir::FnAttribs::Fragment(Some(depth));
					} else {
						self.diagnostics
							.push(attrib.span.error("this attribute is not allowed here") + attrib.span.marker());
					}
				},
				_ => {
					self.diagnostics
						.push(attrib.span.error("this attribute is not allowed here") + attrib.span.marker());
				},
			}
		}

		if let Some(span) = expect_compute {
			self.diagnostics.push(
				span.error(if matches!(out, ir::FnAttribs::Compute(None, _, _)) {
					"`@compute` without `@workgroup_size` attribute"
				} else {
					"`@workgroup_size` without `@compute` attribute"
				}) + span.marker(),
			);
		}

		out
	}

	fn field(&mut self, field: ast::Arg) -> ir::Field {
		let mut attribs = ir::FieldAttribs {
			align: None,
			builtin: None,
			location: None,
			interpolate: None,
			invariant: false,
			size: None,
		};

		let a: Vec<_> = field.attribs.into_iter().filter_map(|x| self.attrib(x)).collect();
		for attrib in a {
			match attrib.ty {
				AttributeType::Align(expr) => {
					if attribs.align.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						attribs.align = Some(self.expr(expr));
					}
				},
				AttributeType::Builtin(b) => {
					if attribs.builtin.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						attribs.builtin = Some(b);
					}
				},
				AttributeType::Location(loc) => {
					if attribs.location.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						attribs.location = Some(self.expr(loc));
					}
				},
				AttributeType::Interpolate(i, s) => {
					if attribs.interpolate.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						attribs.interpolate = Some((i, s));
					}
				},
				AttributeType::Invariant => {
					if attribs.invariant {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						attribs.invariant = true;
					}
				},
				AttributeType::Size(expr) => {
					if attribs.size.is_some() {
						self.diagnostics
							.push(attrib.span.error("duplicate attribute") + attrib.span.marker());
					} else {
						attribs.size = Some(self.expr(expr));
					}
				},
				_ => {
					self.diagnostics
						.push(attrib.span.error("this attribute is not allowed here") + attrib.span.marker());
				},
			}
		}

		ir::Field {
			attribs,
			name: field.name,
			ty: self.ty(field.ty),
		}
	}

	fn var(&mut self, v: ast::VarNoAttribs) -> ir::VarNoAttribs {
		self.verify_ident(v.name);

		let ty = v.ty.map(|x| self.ty(x));

		let as_ = v
			.address_space
			.map(|x| (self.address_space(x), x.span))
			.and_then(|(a, s)| a.map(|a| (a, s)));
		let am = v
			.access_mode
			.map(|x| (self.access_mode(x), x.span))
			.and_then(|(a, s)| a.map(|a| (a, s)));
		let (address_space, access_mode) = self.handle_address_space_and_access_mode(as_, am);

		let (address_space, access_mode) = if address_space == AddressSpace::Handle {
			if let Some(ir::TypeKind::Inbuilt(
				InbuiltType::BindingArray { .. }
				| InbuiltType::Sampler(_)
				| InbuiltType::StorageTexture(..)
				| InbuiltType::SampledTexture(..),
			)) = ty.as_ref().map(|x| &x.kind)
			{
				// Infer handle if its a resource type.
				(AddressSpace::Handle, AccessMode::Read)
			} else {
				(AddressSpace::Private, AccessMode::ReadWrite)
			}
		} else {
			(address_space, access_mode)
		};

		if self.in_function && address_space != AddressSpace::Function {
			let span = as_.unwrap().1;
			self.diagnostics.push(
				span.error(format!(
					"cannot declare variable with address space `{}` in a function",
					address_space
				)) + span.marker(),
			);
		} else if !self.in_function && address_space == AddressSpace::Function {
			let span = as_.unwrap().1;
			self.diagnostics.push(
				span.error("cannot declare variable with address space `function` outside of a function")
					+ span.marker(),
			);
		}

		if let Some(val) = v.val.as_ref() {
			if !matches!(address_space, AddressSpace::Function | AddressSpace::Private) {
				self.diagnostics.push(
					val.span.error(format!(
						"cannot initialize variable with address space `{}`",
						address_space
					)) + val.span.marker(),
				);
			}
		}

		ir::VarNoAttribs {
			address_space,
			access_mode,
			name: v.name,
			ty,
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
				self.dependencies.insert(DeclDependency {
					kind: DeclDependencyKind::Decl(user),
					usage: ident.span,
				});
				ir::TypeKind::User(user)
			} else {
				self.diagnostics
					.push(ident.span.error("undefined type") + ident.span.marker());
				ir::TypeKind::Inbuilt(InbuiltType::Primitive(PrimitiveType::Infer))
			}
		};

		ir::Type { kind, span }
	}

	fn block(&mut self, block: ast::Block) -> ir::Block {
		self.scopes.push(FxHashMap::default());
		let ret = self.block_inner(block);
		self.pop_scope();
		ret
	}

	fn block_inner(&mut self, block: ast::Block) -> ir::Block {
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
			StmtKind::Block(block) => ir::StmtKind::Block(self.block(block)),
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
				let block = self.block_inner(for_.block);
				self.pop_scope();

				ir::StmtKind::For(ir::For {
					init,
					cond,
					update,
					block,
				})
			},
			StmtKind::If(if_) => {
				let cond = self.expr(if_.cond);
				let block = self.block(if_.block);
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
				let block = self.block(block);
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
						block: self.block(case.block),
						span: case.span,
					})
					.collect(),
			}),
			StmtKind::While(while_) => ir::StmtKind::While(ir::While {
				cond: self.expr(while_.cond),
				block: self.block(while_.block),
			}),
			StmtKind::Continuing(c) => {
				if !self.in_loop {
					self.diagnostics
						.push(stmt.span.error("`continuing` must be inside a `loop`") + stmt.span.marker());
				}
				self.in_loop = false;
				self.in_continuing = true;
				let block = self.block(c);
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
			ExprKind::Literal(l) => {
				match l {
					ast::Literal::F16(_) => {
						self.tu
							.features
							.require(Feature::Float16, expr.span, &mut self.diagnostics)
					},
					_ => {},
				}
				ir::ExprKind::Literal(l)
			},
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
							.insert(name.name, (id, name.span, false));

						if let Some((_, span, _)) = old {
							self.diagnostics.push(
								name.span.error("shadowing is not allowed in the same scope")
									+ span.label("previously declared here")
									+ name.span.label("redeclared here"),
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
						ir::ExprStatementKind::Assign(ir::AssignExpr {
							lhs: Box::new(ir::AssignTarget {
								kind: ir::AssignTargetKind::Ignore,
								span: assign.lhs.span,
							}),
							op: ast::AssignOp::Assign,
							rhs: Box::new(self.expr(*assign.rhs)),
						})
					} else {
						self.diagnostics
							.push(assign.lhs.span.error("`_` is not allowed here") + assign.lhs.span.marker());
						return None;
					}
				} else {
					let lhs = self.expr(*assign.lhs);

					let kind = match lhs.kind {
						ir::ExprKind::Local(l) => ir::AssignTargetKind::Local(l),
						ir::ExprKind::Global(g) => ir::AssignTargetKind::Global(g),
						ir::ExprKind::Member(m, i) => ir::AssignTargetKind::Member(m, i),
						ir::ExprKind::Index(i, w) => ir::AssignTargetKind::Index(i, w),
						ir::ExprKind::Unary(unary) if matches!(unary.op, ast::UnaryOp::Deref) => {
							ir::AssignTargetKind::Deref(unary.expr)
						},
						_ => {
							self.diagnostics
								.push(lhs.span.error("cannot assign to this expression") + lhs.span.marker());
							ir::AssignTargetKind::Ignore
						},
					};
					let lhs = Box::new(ir::AssignTarget { kind, span: lhs.span });

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
					self.dependencies.insert(DeclDependency {
						kind: DeclDependencyKind::Decl(decl),
						usage: name.span,
					});
					FnTarget::Decl(decl)
				} else if let Some(ty) = self.constructible_inbuilt(ident) {
					FnTarget::InbuiltType(Box::new(ty))
				} else if let Some(inbuilt) = self.inbuilt_function.get(name.name) {
					self.dependencies.insert(DeclDependency {
						kind: DeclDependencyKind::Inbuilt(inbuilt),
						usage: name.span,
					});
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
					self.tu
						.features
						.require(Feature::Float64, ident.name.span, &mut self.diagnostics)
				},
				PrimitiveType::F16 => {
					self.tu
						.features
						.require(Feature::Float16, ident.name.span, &mut self.diagnostics)
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
							self.tu
								.features
								.require(Feature::Float64, ident.span, &mut self.diagnostics)
						},
						PrimitiveType::F16 => {
							self.tu
								.features
								.require(Feature::Float16, ident.span, &mut self.diagnostics)
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

					let address_space = address_space
						.and_then(|x| {
							self.ty_to_ident(x, "address space")
								.map(|x| (self.address_space(x), x.span))
						})
						.and_then(|(a, s)| a.map(|a| (a, s)));

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
							self.ty_to_ident(access_mode, "access mode")
								.map(|x| (self.access_mode(x), x.span))
						})
						.and_then(|(a, s)| a.map(|a| (a, s)));

					if address_space.is_none() {
						self.diagnostics
							.push(ty.span.error(format!("expected address space")) + ty.span.marker());
					}

					let (address_space, access_mode) =
						self.handle_address_space_and_access_mode(address_space, access_mode);

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
						.and_then(|x| self.texel_format(x))
						.unwrap_or(TexelFormat::Rgba8Unorm);
					let access = generics
						.next()
						.and_then(|x| self.ty_to_ident(x, "access mode"))
						.map(|x| (self.access_mode(x), x.span));

					let access = if let Some((access, span)) = access {
						let access = access.unwrap_or(AccessMode::Write);
						if access != AccessMode::Write {
							self.tu
								.features
								.require(Feature::StorageImageRead, span, &mut self.diagnostics);
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
					self.tu
						.features
						.require(Feature::BindingArray, array.span, &mut self.diagnostics);

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
			x if x == self.kws.align => {
				args(self, 1).map(|_| AttributeType::Align(attrib.exprs.into_iter().next().unwrap()))
			},
			x if x == self.kws.binding => {
				args(self, 1).map(|_| AttributeType::Binding(attrib.exprs.into_iter().next().unwrap()))
			},
			x if x == self.kws.builtin => args(self, 1).and_then(|_| {
				expr_as_ident(self, &attrib.exprs[0])
					.and_then(|ident| self.builtin(ident).map(|x| AttributeType::Builtin(x)))
			}),
			x if x == self.kws.compute => args(self, 0).map(|_| AttributeType::Compute),
			x if x == self.kws.const_ => args(self, 0).map(|_| AttributeType::Const),
			x if x == self.kws.fragment => args(self, 0).map(|_| AttributeType::Fragment),
			x if x == self.kws.group => {
				args(self, 1).map(|_| AttributeType::Group(attrib.exprs.into_iter().next().unwrap()))
			},
			x if x == self.kws.id => args(self, 1).map(|_| AttributeType::Id(attrib.exprs.into_iter().next().unwrap())),
			x if x == self.kws.interpolate => {
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
			x if x == self.kws.invariant => args(self, 0).map(|_| AttributeType::Invariant),
			x if x == self.kws.location => {
				args(self, 1).map(|_| AttributeType::Location(attrib.exprs.into_iter().next().unwrap()))
			},
			x if x == self.kws.size => {
				args(self, 1).map(|_| AttributeType::Size(attrib.exprs.into_iter().next().unwrap()))
			},
			x if x == self.kws.vertex => args(self, 0).map(|_| AttributeType::Vertex),
			x if x == self.kws.workgroup_size => {
				if attrib.exprs.len() < 1 || attrib.exprs.len() > 3 {
					self.diagnostics
						.push(attrib.span.error("expected 1, 2, or 3 arguments") + attrib.span.marker());
					None
				} else {
					let mut iter = attrib.exprs.into_iter();
					let x = iter.next().unwrap();
					let y = iter.next();
					let z = iter.next();
					Some(AttributeType::WorkgroupSize(x, y, z))
				}
			},
			x if x == self.kws.early_depth_test => args(self, 1).map(|_| {
				AttributeType::ConservativeDepth(
					expr_as_ident(self, &attrib.exprs.into_iter().next().unwrap())
						.and_then(|x| self.conservative_depth(x))
						.unwrap_or(ConservativeDepth::Unchanged),
				)
			}),
			_ => {
				self.diagnostics
					.push(attrib.name.span.error("unknown attribute") + attrib.name.span.marker());
				None
			},
		};

		ty.map(|ty| ir::Attribute { span: attrib.span, ty })
	}

	fn access_mode(&mut self, ident: Ident) -> Option<AccessMode> {
		match self.access_mode.get(ident.name) {
			Some(x) => Some(x),
			None => {
				self.diagnostics
					.push(ident.span.error("unknown access mode") + ident.span.marker());
				None
			},
		}
	}

	fn address_space(&mut self, ident: Ident) -> Option<AddressSpace> {
		match self.address_space.get(ident.name) {
			Some(AddressSpace::Handle) => {
				self.diagnostics
					.push(ident.span.error("`handle` address space is not allowed here") + ident.span.marker());
				Some(AddressSpace::Handle)
			},
			Some(x) => Some(x),
			None => {
				self.diagnostics
					.push(ident.span.error("unknown address space") + ident.span.marker());
				None
			},
		}
	}

	fn builtin(&mut self, ident: Ident) -> Option<Builtin> {
		match self.builtin.get(ident.name) {
			Some(x) => Some(x),
			None => {
				self.diagnostics
					.push(ident.span.error("unknown builtin") + ident.span.marker());
				None
			},
		}
	}

	fn interpolation_sample(&mut self, ident: Ident) -> Option<InterpolationSample> {
		match self.interpolation_sample.get(ident.name) {
			Some(x) => Some(x),
			None => {
				self.diagnostics
					.push(ident.span.error("unknown interpolation sample") + ident.span.marker());
				None
			},
		}
	}

	fn interpolation_type(&mut self, ident: Ident) -> Option<InterpolationType> {
		match self.interpolation_type.get(ident.name) {
			Some(x) => Some(x),
			None => {
				self.diagnostics
					.push(ident.span.error("unknown interpolation type") + ident.span.marker());
				None
			},
		}
	}

	fn texel_format(&mut self, ident: Ident) -> Option<TexelFormat> {
		match self.texel_format.get(ident.name) {
			Some(x) => Some(x),
			None => {
				self.diagnostics
					.push(ident.span.error("unknown texel format") + ident.span.marker());
				None
			},
		}
	}

	fn conservative_depth(&mut self, ident: Ident) -> Option<ConservativeDepth> {
		match self.conservative_depth.get(ident.name) {
			Some(x) => Some(x),
			None => {
				self.diagnostics
					.push(ident.span.error("unknown conservative depth") + ident.span.marker());
				None
			},
		}
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

	fn handle_address_space_and_access_mode(
		&mut self, address_space: Option<(AddressSpace, Span)>, access_mode: Option<(AccessMode, Span)>,
	) -> (AddressSpace, AccessMode) {
		let address_space = address_space.map(|x| x.0).unwrap_or_else(|| {
			if self.in_function {
				AddressSpace::Function
			} else {
				AddressSpace::Handle // Is not user-settable, so use as a sentinel.
			}
		});

		let default = || match address_space {
			AddressSpace::Function => AccessMode::ReadWrite,
			AddressSpace::Private => AccessMode::ReadWrite,
			AddressSpace::Storage => AccessMode::Read,
			AddressSpace::Uniform => AccessMode::Read,
			AddressSpace::Workgroup => AccessMode::ReadWrite,
			AddressSpace::Handle => AccessMode::ReadWrite, // Doesn't matter what we return.
			AddressSpace::PushConstant => AccessMode::Read,
		};

		let access_mode = if let Some((mode, span)) = access_mode {
			if address_space != AddressSpace::Storage {
				self.diagnostics
					.push(span.error("access mode is not allowed here") + span.marker());
				default()
			} else {
				mode
			}
		} else {
			default()
		};

		(address_space, access_mode)
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
		for scope in self.scopes.iter_mut().rev() {
			if let Some((id, _, used)) = scope.get_mut(&ident.name) {
				*used = true;
				return ir::ExprKind::Local(*id);
			}
		}

		if let Some(global) = self.index.get(ident.name) {
			self.dependencies.insert(DeclDependency {
				kind: DeclDependencyKind::Decl(global),
				usage: ident.span,
			});
			ir::ExprKind::Global(global)
		} else {
			self.diagnostics
				.push(ident.span.error("undefined identifier") + ident.span.marker());
			ir::ExprKind::Error
		}
	}

	fn pop_scope(&mut self) {
		let scope = self.scopes.pop().unwrap();
		for (_, (_, span, used)) in scope {
			if !used {
				self.diagnostics.push(span.warning("unused variable") + span.marker());
			}
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
	early_depth_test: Text,
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
			early_depth_test: intern.get_static("early_depth_test"),
			array: intern.get_static("array"),
			atomic: intern.get_static("atomic"),
			ptr: intern.get_static("ptr"),
		}
	}
}
