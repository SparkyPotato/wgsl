use std::collections::HashMap;

use crate::{
	ast::*,
	diagnostic::{Diagnostics, Span},
	resolve::ir::DeclId,
	text::Text,
};

pub struct Index {
	decls: HashMap<Text, DeclId>,
	spans: Vec<Span>,
}

impl Index {
	fn insert(&mut self, ident: Ident) -> Option<Span> {
		let id = self.spans.len() as u32;
		let old = self.decls.insert(ident.name, DeclId(id));
		self.spans.push(ident.span);
		old.map(|id| self.spans[id.0 as usize])
	}

	pub fn get(&self, ident: Text) -> Option<DeclId> { self.decls.get(&ident).copied() }
}

pub fn generate_index(tu: &TranslationUnit, diagnostics: &mut Diagnostics) -> Index {
	let mut index = Index {
		decls: HashMap::new(),
		spans: Vec::new(),
	};

	for decl in tu.decls.iter() {
		let prev = match &decl.kind {
			GlobalDeclKind::Fn(f) => index.insert(f.name),
			GlobalDeclKind::Override(o) => index.insert(o.name),
			GlobalDeclKind::Var(v) => index.insert(v.inner.name),
			GlobalDeclKind::Const(c) => index.insert(c.name),
			GlobalDeclKind::Struct(s) => index.insert(s.name),
			GlobalDeclKind::Type(ty) => index.insert(ty.name),
			GlobalDeclKind::StaticAssert(_) => None,
			GlobalDeclKind::Let(_) => {
				diagnostics.push(
					decl.span.error("global `let`s are deprecated")
						+ decl.span.marker() + "consider making it a `const`",
				);
				None
			},
		};

		if let Some(prev) = prev {
			diagnostics.push(decl.span.error("duplicate declaration") + prev.label("previously declared here"));
		}
	}

	index
}
