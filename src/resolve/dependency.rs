use crate::{
	diagnostic::{Diagnostics, Span},
	resolve::ir::{Decl, DeclDependencyKind, DeclId, DeclKind, TranslationUnit},
};

struct StackList<'a, T> {
	prev: Option<&'a StackList<'a, T>>,
	value: Option<T>,
}

impl<'a, T> StackList<'a, T> {
	fn empty() -> Self {
		Self {
			prev: None,
			value: None,
		}
	}

	fn with(&'a self, value: T) -> Self {
		Self {
			prev: Some(self),
			value: Some(value),
		}
	}
}

pub fn resolve_all_dependencies(tu: &mut TranslationUnit, diagnostics: &mut Diagnostics) {
	let mut visited = vec![false; tu.decls.len()];
	let mut temp_visited = vec![false; tu.decls.len()];

	let mut depdendency_order = Vec::with_capacity(tu.decls.len());

	for &decl in tu.roots.iter() {
		recursive_solve(
			decl,
			StackList::empty(),
			&mut tu.decls,
			&mut visited,
			&mut temp_visited,
			&mut depdendency_order,
			diagnostics,
		);
	}

	for id in 0..tu.decls.len() {
		let visit = visited[id];
		if !visit {
			let span = decl_ident_span(&tu.decls[id]);
			diagnostics.push(span.warning("unused declaration") + span.marker());
			recursive_solve(
				DeclId(id as _),
				StackList::empty(),
				&mut tu.decls,
				&mut visited,
				&mut temp_visited,
				&mut depdendency_order,
				diagnostics,
			);
		}
	}

	tu.dependency_order = depdendency_order;
}

fn recursive_solve(
	id: DeclId, ctx: StackList<(DeclId, Span)>, decls: &mut [Decl], visited: &mut [bool], temp_visited: &mut [bool],
	dep_order: &mut Vec<DeclId>, diagnostics: &mut Diagnostics,
) {
	if visited[id.0 as usize] {
		return;
	}

	let decl = &mut decls[id.0 as usize];

	if temp_visited[id.0 as usize] {
		let span = decl_ident_span(decl);
		let mut error = span.error("cyclic dependencies are not allowed") + span.label("cycle in this declaration");

		let mut ctx = &ctx;
		while let Some((i, span)) = ctx.value {
			if i == id {
				error = error + span.label("completing the cycle");
				break;
			} else {
				error = error + span.label("which depends on");
			}

			if let Some(prev) = ctx.prev {
				ctx = prev;
			} else {
				break;
			}
		}

		diagnostics.push(error);
		return;
	}
	temp_visited[id.0 as usize] = true;

	let dec: Vec<_> = decl
		.dependencies
		.iter()
		.filter_map(|dep| match dep.kind {
			DeclDependencyKind::Decl(id) => Some((id, dep.usage)),
			DeclDependencyKind::Inbuilt(_) => None,
		})
		.collect();
	for &(decl, span) in dec.iter() {
		recursive_solve(
			decl,
			ctx.with((id, span)),
			decls,
			visited,
			temp_visited,
			dep_order,
			diagnostics,
		);
	}

	dep_order.push(id);
	let deps: Vec<_> = dec
		.iter()
		.flat_map(|&id| decls[id.0 .0 as usize].dependencies.iter().copied())
		.collect();
	let decl = &mut decls[id.0 as usize];
	decl.dependencies.extend(deps);

	temp_visited[id.0 as usize] = false;
	visited[id.0 as usize] = true;
}

fn decl_ident_span(decl: &Decl) -> Span {
	match &decl.kind {
		DeclKind::Fn(f) => f.name.span,
		DeclKind::Struct(s) => s.name.span,
		DeclKind::Type(t) => t.name.span,
		DeclKind::Const(c) => c.name.span,
		DeclKind::Override(o) => o.name.span,
		DeclKind::Var(v) => v.inner.name.span,
		DeclKind::StaticAssert(_) => unreachable!(),
	}
}
