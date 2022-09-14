use wgsl::{
	diagnostic::{Diagnostics, Span},
	parse::parse,
	text::Interner,
};

fn test(source: &str, file: &str) {
	let mut intern = Interner::new();
	let mut diagnostics = Diagnostics::new();

	let _ = parse(&mut intern, &mut diagnostics, source, file);

	if diagnostics.had_error() {
		for diag in diagnostics.diags() {
			let (line, column) = span_to_line_column(source, diag.span);
			println!(
				"{} at {}:{} (`{}`)",
				diag.message,
				line,
				column,
				&source[diag.span.start as usize..diag.span.end as usize]
			);
		}

		panic!("failed to parse file '{}'", file);
	}
}

fn span_to_line_column(source: &str, span: Span) -> (usize, usize) {
	let mut line = 1;
	let mut column = 1;

	for (i, c) in source.char_indices() {
		if i == span.start as usize {
			break;
		}

		if c == '\n' {
			line += 1;
			column = 1;
		} else {
			column += 1;
		}
	}
	(line, column)
}

#[test]
fn cases() {
	for file in std::fs::read_dir("tests/in").unwrap() {
		let file = file.unwrap();
		if file.file_type().unwrap().is_dir() {
			continue;
		}

		let path = file.path();
		let source = std::fs::read_to_string(&path).unwrap();
		test(&source, &path.to_str().unwrap());
	}
}
