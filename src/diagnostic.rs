use std::ops::{Add, Range};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum DiagnosticKind {
	Error,
	Warning,
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Diagnostic {
	pub kind: DiagnosticKind,
	pub message: String,
	pub span: Span,
	pub labels: Vec<Label>,
	pub notes: Vec<String>,
}

impl Add<Label> for Diagnostic {
	type Output = Diagnostic;

	fn add(mut self, other: Label) -> Self::Output {
		self.labels.push(other);
		self
	}
}

impl<T: ToString> Add<T> for Diagnostic {
	type Output = Diagnostic;

	fn add(mut self, other: T) -> Self::Output {
		self.notes.push(other.to_string());
		self
	}
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Label {
	pub message: String,
	pub span: Span,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Span {
	pub start: u32,
	pub end: u32,
}

impl Add for Span {
	type Output = Span;

	fn add(self, other: Span) -> Self::Output {
		Span {
			start: self.start.min(other.start),
			end: self.end.max(other.end),
		}
	}
}

impl chumsky::Span for Span {
	type Context = ();
	type Offset = u32;

	fn new(_: Self::Context, range: Range<Self::Offset>) -> Self {
		Self {
			start: range.start,
			end: range.end,
		}
	}

	fn context(&self) -> Self::Context { () }

	fn start(&self) -> Self::Offset { self.start }

	fn end(&self) -> Self::Offset { self.end }
}

impl Span {
	pub fn error(&self, message: impl Into<String>) -> Diagnostic {
		Diagnostic {
			kind: DiagnosticKind::Error,
			message: message.into(),
			span: *self,
			labels: Vec::new(),
			notes: Vec::new(),
		}
	}

	pub fn warning(&self, message: impl Into<String>) -> Diagnostic {
		Diagnostic {
			kind: DiagnosticKind::Warning,
			message: message.into(),
			span: *self,
			labels: Vec::new(),
			notes: Vec::new(),
		}
	}

	pub fn label(&self, message: impl Into<String>) -> Label {
		Label {
			message: message.into(),
			span: *self,
		}
	}

	pub fn marker(&self) -> Label {
		Label {
			message: String::new(),
			span: *self,
		}
	}
}

pub struct Diagnostics {
	diagnostics: Vec<Diagnostic>,
	had_error: bool,
}

impl Diagnostics {
	pub fn new() -> Self {
		Self {
			diagnostics: Vec::new(),
			had_error: false,
		}
	}

	pub fn push(&mut self, diagnostic: Diagnostic) {
		if diagnostic.kind == DiagnosticKind::Error {
			self.had_error = true;
		}
		self.diagnostics.push(diagnostic);
	}

	pub fn had_error(&self) -> bool { self.had_error }

	pub fn diags(&self) -> &[Diagnostic] { &self.diagnostics }

	pub fn take(&mut self) -> Vec<Diagnostic> { std::mem::take(&mut self.diagnostics) }
}
