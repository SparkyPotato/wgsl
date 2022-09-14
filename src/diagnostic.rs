use std::ops::{Add, Range};

use crate::text::Text;

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
}

impl Add<Label> for Diagnostic {
	type Output = Diagnostic;

	fn add(mut self, other: Label) -> Self::Output {
		self.labels.push(other);
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
	pub file: Text,
}

impl Add for Span {
	type Output = Span;

	fn add(self, other: Span) -> Self::Output {
		debug_assert_eq!(self.file, other.file);
		Span {
			start: self.start.min(other.start),
			end: self.end.max(other.end),
			file: self.file,
		}
	}
}

impl chumsky::Span for Span {
	type Context = Text;
	type Offset = u32;

	fn new(context: Self::Context, range: Range<Self::Offset>) -> Self {
		Self {
			start: range.start,
			end: range.end,
			file: context,
		}
	}

	fn context(&self) -> Self::Context { self.file }

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
		}
	}

	pub fn warning(&self, message: impl Into<String>) -> Diagnostic {
		Diagnostic {
			kind: DiagnosticKind::Warning,
			message: message.into(),
			span: *self,
			labels: Vec::new(),
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
}
