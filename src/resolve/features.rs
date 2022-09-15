use crate::{ast::Enable, diagnostic::Diagnostics, text::Interner};

#[derive(Copy, Clone, Debug, Default)]
pub struct EnabledFeatures {
	pub float16: bool,
	pub float64: bool,
	pub primitive_index: bool,
	pub binding_array: bool,
	pub push_constant: bool,
	pub storage_image_other_access: bool,
	pub multiview: bool,
}

impl EnabledFeatures {
	pub fn enable(&mut self, enable: Enable, intern: &Interner, diagnostics: &mut Diagnostics) {
		let s = intern.resolve(enable.name.name);
		match s {
			"f16" => self.float16 = true,
			"f64" => self.float64 = true,
			"primitive_index" => self.primitive_index = true,
			"binding_array" => self.binding_array = true,
			"push_constant" => self.push_constant = true,
			"storage_image_other_access" => self.storage_image_other_access = true,
			"multiview" => self.multiview = true,
			_ => diagnostics.push(enable.span.error("unknown feature") + enable.name.span.marker()),
		}
	}
}
