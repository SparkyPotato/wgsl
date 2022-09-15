use std::fmt::Display;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder};

use crate::ast::Expr;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum AccessMode {
	Read,
	Write,
	ReadWrite,
}

impl Display for AccessMode {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			AccessMode::Read => write!(f, "read"),
			AccessMode::Write => write!(f, "write"),
			AccessMode::ReadWrite => write!(f, "read_write"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum AddressSpace {
	Function,
	Private,
	Storage,
	Uniform,
	Workgroup,
	PushConstant,
}

impl Display for AddressSpace {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			AddressSpace::Function => write!(f, "function"),
			AddressSpace::Private => write!(f, "private"),
			AddressSpace::Storage => write!(f, "storage"),
			AddressSpace::Uniform => write!(f, "uniform"),
			AddressSpace::Workgroup => write!(f, "workgroup"),
			AddressSpace::PushConstant => write!(f, "push_constant"),
		}
	}
}

#[derive(Clone, Debug)]
pub enum AttributeType {
	Align(Expr),
	Binding(Expr),
	Builtin(Builtin),
	Compute,
	Const,
	Fragment,
	Group(Expr),
	Id(Expr),
	Interpolate(InterpolationType, InterpolationSample),
	Invariant,
	Location(Expr),
	Size(Expr),
	Vertex,
	WorkgroupSize(Expr, Option<Expr>, Option<Expr>),
}

impl Display for AttributeType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			AttributeType::Align(_) => write!(f, "align"),
			AttributeType::Binding(_) => write!(f, "binding"),
			AttributeType::Builtin(_) => write!(f, "builtin"),
			AttributeType::Compute => write!(f, "compute"),
			AttributeType::Const => write!(f, "const"),
			AttributeType::Fragment => write!(f, "fragment"),
			AttributeType::Group(_) => write!(f, "group"),
			AttributeType::Id(_) => write!(f, "id"),
			AttributeType::Interpolate(..) => write!(f, "interpolate"),
			AttributeType::Invariant => write!(f, "invariant"),
			AttributeType::Location(_) => write!(f, "location"),
			AttributeType::Size(_) => write!(f, "size"),
			AttributeType::Vertex => write!(f, "vertex"),
			AttributeType::WorkgroupSize(..) => write!(f, "workgroup_size"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum Builtin {
	FragDepth,
	FrontFacing,
	GlobalInvocationId,
	InstanceIndex,
	LocalInvocationId,
	LocalInvocationIndex,
	NumWorkgroups,
	Position,
	SampleIndex,
	SampleMask,
	VertexIndex,
	WorkgroupId,
	PrimitiveIndex,
	ViewIndex,
}

impl Display for Builtin {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			Builtin::FragDepth => write!(f, "frag_depth"),
			Builtin::FrontFacing => write!(f, "front_facing"),
			Builtin::GlobalInvocationId => write!(f, "global_invocation_id"),
			Builtin::InstanceIndex => write!(f, "instance_index"),
			Builtin::LocalInvocationId => write!(f, "local_invocation_id"),
			Builtin::LocalInvocationIndex => write!(f, "local_invocation_index"),
			Builtin::NumWorkgroups => write!(f, "num_workgroups"),
			Builtin::Position => write!(f, "position"),
			Builtin::SampleIndex => write!(f, "sample_index"),
			Builtin::SampleMask => write!(f, "sample_mask"),
			Builtin::VertexIndex => write!(f, "vertex_index"),
			Builtin::WorkgroupId => write!(f, "workgroup_id"),
			Builtin::PrimitiveIndex => write!(f, "primitive_index"),
			Builtin::ViewIndex => write!(f, "view_index"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum InterpolationSample {
	Center,
	Centroid,
	Sample,
}

impl Display for InterpolationSample {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			InterpolationSample::Center => write!(f, "center"),
			InterpolationSample::Centroid => write!(f, "centroid"),
			InterpolationSample::Sample => write!(f, "sample"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum InterpolationType {
	Flat,
	Linear,
	Perspective,
}

impl Display for InterpolationType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			InterpolationType::Flat => write!(f, "flat"),
			InterpolationType::Linear => write!(f, "linear"),
			InterpolationType::Perspective => write!(f, "perspective"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum PrimitiveType {
	I32,
	U32,
	F64,
	F32,
	F16,
	Bool,
	Infer,
}

impl Display for PrimitiveType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			PrimitiveType::I32 => write!(f, "i32"),
			PrimitiveType::U32 => write!(f, "u32"),
			PrimitiveType::F64 => write!(f, "f64"),
			PrimitiveType::F32 => write!(f, "f32"),
			PrimitiveType::F16 => write!(f, "f16"),
			PrimitiveType::Bool => write!(f, "bool"),
			PrimitiveType::Infer => write!(f, "_"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum VecType {
	Vec2,
	Vec3,
	Vec4,
}

impl Display for VecType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			VecType::Vec2 => write!(f, "vec2"),
			VecType::Vec3 => write!(f, "vec3"),
			VecType::Vec4 => write!(f, "vec4"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum MatType {
	Mat2x2,
	Mat2x3,
	Mat2x4,
	Mat3x2,
	Mat3x3,
	Mat3x4,
	Mat4x2,
	Mat4x3,
	Mat4x4,
}

impl Display for MatType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			MatType::Mat2x2 => write!(f, "mat2x2"),
			MatType::Mat2x3 => write!(f, "mat2x3"),
			MatType::Mat2x4 => write!(f, "mat2x4"),
			MatType::Mat3x2 => write!(f, "mat3x2"),
			MatType::Mat3x3 => write!(f, "mat3x3"),
			MatType::Mat3x4 => write!(f, "mat3x4"),
			MatType::Mat4x2 => write!(f, "mat4x2"),
			MatType::Mat4x3 => write!(f, "mat4x3"),
			MatType::Mat4x4 => write!(f, "mat4x4"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum SampledTextureType {
	Texture1d,
	Texture1dArray,
	Texture2d,
	TextureMultisampled2d,
	Texture2dArray,
	Texture3d,
	TextureCube,
	TextureCubeArray,
}

impl Display for SampledTextureType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			SampledTextureType::Texture1d => write!(f, "texture_1d"),
			SampledTextureType::Texture1dArray => write!(f, "texture_1d_array"),
			SampledTextureType::Texture2d => write!(f, "texture_2d"),
			SampledTextureType::TextureMultisampled2d => write!(f, "texture_multisampled_2d"),
			SampledTextureType::Texture2dArray => write!(f, "texture_2d_array"),
			SampledTextureType::Texture3d => write!(f, "texture_3d"),
			SampledTextureType::TextureCube => write!(f, "texture_cube"),
			SampledTextureType::TextureCubeArray => write!(f, "texture_cube_array"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum DepthTextureType {
	Depth2d,
	Depth2dArray,
	DepthCube,
	DepthCubeArray,
	DepthMultisampled2d,
}

impl Display for DepthTextureType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			DepthTextureType::Depth2d => write!(f, "texture_depth_2d"),
			DepthTextureType::Depth2dArray => write!(f, "texture_depth_2d_array"),
			DepthTextureType::DepthCube => write!(f, "texture_depth_cube"),
			DepthTextureType::DepthCubeArray => write!(f, "texture_depth_cube_array"),
			DepthTextureType::DepthMultisampled2d => write!(f, "texture_depth_multisampled_2d"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum SamplerType {
	Sampler,
	SamplerComparison,
}

impl Display for SamplerType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			SamplerType::Sampler => write!(f, "sampler"),
			SamplerType::SamplerComparison => write!(f, "sampler_comparison"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum StorageTextureType {
	Storage1d,
	Storage1dArray,
	Storage2d,
	Storage2dArray,
	Storage3d,
}

impl Display for StorageTextureType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			StorageTextureType::Storage1d => write!(f, "storage_1d"),
			StorageTextureType::Storage1dArray => write!(f, "storage_1d_array"),
			StorageTextureType::Storage2d => write!(f, "storage_2d"),
			StorageTextureType::Storage2dArray => write!(f, "storage_2d_array"),
			StorageTextureType::Storage3d => write!(f, "storage_3d"),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum TexelFormat {
	R32Float,
	R32Sint,
	R32Uint,
	Rg32Float,
	Rg32Sint,
	Rg32Uint,
	Rgba16Float,
	Rgba16Sint,
	Rgba16Uint,
	Rgba32Float,
	Rgba32Sint,
	Rgba32Uint,
	Rgba8Sint,
	Rgba8Uint,
	Rgba8Unorm,
	Rgba8Snorm,
}

impl Display for TexelFormat {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			TexelFormat::R32Float => write!(f, "r32float"),
			TexelFormat::R32Sint => write!(f, "r32sint"),
			TexelFormat::R32Uint => write!(f, "r32uint"),
			TexelFormat::Rg32Float => write!(f, "rg32float"),
			TexelFormat::Rg32Sint => write!(f, "rg32sint"),
			TexelFormat::Rg32Uint => write!(f, "rg32uint"),
			TexelFormat::Rgba16Float => write!(f, "rgba16float"),
			TexelFormat::Rgba16Sint => write!(f, "rgba16sint"),
			TexelFormat::Rgba16Uint => write!(f, "rgba16uint"),
			TexelFormat::Rgba32Float => write!(f, "rgba32float"),
			TexelFormat::Rgba32Sint => write!(f, "rgba32sint"),
			TexelFormat::Rgba32Uint => write!(f, "rgba32uint"),
			TexelFormat::Rgba8Sint => write!(f, "rgba8sint"),
			TexelFormat::Rgba8Uint => write!(f, "rgba8uint"),
			TexelFormat::Rgba8Unorm => write!(f, "rgba8unorm"),
			TexelFormat::Rgba8Snorm => write!(f, "rgba8snorm"),
		}
	}
}

pub fn reserved_words_and_keywords() -> &'static [&'static str] {
	&[
		"array",
		"atomic",
		"bool",
		"f32",
		"f16",
		"i32",
		"mat2x2",
		"mat2x3",
		"mat2x4",
		"mat3x2",
		"mat3x3",
		"mat3x4",
		"mat4x2",
		"mat4x3",
		"mat4x4",
		"ptr",
		"sampler",
		"sampler_comparison",
		"texture_1d",
		"texture_1d_array",
		"texture_2d",
		"texture_2d_array",
		"texture_3d",
		"texture_cube",
		"texture_cube_array",
		"texture_multisampled_2d",
		"texture_storage_1d",
		"texture_storage_1d_array",
		"texture_storage_2d",
		"texture_storage_2d_array",
		"texture_storage_3d",
		"texture_depth_2d",
		"texture_depth_2d_array",
		"texture_depth_cube",
		"texture_depth_cube_array",
		"texture_depth_multisampled_2d",
		"u32",
		"vec2",
		"vec3",
		"vec4",
		"bitcast",
		"break",
		"case",
		"const",
		"continue",
		"default",
		"discard",
		"else",
		"enable",
		"false",
		"fn",
		"for",
		"function",
		"if",
		"let",
		"loop",
		"override",
		"private",
		"return",
		"static_assert",
		"storage",
		"struct",
		"switch",
		"true",
		"type",
		"uniform",
		"var",
		"while",
		"workgroup",
		"CompileShader",
		"ComputeShader",
		"DomainShader",
		"GeometryShader",
		"HullShader",
		"NULL",
		"Self",
		"abstract",
		"active",
		"alignas",
		"alignof",
		"as",
		"asm",
		"asm_fragment",
		"async",
		"attribute",
		"auto",
		"await",
		"become",
		"binding_array",
		"cast",
		"catch",
		"class",
		"co_await",
		"co_return",
		"co_yield",
		"coherent",
		"column_major",
		"common",
		"compile",
		"compile_fragment",
		"concept",
		"const_cast",
		"consteval",
		"constexpr",
		"consinit",
		"crate",
		"debugger",
		"decltype",
		"delete",
		"demote",
		"demote_to_helper",
		"do",
		"dynamic_cast",
		"enum",
		"explicit",
		"export",
		"extends",
		"extern",
		"external",
		"fallthrough",
		"filter",
		"final",
		"finally",
		"friend",
		"from",
		"fxgroup",
		"get",
		"goto",
		"groupshared",
		"handle",
		"highp",
		"impl",
		"implements",
		"import",
		"inline",
		"inout",
		"instanceof",
		"interface",
		"layout",
		"line",
		"lineadj",
		"lowp",
		"macro",
		"macro_rules",
		"match",
		"mediump",
		"meta",
		"mod",
		"module",
		"move",
		"mut",
		"mutable",
		"namespace",
		"new",
		"nil",
		"noexcept",
		"noinline",
		"nointerpolation",
		"noperspective",
		"null",
		"nullptr",
		"of",
		"operator",
		"package",
		"packoffset",
		"partition",
		"pass",
		"patch",
		"pixelfragment",
		"point",
		"precise",
		"precision",
		"premerge",
		"priv",
		"protected",
		"pub",
		"public",
		"readonly",
		"ref",
		"regardless",
		"register",
		"reinterpret_cast",
		"requires",
		"resource",
		"restrict",
		"self",
		"set",
		"shared",
		"signed",
		"sizeof",
		"smooth",
		"snorm",
		"static",
		"static_assert",
		"static_cast",
		"std",
		"subroutine",
		"super",
		"target",
		"template",
		"this",
		"thread_local",
		"throw",
		"trait",
		"try",
		"typedef",
		"typeid",
		"typeof",
		"union",
		"unless",
		"unorm",
		"unsafe",
		"unsized",
		"use",
		"using",
		"varying",
		"virtual",
		"volatile",
		"wgsl",
		"where",
		"with",
		"writeonly",
		"yield",
	]
}

pub fn reserved_matcher() -> AhoCorasick {
	AhoCorasickBuilder::new()
		.anchored(true)
		.build(reserved_words_and_keywords())
}
