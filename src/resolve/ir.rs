use crate::{
	ast::{AssignOp, BinaryOp, Ident, Literal, PostfixOp, UnaryOp},
	diagnostic::Span,
	resolve::{
		features::EnabledFeatures,
		inbuilt::{
			AccessMode,
			AddressSpace,
			AttributeType,
			DepthTextureType,
			MatType,
			PrimitiveType,
			SampledTextureType,
			SamplerType,
			StorageTextureType,
			TexelFormat,
			VecType,
		},
		inbuilt_functions::InbuiltFunction,
	},
};

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct DeclId(pub u32);
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct LocalId(pub u32);

#[derive(Clone, Debug, Default)]
pub struct TranslationUnit {
	pub features: EnabledFeatures,
	pub decls: Vec<Decl>,
}

#[derive(Clone, Debug)]
pub struct Attribute {
	pub ty: AttributeType,
	pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Decl {
	pub kind: DeclKind,
	pub span: Span,
}

#[derive(Clone, Debug)]
pub enum DeclKind {
	Fn(Fn),
	Override(Override),
	Var(Var),
	Let(Let),
	Const(Let),
	StaticAssert(Expr),
	Struct(Struct),
	Type(TypeDecl),
}

#[derive(Clone, Debug)]
pub struct Fn {
	pub attribs: Vec<Attribute>,
	pub name: Ident,
	pub args: Vec<Arg>,
	pub ret_attribs: Vec<Attribute>,
	pub ret: Option<Type>,
	pub block: Block,
}

#[derive(Clone, Debug)]
pub struct Override {
	pub attribs: Vec<Attribute>,
	pub name: Ident,
	pub ty: Option<Type>,
	pub val: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct Var {
	pub attribs: Vec<Attribute>,
	pub inner: VarNoAttribs,
}

#[derive(Clone, Debug)]
pub struct VarNoAttribs {
	pub address_space: Option<AddressSpace>,
	pub access_mode: Option<AccessMode>,
	pub name: Ident,
	pub ty: Option<Type>,
	pub val: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct Struct {
	pub name: Ident,
	pub fields: Vec<Arg>,
}

#[derive(Clone, Debug)]
pub struct TypeDecl {
	pub name: Ident,
	pub ty: Type,
}

#[derive(Clone, Debug)]
pub struct Arg {
	pub attribs: Vec<Attribute>,
	pub name: Ident,
	pub ty: Type,
	pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Type {
	pub kind: TypeKind,
	pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TypeKind {
	Inbuilt(InbuiltType),
	User(DeclId),
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum FloatType {
	F16,
	F32,
	F64,
	Infer,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum SampleType {
	F64,
	F32,
	I32,
	U32,
}

#[derive(Clone, Debug)]
pub enum InbuiltType {
	AbstractInt,
	AbstractFloat,
	Primitive(PrimitiveType),
	Vec {
		ty: PrimitiveType,
		comp: VecType,
	},
	Mat {
		ty: FloatType,
		comp: MatType,
	},
	SampledTexture(SampledTextureType, SampleType),
	DepthTexture(DepthTextureType),
	StorageTexture(StorageTextureType, TexelFormat, AccessMode),
	Sampler(SamplerType),
	Array {
		of: Box<Type>,
		len: Option<Expr>,
	},
	BindingArray {
		of: Box<Type>,
		len: Option<Expr>,
	},
	CompareExchangeResult {
		signed: bool, // Only i32 and u32 are allowed.
	},
	Ptr {
		to: Box<Type>,
		address_space: AddressSpace,
		access_mode: AccessMode,
	},
	Atomic {
		signed: bool, // Only i32 and u32 are allowed.
	},
}

#[derive(Clone, Debug)]
pub struct Block {
	pub stmts: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub struct Stmt {
	pub kind: StmtKind,
	pub span: Span,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
	Expr(ExprStatementKind),
	Block(Block),
	Break,
	Continue,
	Discard,
	For(For),
	If(If),
	Loop(Block),
	Return(Option<Expr>),
	StaticAssert(Expr),
	Switch(Switch),
	While(While),
	Continuing(Block),
	BreakIf(Expr),
}

#[derive(Clone, Debug)]
pub struct ExprStatement {
	pub kind: ExprStatementKind,
	pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprStatementKind {
	VarDecl(VarDecl),
	Call(CallExpr),
	IgnoreExpr(Expr),
	Assign(AssignExpr),
	Postfix(PostfixExpr),
}

#[derive(Clone, Debug)]
pub struct CallStmt {
	pub name: Ident,
	pub args: Vec<Expr>,
}

#[derive(Clone, Debug)]
pub struct For {
	pub init: Option<ExprStatement>,
	pub cond: Option<Expr>,
	pub update: Option<ExprStatement>, // var decls are not allowed here.
	pub block: Block,
}

#[derive(Clone, Debug)]
pub struct If {
	pub cond: Expr,
	pub block: Block,
	pub else_: Option<Box<Stmt>>,
}

#[derive(Clone, Debug)]
pub struct While {
	pub cond: Expr,
	pub block: Block,
}

#[derive(Clone, Debug)]
pub struct Switch {
	pub expr: Expr,
	pub cases: Vec<Case>,
}

#[derive(Clone, Debug)]
pub struct Case {
	pub selectors: Vec<CaseSelector>,
	pub block: Block,
	pub span: Span,
}

#[derive(Clone, Debug)]
pub enum CaseSelector {
	Expr(Expr),
	Default,
}

#[derive(Clone, Debug)]
pub struct Expr {
	pub kind: ExprKind,
	pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
	Error,
	Literal(Literal),
	Local(LocalId),
	Global(DeclId),
	Unary(UnaryExpr),
	Binary(BinaryExpr),
	Call(CallExpr),
	Index(Box<Expr>, Box<Expr>),
	Member(Box<Expr>, Ident),
}

#[derive(Clone, Debug)]
pub struct UnaryExpr {
	pub op: UnaryOp,
	pub expr: Box<Expr>,
}

#[derive(Clone, Debug)]
pub struct AssignExpr {
	pub lhs: Box<Expr>,
	pub op: AssignOp,
	pub rhs: Box<Expr>,
}

#[derive(Clone, Debug)]
pub struct CallExpr {
	pub target: FnTarget,
	pub args: Vec<Expr>,
}

#[derive(Clone, Debug)]
pub enum FnTarget {
	Decl(DeclId),
	InbuiltFunction(InbuiltFunction),
	InbuiltType(Box<InbuiltType>),
	Error,
}

#[derive(Clone, Debug)]
pub struct BinaryExpr {
	pub lhs: Box<Expr>,
	pub op: BinaryOp,
	pub rhs: Box<Expr>,
}

#[derive(Clone, Debug)]
pub struct PostfixExpr {
	pub expr: Box<Expr>,
	pub op: PostfixOp,
}

#[derive(Clone, Debug)]
pub struct VarDecl {
	pub kind: VarDeclKind,
	pub local: LocalId,
}

#[derive(Clone, Debug)]
pub enum VarDeclKind {
	Var(VarNoAttribs),
	Const(Let),
	Let(Let),
}

#[derive(Clone, Debug)]
pub struct Let {
	pub name: Ident,
	pub ty: Option<Type>,
	pub val: Expr,
}
