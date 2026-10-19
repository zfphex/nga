use crate::TypeAbi;

pub const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prim {
    Bool,
    F16,
    F32,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HandleKind {
    Texture2d,
    Texture2dRW,
    Texture3d,
    Texture3dRW,
    Texture2dArray,
    TextureCube,
    TextureCubeRW,
    Depth2d,
    Sampler,
    Commands,
    Pipeline,
    Opaque,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Ty {
    Void,
    Scalar(Prim),
    Vec(Prim, u8),
    PackedVec(Prim, u8),
    Mat(u8, u8),
    Array(&'static Ty, u32),
    Struct(&'static TypeAbi),
    Ptr { mutable: bool, pointee: &'static Ty },
    Handle(HandleKind),
    ThreadgroupAtomicU32,
    Infer,
    IntLit,
    FloatLit,
}

impl Ty {
    pub const F32: Ty = Ty::Scalar(Prim::F32);
    pub const I32: Ty = Ty::Scalar(Prim::I32);
    pub const U32: Ty = Ty::Scalar(Prim::U32);
    pub const BOOL: Ty = Ty::Scalar(Prim::Bool);

    pub fn prim(self) -> Prim {
        match self {
            Ty::Scalar(p) | Ty::Vec(p, _) | Ty::PackedVec(p, _) => p,
            Ty::Mat(..) | Ty::FloatLit => Prim::F32,
            Ty::IntLit => Prim::I32,
            _ => panic!("{self:?} has no scalar type"),
        }
    }

    pub fn size(self) -> usize {
        match self {
            Ty::Void => 0,
            Ty::Scalar(p) => prim_size(p),
            Ty::Vec(p, 3) => prim_size(p) * 4,
            Ty::Vec(p, n) | Ty::PackedVec(p, n) => prim_size(p) * n as usize,
            Ty::Mat(c, 3) => 16 * c as usize,
            Ty::Mat(c, r) => 4 * c as usize * r as usize,
            Ty::Array(e, n) => e.size() * n as usize,
            Ty::Struct(abi) => abi.size,
            Ty::Ptr { .. } | Ty::Handle(_) => 8,
            Ty::ThreadgroupAtomicU32 => 4,
            Ty::Infer | Ty::IntLit | Ty::FloatLit => panic!("{self:?} has no size"),
        }
    }

    pub fn align(self) -> usize {
        match self {
            Ty::Vec(p, 3) => prim_size(p) * 4,
            Ty::Vec(p, n) => prim_size(p) * n as usize,
            Ty::PackedVec(p, _) => prim_size(p),
            Ty::Scalar(p) => prim_size(p),
            Ty::Mat(_, 3) => 16,
            Ty::Mat(..) => 4,
            Ty::Array(e, _) => e.align(),
            Ty::Struct(abi) => abi.align,
            Ty::Ptr { .. } | Ty::Handle(_) => 8,
            Ty::ThreadgroupAtomicU32 => 4,
            _ => panic!("{self:?} has no alignment"),
        }
    }

    pub fn deref(self) -> Ty {
        match self {
            Ty::Ptr { pointee, .. } => *pointee,
            other => other,
        }
    }
}

pub fn prim_size(p: Prim) -> usize {
    match p {
        Prim::I8 | Prim::U8 => 1,
        Prim::F16 | Prim::I16 | Prim::U16 => 2,
        Prim::Bool | Prim::F32 | Prim::I32 | Prim::U32 => 4,
        Prim::I64 | Prim::U64 => 8,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    AndAnd,
    OrOr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinOp {
    pub fn compare(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }

    pub fn logical(self) -> bool {
        matches!(self, BinOp::AndAnd | BinOp::OrOr)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Intrinsic {
    Unpack,
    Sqrt,
    InverseSqrt,
    Abs,
    Sign,
    Floor,
    Ceil,
    Round,
    Fract,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Exp,
    Exp2,
    Log,
    Log2,
    Pow,
    Min,
    Max,
    Clamp,
    Mix,
    Step,
    SmoothStep,
    Dot,
    Cross,
    Normalize,
    Length,
    Distance,
    Reflect,
    Select,
    Any,
    All,
    Mul,
    Transpose,
    Barrier,
    ThreadgroupBarrier,
    AtomicAdd,
    AtomicMin,
    AtomicMax,
    AtomicExchange,
    Offset,
    Sample,
    SampleLevel,
    Load,
    Store,
    Dimensions,
    Dfdx,
    Dfdy,
    MeshSetThreadgroups,
    MeshSetPrimitiveCount,
    MeshSetVertex,
    MeshSetIndex,
    TileRead,
    TileWrite,
    CommandSetPipeline,
    CommandSetVertexBuffer,
    CommandDraw,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Const {
    Bool(bool),
    Int(i64),
    Float(f64),
}

#[derive(Clone, Copy, Debug)]
pub enum Node {
    Block(&'static [u32]),
    Let {
        local: u32,
        value: u32,
    },
    Store {
        place: u32,
        value: u32,
    },
    If {
        cond: u32,
        then: u32,
        els: u32,
    },
    While {
        cond: u32,
        body: u32,
    },
    Loop {
        body: u32,
    },

    Range {
        local: u32,
        start: u32,
        end: u32,
        body: u32,
    },
    Break,
    Continue,
    Return(u32),
    Discard,

    Local(u32),
    Param(u32),
    Const(Const),

    Field {
        base: u32,
        name: &'static str,
    },
    Index {
        base: u32,
        index: u32,
    },
    Unary(UnOp, u32),
    Binary(BinOp, u32, u32),
    Cast {
        value: u32,
        to: Ty,
    },

    Construct {
        ty: Ty,
        args: &'static [u32],
    },

    StructLit {
        ty: Ty,
        fields: &'static [(&'static str, u32)],
    },
    Call {
        func: u32,
        args: &'static [u32],
    },
    Intrinsic {
        op: Intrinsic,
        args: &'static [u32],
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Builtin {
    ThreadId,

    ThreadIdInGroup,

    GroupId,

    ThreadIndexInGroup,
    VertexId,
    InstanceId,

    BaseInstance,
    PrimitiveId,

    Position,

    RenderTargetArrayIndex,
    FrontFacing,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Binding {
    None,

    Root,
    Builtin(Builtin),

    Location(u32),

    Constants,

    Payload,

    Buffer(u32),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Stage {
    Helper,
    Vertex,
    Fragment,
    Compute([u32; 3]),
    Object,
    Mesh {
        vertex: Ty,
        max_vertices: u32,
        max_primitives: u32,
    },
    Tile {
        imageblock: Ty,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Param {
    pub name: &'static str,
    pub ty: Ty,
    pub binding: Binding,
}

#[derive(Clone, Copy, Debug)]
pub struct Local {
    pub name: &'static str,

    pub ty: Ty,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuFn {
    pub name: &'static str,
    pub stage: Stage,
    pub params: &'static [Param],
    pub ret: Ty,
    pub locals: &'static [Local],
    pub nodes: &'static [Node],

    pub body: u32,
    pub calls: &'static [&'static GpuFn],
}

impl GpuFn {
    pub fn reachable(&'static self) -> Vec<&'static GpuFn> {
        let mut out = Vec::new();
        visit(self, &mut out);
        out
    }
}

fn visit(f: &'static GpuFn, out: &mut Vec<&'static GpuFn>) {
    if out.iter().any(|other| std::ptr::eq(*other, f)) {
        return;
    }
    for callee in f.calls {
        visit(callee, out);
    }
    out.push(f);
}

pub struct Types {
    pub nodes: Vec<Ty>,
    pub locals: Vec<Ty>,
}

pub fn infer(f: &GpuFn) -> Types {
    let mut types = Types {
        nodes: vec![Ty::Infer; f.nodes.len()],
        locals: f.locals.iter().map(|l| l.ty).collect(),
    };
    stmt(f, &mut types, f.body);

    for ty in &mut types.nodes {
        match ty {
            Ty::IntLit => *ty = Ty::I32,
            Ty::FloatLit => *ty = Ty::F32,
            _ => {}
        }
    }
    types
}

fn stmt(f: &GpuFn, types: &mut Types, node: u32) {
    if node == NONE {
        return;
    }
    match f.nodes[node as usize] {
        Node::Block(stmts) => {
            for &s in stmts {
                stmt(f, types, s);
            }
        }
        Node::Let { local, value } => {
            let ty = expr(f, types, value);
            match types.locals[local as usize] {
                Ty::Infer => types.locals[local as usize] = concrete(ty),
                annotated => coerce(f, types, value, annotated),
            }
        }
        Node::Store { place, value } => {
            let ty = expr(f, types, place);
            expr(f, types, value);
            coerce(f, types, value, ty);
        }
        Node::If { cond, then, els } => {
            expr(f, types, cond);
            stmt(f, types, then);
            stmt(f, types, els);
        }
        Node::While { cond, body } => {
            expr(f, types, cond);
            stmt(f, types, body);
        }
        Node::Loop { body } => stmt(f, types, body),
        Node::Range {
            local,
            start,
            end,
            body,
        } => {
            let start_ty = expr(f, types, start);
            let end_ty = expr(f, types, end);
            let ty = match (start_ty, end_ty) {
                (Ty::IntLit | Ty::FloatLit, other)
                    if !matches!(other, Ty::Infer | Ty::IntLit | Ty::FloatLit) =>
                {
                    other
                }
                (other, Ty::IntLit | Ty::FloatLit)
                    if !matches!(other, Ty::Infer | Ty::IntLit | Ty::FloatLit) =>
                {
                    other
                }
                _ => concrete(start_ty),
            };
            coerce(f, types, start, ty);
            coerce(f, types, end, ty);
            if types.locals[local as usize] == Ty::Infer {
                types.locals[local as usize] = ty;
            }
            stmt(f, types, body);
        }
        Node::Return(value) => {
            if value != NONE {
                expr(f, types, value);
                coerce(f, types, value, f.ret);
            }
        }
        Node::Break | Node::Continue | Node::Discard => {}

        _ => {
            expr(f, types, node);
        }
    }
}

fn concrete(ty: Ty) -> Ty {
    match ty {
        Ty::IntLit => Ty::I32,
        Ty::FloatLit => Ty::F32,
        other => other,
    }
}

fn coerce(f: &GpuFn, types: &mut Types, node: u32, want: Ty) {
    if !matches!(types.nodes[node as usize], Ty::IntLit | Ty::FloatLit) {
        return;
    }
    types.nodes[node as usize] = match want {
        Ty::Vec(p, _) | Ty::PackedVec(p, _) => Ty::Scalar(p),
        other => other,
    };

    if let Node::Construct { args, .. } | Node::Intrinsic { args, .. } = f.nodes[node as usize] {
        for &arg in args {
            coerce(f, types, arg, want);
        }
    }
}

fn expr(f: &GpuFn, types: &mut Types, node: u32) -> Ty {
    let ty = match f.nodes[node as usize] {
        Node::Local(i) => types.locals[i as usize],
        Node::Param(i) => f.params[i as usize].ty,
        Node::Const(Const::Bool(_)) => Ty::BOOL,
        Node::Const(Const::Int(_)) => Ty::IntLit,
        Node::Const(Const::Float(_)) => Ty::FloatLit,
        Node::Field { base, name } => {
            let base = expr(f, types, base).deref();
            match base {
                Ty::Vec(p, _) if name.len() == 1 => Ty::Scalar(p),
                Ty::Vec(p, _) => Ty::Vec(p, name.len() as u8),
                Ty::PackedVec(p, _) if name.len() == 1 => Ty::Scalar(p),
                Ty::PackedVec(p, _) => Ty::Vec(p, name.len() as u8),
                Ty::Mat(_, r) => Ty::Vec(Prim::F32, r),
                Ty::Struct(abi) => field(abi, name).ty,
                other => panic!("`.{name}` on {other:?}"),
            }
        }
        Node::Index { base, index } => {
            expr(f, types, index);
            coerce(f, types, index, Ty::U32);
            match expr(f, types, base) {
                Ty::Ptr { pointee, .. } => *pointee,
                Ty::Array(e, _) => *e,
                Ty::Vec(p, _) => Ty::Scalar(p),
                Ty::PackedVec(p, _) => Ty::Scalar(p),
                Ty::Mat(_, r) => Ty::Vec(Prim::F32, r),
                other => other,
            }
        }
        Node::Unary(_, value) => expr(f, types, value),
        Node::Binary(op, lhs, rhs) => {
            let (l, r) = (expr(f, types, lhs), expr(f, types, rhs));
            coerce(f, types, lhs, r);
            coerce(f, types, rhs, l);
            let (l, r) = (types.nodes[lhs as usize], types.nodes[rhs as usize]);
            if op.compare() {
                match l {
                    Ty::Vec(_, n) => Ty::Vec(Prim::Bool, n),
                    _ => Ty::BOOL,
                }
            } else if op.logical() {
                Ty::BOOL
            } else if matches!(l, Ty::Mat(..)) || matches!(r, Ty::Mat(..)) {
                panic!("matrices multiply with `mul`, not `*`")
            } else {
                match (l, r) {
                    (Ty::Vec(..), _) => l,
                    (_, Ty::Vec(..)) => r,
                    _ => l,
                }
            }
        }
        Node::Cast { value, to } => {
            expr(f, types, value);
            to
        }
        Node::Construct { ty, args } => {
            for &arg in args {
                expr(f, types, arg);
                coerce(f, types, arg, ty);
            }
            ty
        }
        Node::StructLit { ty, fields } => {
            let Ty::Struct(abi) = ty else {
                panic!("{ty:?} is not a struct");
            };
            for &(name, value) in fields {
                expr(f, types, value);
                coerce(f, types, value, field(abi, name).ty);
            }
            ty
        }
        Node::Call { func, args } => {
            let callee = f.calls[func as usize];
            for (i, &arg) in args.iter().enumerate() {
                expr(f, types, arg);
                coerce(f, types, arg, callee.params[i].ty);
            }
            callee.ret
        }
        Node::Intrinsic { op, args } => {
            let mut first = Ty::Void;
            for (i, &arg) in args.iter().enumerate() {
                let ty = expr(f, types, arg);
                if i == 0 {
                    first = ty;
                }
            }

            let seed = concrete(first);
            if matches!(seed, Ty::Scalar(_) | Ty::Vec(..) | Ty::Mat(..)) {
                for &arg in args {
                    coerce(f, types, arg, seed);
                }
            } else if op == Intrinsic::Offset && args.len() > 1 {
                coerce(f, types, args[1], Ty::U32);
            }
            intrinsic(op, seed, f, types, args)
        }
        other => panic!("{other:?} is not an expression"),
    };
    types.nodes[node as usize] = ty;
    ty
}

fn intrinsic(op: Intrinsic, first: Ty, f: &GpuFn, types: &Types, args: &[u32]) -> Ty {
    let _ = f;
    match op {
        Intrinsic::Unpack => match first {
            Ty::PackedVec(p, n) => Ty::Vec(p, n),
            other => panic!("unpack({other:?})"),
        },
        Intrinsic::Dot | Intrinsic::Length | Intrinsic::Distance => Ty::Scalar(first.prim()),
        Intrinsic::Any | Intrinsic::All => Ty::BOOL,
        Intrinsic::Barrier
        | Intrinsic::ThreadgroupBarrier
        | Intrinsic::Store
        | Intrinsic::MeshSetThreadgroups
        | Intrinsic::MeshSetPrimitiveCount
        | Intrinsic::MeshSetVertex
        | Intrinsic::MeshSetIndex
        | Intrinsic::TileWrite
        | Intrinsic::CommandSetPipeline
        | Intrinsic::CommandSetVertexBuffer
        | Intrinsic::CommandDraw => Ty::Void,
        Intrinsic::TileRead => match f.stage {
            Stage::Tile { imageblock } => imageblock,
            _ => panic!("tile_read outside a tile stage"),
        },

        Intrinsic::Mul => match (first, types.nodes[args[1] as usize]) {
            (Ty::Mat(_, r), Ty::Vec(..)) => Ty::Vec(Prim::F32, r),
            (Ty::Vec(..), Ty::Mat(c, _)) => Ty::Vec(Prim::F32, c),
            (Ty::Mat(_, r), Ty::Mat(c, _)) => Ty::Mat(c, r),
            (a, b) => panic!("mul({a:?}, {b:?})"),
        },
        Intrinsic::Transpose => match first {
            Ty::Mat(c, r) => Ty::Mat(r, c),
            other => panic!("transpose({other:?})"),
        },
        Intrinsic::Sample | Intrinsic::SampleLevel | Intrinsic::Load => Ty::Vec(Prim::F32, 4),
        Intrinsic::Dimensions => Ty::Vec(Prim::U32, 2),
        Intrinsic::AtomicAdd
        | Intrinsic::AtomicMin
        | Intrinsic::AtomicMax
        | Intrinsic::AtomicExchange => Ty::U32,

        Intrinsic::Select => first,
        Intrinsic::Cross => Ty::Vec(Prim::F32, 3),

        _ => first,
    }
}

pub fn field(abi: &'static TypeAbi, name: &str) -> &'static crate::FieldAbi {
    abi.fields
        .iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("`{}` has no field `{name}`", abi.name))
}
