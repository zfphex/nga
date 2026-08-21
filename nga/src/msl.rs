//! The Metal backend: `GpuFn` in, MSL source out.
//!
//! A recursive walk of the arena, printing as it goes. The only structural work is padding:
//! struct members land on the byte offsets `#[gpu_struct]` recorded from Rust, so a `uint`
//! array fills every gap. Rust's layout is authoritative and MSL is made to match, which is the
//! same contract `slang.rs` had, minus the guessing — the offsets are stated, not arranged.

use crate::ir::{self, BinOp, Builtin, Const, GpuFn, HandleKind, Intrinsic, NONE, Node, Prim, Stage, Ty, UnOp};
use crate::TypeAbi;

pub const PRELUDE: &str = include_str!("prelude.metal");

/// Everything one `MTLLibrary` holds: the entry points asked for, and every function they reach.
pub fn module(entries: &[&'static GpuFn]) -> String {
    let mut functions: Vec<&'static GpuFn> = Vec::new();
    for entry in entries {
        for f in entry.reachable() {
            if !functions.iter().any(|g| std::ptr::eq(*g, f)) {
                functions.push(f);
            }
        }
    }

    // A struct reached through a pointer or a local is memory and gets padded; one a vertex
    // stage returns or a fragment stage takes is an interface and gets attributes instead.
    let mut io: Vec<&'static TypeAbi> = Vec::new();
    for f in &functions {
        if f.stage == Stage::Vertex && let Ty::Struct(abi) = f.ret {
            io.push(abi);
        }
        if f.stage == Stage::Fragment {
            for param in f.params {
                if let Ty::Struct(abi) = param.ty {
                    io.push(abi);
                }
            }
        }
    }

    let mut structs: Vec<&'static TypeAbi> = Vec::new();
    let mut types = Vec::new();
    for f in &functions {
        let t = ir::infer(f);
        // A constants block is not memory and gets no declaration; its fields are the module's
        // function constants and nothing ever holds the struct itself.
        let params = f
            .params
            .iter()
            .filter(|p| p.binding != ir::Binding::Constants)
            .map(|p| p.ty);
        for ty in params.chain([f.ret]).chain(t.locals.iter().copied()).chain(t.nodes.iter().copied()) {
            collect(ty, &mut structs);
        }
        types.push(t);
    }

    let mut out = String::from(PRELUDE);
    // Function constants are module scope in MSL, and their index is the field's own, which is
    // the number `Constants::values` pushes on the host side.
    let mut declared: Vec<&'static TypeAbi> = Vec::new();
    for f in &functions {
        for param in f.params {
            let (ir::Binding::Constants, Ty::Struct(abi)) = (param.binding, param.ty) else {
                continue;
            };
            if declared.iter().any(|other| std::ptr::eq(*other, abi)) {
                continue;
            }
            declared.push(abi);
            out.push('\n');
            for (index, field) in abi.fields.iter().enumerate() {
                out += &format!(
                    "constant {} [[function_constant({index})]];\n",
                    decl(field.ty, &constant_name(abi, field.name), false)
                );
            }
        }
    }
    for abi in &structs {
        let interface = io.iter().any(|other| std::ptr::eq(*other, *abi));
        out += &declare(abi, interface);
    }
    for (f, types) in functions.iter().zip(&types) {
        out += &function(f, types, &io);
    }
    out
}

fn collect(ty: Ty, out: &mut Vec<&'static TypeAbi>) {
    match ty {
        Ty::Struct(abi) => {
            if out.iter().any(|other| std::ptr::eq(*other, abi)) {
                return;
            }
            for field in abi.fields {
                collect(field.ty, out);
            }
            out.push(abi);
        }
        Ty::Array(element, _) => collect(*element, out),
        Ty::Ptr { pointee, .. } => collect(*pointee, out),
        _ => {}
    }
}

// -- types -------------------------------------------------------------------------------

fn prim(p: Prim) -> &'static str {
    match p {
        Prim::Bool => "bool",
        Prim::F16 => "half",
        Prim::F32 => "float",
        Prim::I8 => "char",
        Prim::U8 => "uchar",
        Prim::I16 => "short",
        Prim::U16 => "ushort",
        Prim::I32 => "int",
        Prim::U32 => "uint",
        Prim::I64 => "long",
        Prim::U64 => "ulong",
    }
}

fn handle(kind: HandleKind) -> &'static str {
    match kind {
        HandleKind::Texture2d => "nga_texture2d",
        HandleKind::Texture2dRW => "nga_texture2d_rw",
        HandleKind::Texture3d => "nga_texture3d",
        HandleKind::Texture3dRW => "nga_texture3d_rw",
        HandleKind::Texture2dArray => "nga_texture2d_array",
        HandleKind::TextureCube => "nga_texturecube",
        HandleKind::Depth2d => "nga_depth2d",
        HandleKind::Sampler => "nga_sampler",
        HandleKind::Opaque => "nga_opaque",
    }
}

/// `packed` is the difference between a struct member and everything else: a three-vector is
/// twelve bytes in memory and sixteen in a register, and only the memory one may be indexed by
/// a byte offset Rust chose.
fn name(ty: Ty, packed: bool) -> String {
    match ty {
        Ty::Void => "void".into(),
        Ty::Scalar(p) => prim(p).into(),
        Ty::Vec(p, 3) if packed => format!("packed_{}3", prim(p)),
        Ty::Vec(p, n) => format!("{}{n}", prim(p)),
        Ty::Mat(c, r) => format!("float{c}x{r}"),
        Ty::Struct(abi) => abi.name.into(),
        Ty::Handle(kind) => handle(kind).into(),
        Ty::Ptr { mutable, pointee } => match mutable {
            true => format!("device {}*", name(*pointee, true)),
            false => format!("device const {}*", name(*pointee, true)),
        },
        // An array only ever appears through `decl`, which puts the length after the name.
        Ty::Array(element, _) => name(*element, packed),
        other => panic!("{other:?} has no MSL spelling"),
    }
}

/// C declarator order: an array's length belongs to the name, not the type.
fn decl(ty: Ty, binding: &str, packed: bool) -> String {
    match ty {
        Ty::Array(element, count) => format!("{} {binding}[{count}]", name(*element, packed)),
        other => format!("{} {binding}", name(other, packed)),
    }
}

fn declare(abi: &TypeAbi, interface: bool) -> String {
    let mut out = format!("\nstruct {} {{\n", abi.name);
    let mut cursor = 0usize;
    let mut pad = 0usize;
    for (index, field) in abi.fields.iter().enumerate() {
        if interface {
            // A varying carries no layout: the compiler assigns it, and `position` is the one
            // member either stage names.
            // An integer varying cannot be interpolated, and saying so is not optional on
            // either target.
            let flat = match field.ty {
                Ty::Scalar(p) | Ty::Vec(p, _) if !matches!(p, Prim::F16 | Prim::F32) => " [[flat]]",
                _ => "",
            };
            let attribute = match field.name {
                "position" => " [[position]]".into(),
                _ => format!(" [[user(nga{index})]]{flat}"),
            };
            out += &format!("    {}{attribute};\n", decl(field.ty, field.name, false));
            continue;
        }
        if field.offset > cursor {
            out += &padding(field.offset - cursor, &mut pad);
        }
        out += &format!("    {};\n", decl(field.ty, field.name, true));
        cursor = field.offset + field.size;
    }
    if !interface && abi.size > cursor {
        out += &padding(abi.size - cursor, &mut pad);
    }
    out += "};\n";
    out
}

fn constant_name(abi: &TypeAbi, field: &str) -> String {
    format!("nga_{}_{field}", abi.name)
}

fn padding(bytes: usize, index: &mut usize) -> String {
    let name = format!("_nga_pad{index}");
    *index += 1;
    match bytes % 4 {
        0 => format!("    uint {name}[{}];\n", bytes / 4),
        _ => format!("    uchar {name}[{bytes}];\n"),
    }
}

// -- functions ---------------------------------------------------------------------------

fn builtin(b: Builtin) -> &'static str {
    match b {
        Builtin::ThreadId => "[[thread_position_in_grid]]",
        Builtin::ThreadIdInGroup => "[[thread_position_in_threadgroup]]",
        Builtin::GroupId => "[[threadgroup_position_in_grid]]",
        Builtin::ThreadIndexInGroup => "[[thread_index_in_threadgroup]]",
        Builtin::VertexId => "[[vertex_id]]",
        Builtin::InstanceId => "[[instance_id]]",
        Builtin::BaseInstance => "[[base_instance]]",
        Builtin::Position => "[[position]]",
        Builtin::FrontFacing => "[[front_facing]]",
    }
}

struct W<'a> {
    f: &'a GpuFn,
    types: &'a ir::Types,
    /// The structs emitted as stage interfaces, which carry no padding for a literal to fill.
    io: &'a [&'static TypeAbi],
    /// One name per local, disambiguated where a shadowing `let` reused one.
    names: Vec<String>,
    out: String,
}

fn function(f: &'static GpuFn, types: &ir::Types, io: &[&'static TypeAbi]) -> String {
    let mut names: Vec<String> = Vec::new();
    for (index, local) in f.locals.iter().enumerate() {
        let shadowed = f.locals[..index].iter().any(|other| other.name == local.name);
        names.push(match shadowed {
            true => format!("{}_{index}", local.name),
            false => local.name.into(),
        });
    }

    // Everything a parameter can be, in one pass. What arrives some other way — the root
    // pointer, a function constant — leaves an empty argument that is dropped at the end.
    let mut arguments = Vec::new();
    let mut root = String::new();
    for param in f.params {
        arguments.push(match param.binding {
            ir::Binding::Root => {
                root = format!(
                    "    {} = ({})nga_push_data->root;\n",
                    decl(param.ty, param.name, true),
                    name(param.ty, true)
                );
                String::new()
            }
            ir::Binding::Builtin(b) => {
                format!("{} {}", decl(param.ty, param.name, false), builtin(b))
            }
            ir::Binding::Constants => String::new(),
            // A stage-in struct carries no binding of its own; the fragment stage takes one.
            _ if f.stage == Stage::Fragment => {
                format!("{} [[stage_in]]", decl(param.ty, param.name, false))
            }
            _ => decl(param.ty, param.name, false),
        });
    }
    arguments.retain(|argument| !argument.is_empty());
    if f.stage != Stage::Helper {
        arguments.insert(0, "constant nga_push* nga_push_data [[buffer(0)]]".into());
    }

    let qualifier = match f.stage {
        Stage::Helper => "",
        Stage::Vertex => "vertex ",
        Stage::Fragment => "fragment ",
        Stage::Compute(_) => "kernel ",
    };
    let mut w = W {
        f,
        types,
        io,
        names,
        out: String::new(),
    };
    // Metal takes the threadgroup size at dispatch, so `Stage::Compute`'s sizes are the host's
    // business and appear nowhere in the source.
    w.out += &format!(
        "\n{qualifier}{} {}({}) {{\n{root}",
        name(f.ret, false),
        f.name,
        arguments.join(", ")
    );
    stmt(&mut w, f.body, 1);
    w.out += "}\n";
    w.out
}

fn indent(depth: usize) -> String {
    "    ".repeat(depth)
}

fn stmt(w: &mut W, node: u32, depth: usize) {
    if node == NONE {
        return;
    }
    let pad = indent(depth);
    match w.f.nodes[node as usize] {
        Node::Block(stmts) => {
            for &s in stmts {
                stmt(w, s, depth);
            }
        }
        Node::Let { local, value } => {
            let ty = w.types.locals[local as usize];
            let value = expr(w, value);
            w.out += &format!(
                "{pad}{} = {value};\n",
                decl(ty, &w.names[local as usize].clone(), false)
            );
        }
        Node::Store { place, value } => {
            let place = expr(w, place);
            let value = expr(w, value);
            w.out += &format!("{pad}{place} = {value};\n");
        }
        Node::If { cond, then, els } => {
            let cond = expr(w, cond);
            w.out += &format!("{pad}if ({cond}) {{\n");
            stmt(w, then, depth + 1);
            if els == NONE {
                w.out += &format!("{pad}}}\n");
            } else {
                w.out += &format!("{pad}}} else {{\n");
                stmt(w, els, depth + 1);
                w.out += &format!("{pad}}}\n");
            }
        }
        Node::While { cond, body } => {
            let cond = expr(w, cond);
            w.out += &format!("{pad}while ({cond}) {{\n");
            stmt(w, body, depth + 1);
            w.out += &format!("{pad}}}\n");
        }
        Node::Loop { body } => {
            w.out += &format!("{pad}while (true) {{\n");
            stmt(w, body, depth + 1);
            w.out += &format!("{pad}}}\n");
        }
        Node::Range {
            local,
            start,
            end,
            body,
        } => {
            let ty = w.types.locals[local as usize];
            let name = w.names[local as usize].clone();
            let start = expr(w, start);
            let end = expr(w, end);
            w.out += &format!(
                "{pad}for ({} = {start}; {name} < {end}; {name}++) {{\n",
                decl(ty, &name, false)
            );
            stmt(w, body, depth + 1);
            w.out += &format!("{pad}}}\n");
        }
        Node::Break => w.out += &format!("{pad}break;\n"),
        Node::Continue => w.out += &format!("{pad}continue;\n"),
        Node::Discard => w.out += &format!("{pad}discard_fragment();\n"),
        Node::Return(value) => {
            if value == NONE {
                w.out += &format!("{pad}return;\n");
            } else {
                let ret = w.f.ret;
                let value = coerced(w, value, ret);
                w.out += &format!("{pad}return {value};\n");
            }
        }
        _ => {
            let value = expr(w, node);
            w.out += &format!("{pad}{value};\n");
        }
    }
}

fn binary(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::And => "&",
        BinOp::Or => "|",
        BinOp::Xor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::AndAnd => "&&",
        BinOp::OrOr => "||",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
    }
}

/// A value where the type around it is fixed. Rust inferred the shader's integer literals for
/// itself and `infer` only approximates that, so wherever the recorded types disagree the
/// conversion is written out rather than left to a language that may not do it silently — MSL
/// warns, and SPIR-V rejects it outright.
fn coerced(w: &mut W, node: u32, want: Ty) -> String {
    let have = w.types.nodes[node as usize];
    let convertible = matches!(have, Ty::Scalar(_) | Ty::Vec(..)) && matches!(want, Ty::Scalar(_) | Ty::Vec(..));
    let value = expr(w, node);
    match have == want || !convertible {
        true => value,
        false => format!("{}({value})", name(want, false)),
    }
}

fn expr(w: &mut W, node: u32) -> String {
    match w.f.nodes[node as usize] {
        Node::Local(i) => w.names[i as usize].clone(),
        Node::Param(i) => w.f.params[i as usize].name.into(),
        Node::Const(Const::Bool(value)) => value.to_string(),
        Node::Const(Const::Int(value)) => match w.types.nodes[node as usize] {
            Ty::Scalar(Prim::U32) => format!("{value}u"),
            _ => value.to_string(),
        },
        // MSL is C++: an unsuffixed float literal is a `double`, which Metal will not promote
        // silently in every position. The suffix is not optional.
        Node::Const(Const::Float(value)) => format!("{value:?}f"),
        Node::Field { base, name } => {
            // A constants parameter has no value of its own; each field is its own constant.
            if let Node::Param(index) = w.f.nodes[base as usize]
                && w.f.params[index as usize].binding == ir::Binding::Constants
            {
                let Ty::Struct(abi) = w.f.params[index as usize].ty else {
                    unreachable!("a constants parameter is a struct")
                };
                return constant_name(abi, name);
            }
            let pointer = matches!(w.types.nodes[base as usize], Ty::Ptr { .. });
            let base = expr(w, base);
            match pointer {
                true => format!("{base}->{name}"),
                false => format!("{base}.{name}"),
            }
        }
        Node::Index { base, index } => {
            let base = expr(w, base);
            let index = expr(w, index);
            format!("{base}[{index}]")
        }
        Node::Unary(op, value) => {
            let value = expr(w, value);
            match op {
                UnOp::Neg => format!("(-{value})"),
                UnOp::Not => format!("(!{value})"),
            }
        }
        Node::Binary(op, lhs, rhs) => {
            // A shift's two sides are independent; everything else operates at one type, and a
            // comparison's is its operands' rather than the `bool` it produces.
            let want = match (op.compare() || op.logical(), matches!(op, BinOp::Shl | BinOp::Shr)) {
                (_, true) => Ty::Infer,
                (true, _) => w.types.nodes[lhs as usize],
                (false, _) => w.types.nodes[node as usize],
            };
            let lhs = coerced(w, lhs, want);
            let rhs = coerced(w, rhs, want);
            format!("({lhs} {} {rhs})", binary(op))
        }
        Node::Cast { value, to } => {
            let value = expr(w, value);
            format!("{}({value})", name(to, false))
        }
        Node::Construct { ty, args } => {
            let element = match ty {
                Ty::Vec(p, _) => Ty::Scalar(p),
                Ty::Array(element, _) => *element,
                other => other,
            };
            let args: Vec<String> = args.iter().map(|&a| coerced(w, a, element)).collect();
            match ty {
                Ty::Array(..) => format!("{{ {} }}", args.join(", ")),
                _ => format!("{}({})", name(ty, false), args.join(", ")),
            }
        }
        Node::StructLit { ty, fields } => {
            let Ty::Struct(abi) = ty else {
                unreachable!("a struct literal is a struct")
            };
            // An interface struct has no padding members, so its initialiser has no `{}` for
            // them either. A memory one is padded to the byte offsets Rust chose, and the
            // padding is positional like everything else.
            let interface = w.io.iter().any(|other| std::ptr::eq(*other, abi));
            let mut cursor = 0usize;
            let mut values = Vec::new();
            for field in abi.fields {
                if field.offset > cursor && !interface {
                    values.push("{}".into());
                }
                let Some(&(_, value)) = fields.iter().find(|(name, _)| *name == field.name) else {
                    panic!("`{}` has no value for `{}`", abi.name, field.name);
                };
                values.push(coerced(w, value, field.ty));
                cursor = field.offset + field.size;
            }
            if abi.size > cursor && !interface {
                values.push("{}".into());
            }
            format!("{} {{ {} }}", abi.name, values.join(", "))
        }
        Node::Call { func, args } => {
            let callee = w.f.calls[func as usize];
            let args: Vec<String> = args
                .iter()
                .enumerate()
                .map(|(i, &a)| coerced(w, a, callee.params[i].ty))
                .collect();
            format!("{}({})", callee.name, args.join(", "))
        }
        Node::Intrinsic { op, args } => intrinsic(w, op, args),
        other => panic!("{other:?} is not an MSL expression"),
    }
}

fn intrinsic(w: &mut W, op: Intrinsic, args: &[u32]) -> String {
    let a: Vec<String> = args.iter().map(|&arg| expr(w, arg)).collect();
    let call = |name: &str| format!("{name}({})", a.join(", "));
    match op {
        Intrinsic::Sqrt => call("sqrt"),
        Intrinsic::InverseSqrt => call("rsqrt"),
        Intrinsic::Abs => call("abs"),
        Intrinsic::Sign => call("sign"),
        Intrinsic::Floor => call("floor"),
        Intrinsic::Ceil => call("ceil"),
        Intrinsic::Round => call("rint"),
        Intrinsic::Fract => call("fract"),
        Intrinsic::Sin => call("sin"),
        Intrinsic::Cos => call("cos"),
        Intrinsic::Tan => call("tan"),
        Intrinsic::Asin => call("asin"),
        Intrinsic::Acos => call("acos"),
        Intrinsic::Atan => call("atan"),
        Intrinsic::Atan2 => call("atan2"),
        Intrinsic::Exp => call("exp"),
        Intrinsic::Exp2 => call("exp2"),
        Intrinsic::Log => call("log"),
        Intrinsic::Log2 => call("log2"),
        Intrinsic::Pow => call("pow"),
        Intrinsic::Min => call("min"),
        Intrinsic::Max => call("max"),
        Intrinsic::Clamp => call("clamp"),
        Intrinsic::Mix => call("mix"),
        Intrinsic::Dot => call("dot"),
        Intrinsic::Cross => call("cross"),
        Intrinsic::Normalize => call("normalize"),
        Intrinsic::Length => call("length"),
        Intrinsic::Distance => call("distance"),
        Intrinsic::Reflect => call("reflect"),
        Intrinsic::Any => call("any"),
        Intrinsic::All => call("all"),
        Intrinsic::Transpose => call("transpose"),
        // The receiver-first Rust spellings whose MSL forms take their arguments elsewhere.
        Intrinsic::Step => format!("step({}, {})", a[1], a[0]),
        Intrinsic::SmoothStep => format!("smoothstep({}, {}, {})", a[1], a[2], a[0]),
        Intrinsic::Select => format!("select({}, {}, {})", a[2], a[0], a[1]),
        Intrinsic::Mul => format!("({} * {})", a[0], a[1]),
        Intrinsic::Barrier => "threadgroup_barrier(mem_flags::mem_device)".into(),
        Intrinsic::Sample => format!("{}.t.sample({}.s, {})", a[0], a[1], a[2]),
        Intrinsic::SampleLevel => {
            format!("{}.t.sample({}.s, {}, level({}))", a[0], a[1], a[2], a[3])
        }
        Intrinsic::Load => format!("{}.t.read({})", a[0], a[1]),
        Intrinsic::Store => format!("{}.t.write({}, {})", a[0], a[2], a[1]),
        Intrinsic::Dimensions => {
            format!("uint2({}.t.get_width(), {}.t.get_height())", a[0], a[0])
        }
        Intrinsic::AtomicAdd => atomic(&a, "add"),
        Intrinsic::AtomicMin => atomic(&a, "min"),
        Intrinsic::AtomicMax => atomic(&a, "max"),
        Intrinsic::AtomicExchange => format!(
            "atomic_exchange_explicit((device atomic_uint*){}, {}, memory_order_relaxed)",
            a[0], a[1]
        ),
    }
}

fn atomic(a: &[String], op: &str) -> String {
    format!(
        "atomic_fetch_{op}_explicit((device atomic_uint*){}, {}, memory_order_relaxed)",
        a[0], a[1]
    )
}
