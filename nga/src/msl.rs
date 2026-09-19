//! The Metal backend: `GpuFn` in, MSL source out.
//!
//! A recursive walk of the arena, printing as it goes. The only structural work is padding:
//! struct members land on the byte offsets `#[gpu_mod]` recorded from Rust, so a `uint`
//! array fills every gap. Rust's layout is authoritative and MSL is made to match, and nothing
//! is inferred on the way: the offsets are stated outright, not arranged for.

use crate::TypeAbi;
use crate::ir::{
    self, BinOp, Builtin, Const, GpuFn, HandleKind, Intrinsic, NONE, Node, Prim, Stage, Ty, UnOp,
};

pub const PRELUDE: &str = include_str!("prelude.metal");

/// Names Rust allows and MSL does not. A shader function called `vertex` is perfectly good
/// Rust and a syntax error in Metal, so any identifier that lands on one of these is emitted
/// with a trailing underscore. `metal.rs` runs the same pass on the name a pipeline asks for,
/// which is why this is public: the two spellings have to agree.
pub fn escape(name: &str) -> String {
    const RESERVED: &[&str] = &[
        "vertex",
        "fragment",
        "kernel",
        "constant",
        "device",
        "threadgroup",
        "thread",
        "threadgroup_imageblock",
        "ray_data",
        "object_data",
        "half",
        "float",
        "int",
        "uint",
        "short",
        "ushort",
        "char",
        "uchar",
        "long",
        "ulong",
        "bool",
        "void",
        "sampler",
        "texture",
        "packed_float3",
        "auto",
        "class",
        "template",
        "typename",
        "namespace",
        "using",
        "public",
        "private",
        "protected",
        "operator",
        "new",
        "delete",
        "this",
        "inline",
        "extern",
        "static",
        "register",
        "volatile",
        "union",
        "signed",
        "unsigned",
        "sizeof",
        "typedef",
        "goto",
        "switch",
        "case",
        "default",
        "do",
        "try",
        "catch",
        "throw",
        "friend",
        "virtual",
        "explicit",
        "export",
        "asm",
    ];
    match RESERVED.contains(&name) {
        true => format!("{name}_"),
        false => name.into(),
    }
}

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

    // A struct reached through a pointer or a local is memory and gets padded; stage interfaces
    // carry attributes instead.
    let mut io: Vec<&'static TypeAbi> = Vec::new();
    let mut mesh_io: Vec<&'static TypeAbi> = Vec::new();
    let mut tile_io: Vec<&'static TypeAbi> = Vec::new();
    for f in &functions {
        if f.stage == Stage::Vertex
            && let Ty::Struct(abi) = f.ret
        {
            io.push(abi);
        }
        if f.stage == Stage::Fragment {
            for param in f.params {
                if let Ty::Struct(abi) = param.ty {
                    io.push(abi);
                }
            }
        }
        match f.stage {
            Stage::Mesh {
                vertex: Ty::Struct(abi),
                ..
            } => {
                io.push(abi);
                mesh_io.push(abi);
            }
            Stage::Tile {
                imageblock: Ty::Struct(abi),
            } => tile_io.push(abi),
            _ => {}
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
        for ty in params
            .chain([f.ret])
            .chain(t.locals.iter().copied())
            .chain(t.nodes.iter().copied())
        {
            collect(ty, &mut structs);
        }
        match f.stage {
            Stage::Mesh { vertex, .. } => collect(vertex, &mut structs),
            Stage::Tile { imageblock } => collect(imageblock, &mut structs),
            _ => {}
        }
        types.push(t);
    }

    let mut out = String::from(PRELUDE);
    if functions
        .iter()
        .any(|f| matches!(f.stage, Stage::Object | Stage::Mesh { .. }))
        || structs.iter().any(|abi| {
            abi.fields.iter().any(|field| {
                matches!(
                    field.ty,
                    Ty::Handle(HandleKind::Commands | HandleKind::Pipeline)
                        | Ty::Array(&Ty::Handle(HandleKind::Commands | HandleKind::Pipeline), _)
                )
            })
        })
    {
        out += "\n#include <metal_mesh>\n\
struct nga_commands { command_buffer cb; };\n\
struct nga_pipeline { render_pipeline_state p; };\n\
static_assert(sizeof(nga_commands) == 8, \"nga_commands is not one resource id\");\n\
static_assert(sizeof(nga_pipeline) == 8, \"nga_pipeline is not one resource id\");\n\
static inline void nga_command_set_pipeline(nga_commands commands, uint slot, nga_pipeline pipeline) {\n\
    render_command command(commands.cb, slot);\n\
    command.set_render_pipeline_state(pipeline.p);\n\
}\n\
template<typename T>\n\
static inline void nga_command_set_vertex_buffer(nga_commands commands, uint slot, device T* value, uint index) {\n\
    render_command command(commands.cb, slot);\n\
    command.set_vertex_buffer(value, index);\n\
}\n\
static inline void nga_command_draw(nga_commands commands, uint slot, bool lines, uint start, uint count, uint instances, uint base_instance) {\n\
    render_command command(commands.cb, slot);\n\
    command.draw_primitives(lines ? primitive_type::line : primitive_type::triangle, start, count, instances, base_instance);\n\
}\n";
    }
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
                    decl(field.ty, &constant_name(abi, field.name))
                );
            }
        }
    }
    for abi in &structs {
        if tile_io.iter().any(|other| std::ptr::eq(*other, *abi)) {
            out += &declare_tile(abi);
        } else if mesh_io.iter().any(|other| std::ptr::eq(*other, *abi)) {
            out += &declare_mesh(abi);
        } else {
            let interface = io.iter().any(|other| std::ptr::eq(*other, *abi));
            out += &declare(abi, interface);
        }
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
        HandleKind::Commands => "nga_commands",
        HandleKind::Pipeline => "nga_pipeline",
        HandleKind::Opaque => "nga_opaque",
    }
}

fn name(ty: Ty) -> String {
    match ty {
        Ty::Void => "void".into(),
        Ty::Scalar(p) => prim(p).into(),
        Ty::Vec(p, n) => format!("{}{n}", prim(p)),
        Ty::PackedVec(p, n) => format!("packed_{}{n}", prim(p)),
        Ty::Mat(c, r) => format!("float{c}x{r}"),
        Ty::Struct(abi) => abi.name.into(),
        Ty::Handle(kind) => handle(kind).into(),
        Ty::ThreadgroupAtomicU32 => "atomic_uint".into(),
        Ty::Ptr { mutable, pointee } => match mutable {
            true => format!("device {}*", name(*pointee)),
            false => format!("device const {}*", name(*pointee)),
        },
        // An array only ever appears through `decl`, which puts the length after the name.
        Ty::Array(element, _) => name(*element),
        other => panic!("{other:?} has no MSL spelling"),
    }
}

/// C declarator order: an array's length belongs to the name, not the type.
fn decl(ty: Ty, binding: &str) -> String {
    match ty {
        Ty::Array(element, count) => format!("{} {binding}[{count}]", name(*element)),
        other => format!("{} {binding}", name(other)),
    }
}

fn emitted_size(ty: Ty) -> usize {
    match ty {
        Ty::Array(element, count) => emitted_size(*element) * count as usize,
        Ty::Struct(abi) => abi.size,
        other => other.size(),
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
                "render_target_array_index" => " [[render_target_array_index]]".into(),
                _ => format!(" [[user(nga{index})]]{flat}"),
            };
            out += &format!("    {}{attribute};\n", decl(field.ty, field.name));
            continue;
        }
        if field.offset > cursor {
            out += &padding(field.offset - cursor, &mut pad);
        }
        out += &format!("    {};\n", decl(field.ty, field.name));
        let occupied = emitted_size(field.ty);
        if field.size > occupied {
            out += &padding(field.size - occupied, &mut pad);
        }
        cursor = field.offset + field.size;
    }
    if !interface && abi.size > cursor {
        out += &padding(abi.size - cursor, &mut pad);
    }
    out += "};\n";
    if !interface {
        out += &format!(
            "static_assert(sizeof({}) == {}, \"{}\");\n",
            abi.name, abi.size, abi.name
        );
    }
    out
}

fn declare_tile(abi: &TypeAbi) -> String {
    let mut out = format!("\nstruct {} {{\n", abi.name);
    for (index, field) in abi.fields.iter().enumerate() {
        out += &format!("    {} [[color({index})]];\n", decl(field.ty, field.name));
    }
    out += "};\n";
    out
}

fn declare_mesh(abi: &TypeAbi) -> String {
    let mut out = format!("\nstruct {} {{\n", abi.name);
    for field in abi.fields {
        let attribute = match field.name {
            "position" => " [[position]]",
            "render_target_array_index" => " [[render_target_array_index]]",
            _ => "",
        };
        out += &format!("    {}{attribute};\n", decl(field.ty, field.name));
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
        Builtin::RenderTargetArrayIndex => "[[render_target_array_index]]",
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
        let shadowed = f.locals[..index]
            .iter()
            .any(|other| other.name == local.name);
        names.push(match shadowed {
            true => format!("{}_{index}", local.name),
            false => escape(local.name),
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
                    decl(param.ty, &escape(param.name)),
                    name(param.ty)
                );
                String::new()
            }
            ir::Binding::Builtin(b) => {
                format!("{} {}", decl(param.ty, &escape(param.name)), builtin(b))
            }
            ir::Binding::Constants => String::new(),
            ir::Binding::Payload => match f.stage {
                Stage::Object => format!(
                    "object_data {}& {} [[payload]]",
                    name(param.ty),
                    escape(param.name)
                ),
                Stage::Mesh { .. } => format!(
                    "const object_data {}& {} [[payload]]",
                    name(param.ty),
                    escape(param.name)
                ),
                _ => panic!("payload binding outside object or mesh stage"),
            },
            ir::Binding::Buffer(index) => {
                format!(
                    "{} [[buffer({index})]]",
                    decl(param.ty, &escape(param.name))
                )
            }
            // A stage-in struct carries no binding of its own; the fragment stage takes one.
            _ if f.stage == Stage::Fragment => {
                format!("{} [[stage_in]]", decl(param.ty, &escape(param.name)))
            }
            _ => decl(param.ty, &escape(param.name)),
        });
    }
    arguments.retain(|argument| !argument.is_empty());
    if f.stage != Stage::Helper {
        arguments.insert(0, "constant nga_push* nga_push_data [[buffer(0)]]".into());
    }
    match f.stage {
        Stage::Object => arguments.insert(0, "mesh_grid_properties nga_mesh_grid".into()),
        Stage::Mesh {
            vertex,
            max_vertices,
            max_primitives,
        } => {
            arguments.insert(0, format!("NgaMeshOut_{} nga_mesh_out", escape(f.name)));
            let _ = (vertex, max_vertices, max_primitives);
        }
        Stage::Tile { imageblock } => arguments.insert(
            0,
            format!(
                "imageblock<{}, imageblock_layout_implicit> nga_imageblock",
                name(imageblock)
            ),
        ),
        _ => {}
    }

    let qualifier = match f.stage {
        Stage::Helper => "",
        Stage::Vertex => "vertex ",
        Stage::Fragment => "fragment ",
        Stage::Compute(_) | Stage::Tile { .. } => "kernel ",
        Stage::Object => "[[object]] ",
        Stage::Mesh { .. } => "[[mesh]] ",
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
    if let Stage::Mesh {
        vertex,
        max_vertices,
        max_primitives,
    } = f.stage
    {
        w.out += &format!(
            "\nusing NgaMeshOut_{} = metal::mesh<{}, void, {}, {}, metal::topology::triangle>;\n",
            escape(f.name),
            name(vertex),
            max_vertices,
            max_primitives
        );
    }
    let ret = match f.stage {
        Stage::Object | Stage::Mesh { .. } | Stage::Tile { .. } => Ty::Void,
        _ => f.ret,
    };
    w.out += &format!(
        "\n{qualifier}{} {}({}) {{\n{root}",
        name(ret),
        escape(f.name),
        arguments.join(", ")
    );
    stmt(&mut w, f.body, 1);
    w.out += "}\n";
    w.out
}

fn stmt(w: &mut W, node: u32, depth: usize) {
    if node == NONE {
        return;
    }
    let pad = "    ".repeat(depth);
    match w.f.nodes[node as usize] {
        Node::Block(stmts) => {
            for &s in stmts {
                stmt(w, s, depth);
            }
        }
        Node::Let { local, value } => {
            let ty = w.types.locals[local as usize];
            if ty == Ty::ThreadgroupAtomicU32 {
                w.out += &format!(
                    "{pad}threadgroup atomic_uint {};\n",
                    w.names[local as usize]
                );
                return;
            }
            let value = coerced(w, value, ty);
            w.out += &format!(
                "{pad}{} = {value};\n",
                decl(ty, &w.names[local as usize].clone())
            );
        }
        Node::Store { place, value } => {
            let ty = w.types.nodes[place as usize];
            let place = expr(w, place);
            let value = coerced(w, value, ty);
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
                decl(ty, &name)
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
    let convertible =
        matches!(have, Ty::Scalar(_) | Ty::Vec(..)) && matches!(want, Ty::Scalar(_) | Ty::Vec(..));
    let value = expr(w, node);
    match have == want || !convertible {
        true => value,
        false => format!("{}({value})", name(want)),
    }
}

fn expr(w: &mut W, node: u32) -> String {
    match w.f.nodes[node as usize] {
        Node::Local(i) => w.names[i as usize].clone(),
        Node::Param(i) => escape(w.f.params[i as usize].name),
        Node::Const(Const::Bool(value)) => value.to_string(),
        Node::Const(Const::Int(value)) => match w.types.nodes[node as usize] {
            Ty::Scalar(Prim::U32) => format!("{value}u"),
            Ty::Scalar(Prim::U64) => format!("{value}ul"),
            Ty::Scalar(Prim::I64) => format!("{value}l"),
            Ty::Scalar(Prim::U16) => format!("ushort({value})"),
            Ty::Scalar(Prim::I16) => format!("short({value})"),
            Ty::Scalar(Prim::U8) => format!("uchar({value})"),
            Ty::Scalar(Prim::I8) => format!("char({value})"),
            Ty::Scalar(Prim::F32) => format!("{:?}f", value as f64),
            Ty::Scalar(Prim::F16) => format!("{:?}h", value as f64),
            _ => value.to_string(),
        },
        // MSL is C++: an unsuffixed float literal is a `double`, which Metal will not promote
        // silently in every position. The suffix is not optional.
        Node::Const(Const::Float(value)) => match w.types.nodes[node as usize] {
            Ty::Scalar(Prim::U32) => format!("{}u", value as u32),
            Ty::Scalar(Prim::U64) => format!("{}ul", value as u64),
            Ty::Scalar(Prim::I32) => format!("{}", value as i32),
            Ty::Scalar(Prim::I64) => format!("{}l", value as i64),
            Ty::Scalar(Prim::U16) => format!("ushort({})", value as u16),
            Ty::Scalar(Prim::I16) => format!("short({})", value as i16),
            Ty::Scalar(Prim::U8) => format!("uchar({})", value as u8),
            Ty::Scalar(Prim::I8) => format!("char({})", value as i8),
            Ty::Scalar(Prim::F16) => format!("{value:?}h"),
            _ => format!("{value:?}f"),
        },
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
            let have = w.types.nodes[value as usize];
            let value = coerced(w, value, have);
            match op {
                UnOp::Neg => format!("(-{value})"),
                UnOp::Not => format!("(!{value})"),
            }
        }
        Node::Binary(op, lhs, rhs) => {
            // A shift's two sides are independent; everything else operates at one type, and a
            // comparison's is its operands' rather than the `bool` it produces.
            let want = match (
                op.compare() || op.logical(),
                matches!(op, BinOp::Shl | BinOp::Shr),
            ) {
                (_, true) => Ty::Infer,
                (true, _) => w.types.nodes[lhs as usize],
                (false, _) => w.types.nodes[node as usize],
            };
            let lhs = coerced(w, lhs, want);
            let rhs = coerced(w, rhs, want);
            format!("({lhs} {} {rhs})", binary(op))
        }
        Node::Cast { value, to } => {
            let text = expr(w, value);
            format!("{}({text})", name(to))
        }
        Node::Construct { ty, args } => {
            let element = match ty {
                Ty::Vec(p, _) | Ty::PackedVec(p, _) => Ty::Scalar(p),
                Ty::Array(element, _) => *element,
                other => other,
            };
            let args: Vec<String> = args.iter().map(|&a| coerced(w, a, element)).collect();
            match ty {
                Ty::Array(..) => format!("{{ {} }}", args.join(", ")),
                _ => format!("{}({})", name(ty), args.join(", ")),
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
            format!("{}({})", escape(callee.name), args.join(", "))
        }
        Node::Intrinsic { op, args } => intrinsic(w, op, args),
        other => panic!("{other:?} is not an MSL expression"),
    }
}

fn tex_expr(w: &W, arg: u32, expr: &str) -> String {
    match w.types.nodes[arg as usize] {
        Ty::Handle(HandleKind::Texture2d) => format!("nga_deref_tex2d(nga_push_data, {expr})"),
        Ty::Handle(HandleKind::Texture2dRW) => {
            format!("nga_deref_tex2d_rw(nga_push_data, {expr})")
        }
        Ty::Handle(HandleKind::Texture2dArray) => {
            format!("nga_deref_tex2d_array(nga_push_data, {expr})")
        }
        Ty::Handle(HandleKind::Texture3d) => format!("nga_deref_tex3d(nga_push_data, {expr})"),
        Ty::Handle(HandleKind::Texture3dRW) => {
            format!("nga_deref_tex3d_rw(nga_push_data, {expr})")
        }
        Ty::Handle(HandleKind::TextureCube) => {
            format!("nga_deref_texcube(nga_push_data, {expr})")
        }
        Ty::Handle(HandleKind::Depth2d) => format!("nga_deref_depth2d(nga_push_data, {expr})"),
        _ => format!("{expr}.t"),
    }
}

fn intrinsic(w: &mut W, op: Intrinsic, args: &[u32]) -> String {
    let a: Vec<String> = args
        .iter()
        .map(|&arg| coerced(w, arg, w.types.nodes[arg as usize]))
        .collect();
    let call = |name: &str| format!("{name}({})", a.join(", "));
    match op {
        Intrinsic::Unpack => {
            let p = w.types.nodes[args[0] as usize].prim();
            format!("{}3({}.x, {}.y, {}.z)", prim(p), a[0], a[0], a[0])
        }
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
        Intrinsic::ThreadgroupBarrier => "threadgroup_barrier(mem_flags::mem_threadgroup)".into(),
        Intrinsic::Offset => format!("nga_offset({}, {})", a[0], a[1]),
        Intrinsic::Sample => {
            let tex = tex_expr(w, args[0], &a[0]);
            format!("{tex}.sample({}.s, {})", a[1], a[2])
        }
        Intrinsic::SampleLevel => {
            let tex = tex_expr(w, args[0], &a[0]);
            format!("{tex}.sample({}.s, {}, level({}))", a[1], a[2], a[3])
        }
        Intrinsic::Load => {
            let tex = tex_expr(w, args[0], &a[0]);
            format!("{tex}.read({})", a[1])
        }
        Intrinsic::Store => {
            let tex = tex_expr(w, args[0], &a[0]);
            format!("{tex}.write({}, {})", a[2], a[1])
        }
        Intrinsic::Dimensions => {
            let tex = tex_expr(w, args[0], &a[0]);
            format!("uint2({tex}.get_width(), {tex}.get_height())")
        }
        Intrinsic::Dfdx => call("dfdx"),
        Intrinsic::Dfdy => call("dfdy"),
        Intrinsic::MeshSetThreadgroups => {
            format!("nga_mesh_grid.set_threadgroups_per_grid({})", a[0])
        }
        Intrinsic::MeshSetPrimitiveCount => {
            format!("nga_mesh_out.set_primitive_count({})", a[0])
        }
        Intrinsic::MeshSetVertex => {
            format!("nga_mesh_out.set_vertex({}, {})", a[0], a[1])
        }
        Intrinsic::MeshSetIndex => {
            format!("nga_mesh_out.set_index({}, uchar({}))", a[0], a[1])
        }
        Intrinsic::TileRead => format!("nga_imageblock.read(ushort2({}))", a[0]),
        Intrinsic::TileWrite => {
            format!("nga_imageblock.write({}, ushort2({}))", a[0], a[1])
        }
        Intrinsic::CommandSetPipeline => {
            format!("nga_command_set_pipeline({}, {}, {})", a[0], a[1], a[2])
        }
        Intrinsic::CommandSetVertexBuffer => format!(
            "nga_command_set_vertex_buffer({}, {}, &({}), {})",
            a[0], a[1], a[2], a[3]
        ),
        Intrinsic::CommandDraw => format!(
            "nga_command_draw({}, {}, {}, {}, {}, {}, {})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6]
        ),
        Intrinsic::AtomicAdd => atomic(w, args, &a, "add"),
        Intrinsic::AtomicMin => atomic(w, args, &a, "min"),
        Intrinsic::AtomicMax => atomic(w, args, &a, "max"),
        Intrinsic::AtomicExchange => format!(
            "atomic_exchange_explicit({}, {}, memory_order_relaxed)",
            atomic_target(w.types.nodes[args[0] as usize], &a[0]),
            a[1]
        ),
    }
}

fn atomic(w: &W, args: &[u32], a: &[String], op: &str) -> String {
    format!(
        "atomic_fetch_{op}_explicit({}, {}, memory_order_relaxed)",
        atomic_target(w.types.nodes[args[0] as usize], &a[0]),
        a[1]
    )
}

fn atomic_target(ty: Ty, value: &str) -> String {
    match ty {
        Ty::ThreadgroupAtomicU32 => format!("&({value})"),
        _ => format!("(device atomic_uint*){value}"),
    }
}

/// Metal's compiler log for a failed `newLibrary` is mostly rejected `operator*` candidates
/// from the standard headers. Keep the shader's own errors and the line they name.
pub fn format_compile_error(source: &str, compiler: &str) -> String {
    let src_lines: Vec<&str> = source.lines().collect();
    let mut out = String::from("Metal shader compile failed:\n");
    let mut kept = 0usize;
    let mut i = 0;
    let log: Vec<&str> = compiler.lines().collect();
    while i < log.len() {
        let line = log[i];
        let Some(diag) = diagnostic(line) else {
            i += 1;
            continue;
        };
        let start = i;
        i += 1;
        while i < log.len() && diagnostic(log[i]).is_none() {
            i += 1;
        }
        if !keep_diagnostic(&diag, line) {
            continue;
        }
        kept += 1;
        out.push_str(line);
        out.push('\n');
        for extra in &log[start + 1..i] {
            if extra.trim().is_empty() {
                continue;
            }
            out.push_str(extra);
            out.push('\n');
        }
        if diag.file == "program_source"
            && let Some(src) = src_lines.get(diag.line.saturating_sub(1))
        {
            out.push_str(&format!("    {}\n", src.trim_end()));
        }
    }
    if kept == 0 {
        let stripped: String = compiler
            .lines()
            .filter(|line| !line.contains("candidate template ignored"))
            .collect::<Vec<_>>()
            .join("\n");
        return format!("Metal shader compile failed:\n{stripped}");
    }
    out
}

struct Diagnostic {
    file: String,
    line: usize,
    severity: String,
}

fn diagnostic(line: &str) -> Option<Diagnostic> {
    for severity in ["error", "warning", "note"] {
        let marker = format!(": {severity}: ");
        let Some(pos) = line.find(&marker) else {
            continue;
        };
        let prefix = &line[..pos];
        let mut parts = prefix.rsplitn(3, ':');
        let _col = parts.next()?;
        let line_no = parts.next()?.parse::<usize>().ok()?;
        let file = parts.next()?.to_string();
        return Some(Diagnostic {
            file,
            line: line_no,
            severity: severity.into(),
        });
    }
    None
}

fn keep_diagnostic(diag: &Diagnostic, line: &str) -> bool {
    if line.contains("candidate template ignored") {
        return false;
    }
    if diag.severity == "note" {
        return diag.file == "program_source";
    }
    diag.file == "program_source" || diag.severity == "error"
}
