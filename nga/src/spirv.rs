//! The Vulkan backend: `GpuFn` in, a SPIR-V module out. Binary, not assembly — the format is a
//! flat stream of words and writing it directly is less code than printing text for someone
//! else to assemble.
//!
//! Two decisions carry most of the file. Locals are `OpVariable`s in the `Function` storage
//! class with `OpLoad`/`OpStore` around them, so nothing here builds SSA or places phi nodes;
//! the driver's own mem2reg does that, and it is what glslang and DXC emit too. And control
//! flow is structured because Rust's `if`/`while`/`loop` already are — the merge blocks fall
//! out of the AST instead of being reconstructed from a graph.
//!
//! Layout needs no arranging either: `#[gpu_mod]` recorded the byte offset of every field
//! from Rust, and those numbers become `OpMemberDecorate Offset` unchanged. That is what
//! `-fvk-use-scalar-layout` used to buy from slangc.

use crate::ir::{self, BinOp, Builtin, Const, GpuFn, HandleKind, Intrinsic, NONE, Node, Prim, Stage, Ty, UnOp};
use crate::TypeAbi;

const MAGIC: u32 = 0x0723_0203;
const VERSION: u32 = 0x0001_0500;
/// Not a registered generator; the low half is a build number, which nothing reads.
const GENERATOR: u32 = 0x00fa_0001;

// Opcodes, in the order they are first used below.
const OP_EXT_INST_IMPORT: u32 = 11;
const OP_EXT_INST: u32 = 12;
const OP_MEMORY_MODEL: u32 = 14;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_CAPABILITY: u32 = 17;
const OP_TYPE_VOID: u32 = 19;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLER: u32 = 26;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_TYPE_FUNCTION: u32 = 33;
const OP_SPEC_CONSTANT: u32 = 50;
const OP_CONSTANT_TRUE: u32 = 41;
const OP_CONSTANT_FALSE: u32 = 42;
const OP_CONSTANT: u32 = 43;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_PARAMETER: u32 = 55;
const OP_FUNCTION_END: u32 = 56;
const OP_FUNCTION_CALL: u32 = 57;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_PTR_ACCESS_CHAIN: u32 = 67;
const OP_NAME: u32 = 5;
const OP_MEMBER_NAME: u32 = 6;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_VECTOR_SHUFFLE: u32 = 79;
const OP_COMPOSITE_CONSTRUCT: u32 = 80;
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_TRANSPOSE: u32 = 84;
const OP_SAMPLED_IMAGE: u32 = 86;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u32 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u32 = 88;
const OP_IMAGE_READ: u32 = 98;
const OP_IMAGE_WRITE: u32 = 99;
const OP_IMAGE_QUERY_SIZE: u32 = 104;
const OP_CONVERT_F_TO_U: u32 = 109;
const OP_CONVERT_F_TO_S: u32 = 110;
const OP_CONVERT_S_TO_F: u32 = 111;
const OP_CONVERT_U_TO_F: u32 = 112;
const OP_U_CONVERT: u32 = 113;
const OP_S_CONVERT: u32 = 114;
const OP_F_CONVERT: u32 = 115;
const OP_CONVERT_U_TO_PTR: u32 = 120;
const OP_S_NEGATE: u32 = 126;
const OP_F_NEGATE: u32 = 127;
const OP_I_ADD: u32 = 128;
const OP_F_ADD: u32 = 129;
const OP_I_SUB: u32 = 130;
const OP_F_SUB: u32 = 131;
const OP_I_MUL: u32 = 132;
const OP_F_MUL: u32 = 133;
const OP_U_DIV: u32 = 134;
const OP_S_DIV: u32 = 135;
const OP_F_DIV: u32 = 136;
const OP_U_MOD: u32 = 137;
const OP_S_REM: u32 = 138;
const OP_F_REM: u32 = 140;
const OP_MATRIX_TIMES_VECTOR: u32 = 145;
const OP_MATRIX_TIMES_MATRIX: u32 = 146;
const OP_DOT: u32 = 148;
const OP_LOGICAL_OR: u32 = 166;
const OP_LOGICAL_AND: u32 = 167;
const OP_LOGICAL_NOT: u32 = 168;
const OP_SELECT: u32 = 169;
const OP_I_EQUAL: u32 = 170;
const OP_I_NOT_EQUAL: u32 = 171;
const OP_U_GREATER_THAN: u32 = 172;
const OP_S_GREATER_THAN: u32 = 173;
const OP_U_GREATER_THAN_EQUAL: u32 = 174;
const OP_S_GREATER_THAN_EQUAL: u32 = 175;
const OP_U_LESS_THAN: u32 = 176;
const OP_S_LESS_THAN: u32 = 177;
const OP_U_LESS_THAN_EQUAL: u32 = 178;
const OP_S_LESS_THAN_EQUAL: u32 = 179;
const OP_F_ORD_EQUAL: u32 = 180;
const OP_F_ORD_NOT_EQUAL: u32 = 182;
const OP_F_ORD_LESS_THAN: u32 = 184;
const OP_F_ORD_GREATER_THAN: u32 = 186;
const OP_F_ORD_LESS_THAN_EQUAL: u32 = 188;
const OP_F_ORD_GREATER_THAN_EQUAL: u32 = 190;
const OP_SHIFT_RIGHT_LOGICAL: u32 = 194;
const OP_SHIFT_RIGHT_ARITHMETIC: u32 = 195;
const OP_SHIFT_LEFT_LOGICAL: u32 = 196;
const OP_BITWISE_OR: u32 = 197;
const OP_BITWISE_XOR: u32 = 198;
const OP_BITWISE_AND: u32 = 199;
const OP_NOT: u32 = 200;
const OP_ANY: u32 = 154;
const OP_ALL: u32 = 155;
const OP_CONTROL_BARRIER: u32 = 224;
const OP_ATOMIC_EXCHANGE: u32 = 229;
const OP_ATOMIC_I_ADD: u32 = 234;
const OP_ATOMIC_U_MIN: u32 = 237;
const OP_ATOMIC_U_MAX: u32 = 239;
const OP_LOOP_MERGE: u32 = 246;
const OP_SELECTION_MERGE: u32 = 247;
const OP_LABEL: u32 = 248;
const OP_BRANCH: u32 = 249;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const OP_KILL: u32 = 252;
const OP_RETURN: u32 = 253;
const OP_RETURN_VALUE: u32 = 254;

// GLSL.std.450 instruction numbers.
const GLSL_ROUND: u32 = 1;
const GLSL_F_ABS: u32 = 4;
const GLSL_S_ABS: u32 = 5;
const GLSL_F_SIGN: u32 = 6;
const GLSL_FLOOR: u32 = 8;
const GLSL_CEIL: u32 = 9;
const GLSL_FRACT: u32 = 10;
const GLSL_SIN: u32 = 13;
const GLSL_COS: u32 = 14;
const GLSL_TAN: u32 = 15;
const GLSL_ASIN: u32 = 16;
const GLSL_ACOS: u32 = 17;
const GLSL_ATAN: u32 = 18;
const GLSL_ATAN2: u32 = 25;
const GLSL_POW: u32 = 26;
const GLSL_EXP: u32 = 27;
const GLSL_LOG: u32 = 28;
const GLSL_EXP2: u32 = 29;
const GLSL_LOG2: u32 = 30;
const GLSL_SQRT: u32 = 31;
const GLSL_INVERSE_SQRT: u32 = 32;
const GLSL_F_MIN: u32 = 37;
const GLSL_U_MIN: u32 = 38;
const GLSL_S_MIN: u32 = 39;
const GLSL_F_MAX: u32 = 40;
const GLSL_U_MAX: u32 = 41;
const GLSL_S_MAX: u32 = 42;
const GLSL_F_CLAMP: u32 = 43;
const GLSL_U_CLAMP: u32 = 44;
const GLSL_S_CLAMP: u32 = 45;
const GLSL_F_MIX: u32 = 46;
const GLSL_STEP: u32 = 48;
const GLSL_SMOOTH_STEP: u32 = 49;
const GLSL_LENGTH: u32 = 66;
const GLSL_DISTANCE: u32 = 67;
const GLSL_CROSS: u32 = 68;
const GLSL_NORMALIZE: u32 = 69;
const GLSL_REFLECT: u32 = 71;

// Storage classes.
const UNIFORM_CONSTANT: u32 = 0;
const INPUT: u32 = 1;
const OUTPUT: u32 = 3;
const FUNCTION: u32 = 7;
const PUSH_CONSTANT: u32 = 9;
const PHYSICAL_STORAGE_BUFFER: u32 = 5349;

// Decorations.
const DEC_SPEC_ID: u32 = 1;
const DEC_BLOCK: u32 = 2;
const DEC_COL_MAJOR: u32 = 5;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_MATRIX_STRIDE: u32 = 7;
const DEC_BUILTIN: u32 = 11;
const DEC_FLAT: u32 = 14;
const DEC_LOCATION: u32 = 30;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;

const MEMORY_ALIGNED: u32 = 0x2;

fn inst(out: &mut Vec<u32>, op: u32, operands: &[u32]) {
    out.push(((operands.len() as u32 + 1) << 16) | op);
    out.extend_from_slice(operands);
}

/// A literal string: null-terminated UTF-8, packed four bytes to a word, low byte first.
fn string(text: &str) -> Vec<u32> {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct M {
    next: u32,
    glsl: u32,
    capabilities: Vec<u32>,
    extensions: Vec<String>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    names: Vec<u32>,
    decorations: Vec<u32>,
    globals: Vec<u32>,
    functions: Vec<u32>,
    types: Vec<(Ty, bool, u32)>,
    pointers: Vec<(u32, u32, u32)>,
    scalar_constants: Vec<(u32, u64, u32)>,
    function_types: Vec<(Vec<u32>, u32)>,
    strided: Vec<u32>,
    builtins: Vec<(Builtin, u32, u32)>,
    /// The push constant block, the three descriptor arrays and their element types, each made
    /// on first use. Zero means not yet.
    push: u32,
    textures: u32,
    storage: u32,
    samplers: u32,
    texture_image: u32,
    storage_image: u32,
    sampler_type: u32,
    sampled_image: u32,
}

impl M {
    fn id(&mut self) -> u32 {
        self.next += 1;
        self.next - 1
    }

    fn capability(&mut self, capability: u32) {
        if !self.capabilities.contains(&capability) {
            self.capabilities.push(capability);
        }
    }

    fn extension(&mut self, name: &str) {
        if !self.extensions.iter().any(|e| e == name) {
            self.extensions.push(name.into());
        }
    }

    fn name(&mut self, id: u32, text: &str) {
        let mut operands = vec![id];
        operands.extend(string(text));
        inst(&mut self.names, OP_NAME, &operands);
    }

    fn decorate(&mut self, id: u32, decoration: u32, extra: &[u32]) {
        let mut operands = vec![id, decoration];
        operands.extend_from_slice(extra);
        inst(&mut self.decorations, OP_DECORATE, &operands);
    }

    // -- types --

    /// A type as a value: no `Offset`, no `ArrayStride`. SPIR-V forbids explicit layout on
    /// anything in the `Function` class, so a struct read out of device memory is a different
    /// type id from the struct that lives there.
    fn ty(&mut self, ty: Ty) -> u32 {
        self.type_of(ty, false)
    }

    fn type_of(&mut self, ty: Ty, laid: bool) -> u32 {
        if let Ty::PackedVec(p, n) = ty {
            return self.type_of(Ty::Vec(p, n), laid);
        }
        // Only aggregates carry layout, so everything else has one id either way.
        let laid = laid && matches!(ty, Ty::Struct(_) | Ty::Array(..));
        if let Some((.., id)) = self
            .types
            .iter()
            .find(|(other, other_laid, _)| *other == ty && *other_laid == laid)
        {
            return *id;
        }
        let id = match ty {
            Ty::Void => {
                let id = self.id();
                inst(&mut self.globals, OP_TYPE_VOID, &[id]);
                id
            }
            Ty::Scalar(Prim::Bool) => {
                let id = self.id();
                inst(&mut self.globals, OP_TYPE_BOOL, &[id]);
                id
            }
            Ty::Scalar(p) => {
                let id = self.id();
                let bits = 8 * ir::prim_size(p) as u32;
                match p {
                    Prim::F16 | Prim::F32 => inst(&mut self.globals, OP_TYPE_FLOAT, &[id, bits]),
                    Prim::I8 | Prim::I16 | Prim::I32 | Prim::I64 => {
                        inst(&mut self.globals, OP_TYPE_INT, &[id, bits, 1])
                    }
                    _ => inst(&mut self.globals, OP_TYPE_INT, &[id, bits, 0]),
                }
                match p {
                    Prim::I64 | Prim::U64 => self.capability(11),
                    Prim::F16 => self.capability(9),
                    Prim::I16 | Prim::U16 => self.capability(22),
                    Prim::I8 | Prim::U8 => self.capability(39),
                    _ => {}
                }
                id
            }
            Ty::Vec(p, n) => {
                let element = self.ty(Ty::Scalar(p));
                let id = self.id();
                inst(&mut self.globals, OP_TYPE_VECTOR, &[id, element, n as u32]);
                id
            }
            Ty::PackedVec(..) => unreachable!(),
            Ty::Mat(columns, rows) => {
                let column = self.ty(Ty::Vec(Prim::F32, rows));
                let id = self.id();
                inst(
                    &mut self.globals,
                    OP_TYPE_MATRIX,
                    &[id, column, columns as u32],
                );
                id
            }
            Ty::Array(element, count) => {
                let element_id = self.type_of(*element, laid);
                let length = self.constant(Ty::U32, count as u64);
                let id = self.id();
                inst(&mut self.globals, OP_TYPE_ARRAY, &[id, element_id, length]);
                if laid {
                    self.decorate(id, DEC_ARRAY_STRIDE, &[element.size() as u32]);
                }
                id
            }
            Ty::Struct(abi) => self.structure(abi, laid),
            // Whatever a device pointer points at is memory, so its pointee is always the
            // laid-out form.
            Ty::Ptr { pointee, .. } => {
                self.capability(5347);
                self.extension("SPV_KHR_physical_storage_buffer");
                self.ptr(PHYSICAL_STORAGE_BUFFER, *pointee)
            }
            // A handle is eight bytes in a struct and an index into a descriptor array
            // everywhere else. The image types themselves are the descriptor arrays' business.
            Ty::Handle(_) => self.ty(Ty::Scalar(Prim::U64)),
            other => panic!("{other:?} has no SPIR-V type"),
        };
        self.types.push((ty, laid, id));
        id
    }

    fn structure(&mut self, abi: &'static TypeAbi, laid: bool) -> u32 {
        let members: Vec<u32> = abi
            .fields
            .iter()
            .map(|f| self.type_of(f.ty, laid))
            .collect();
        let id = self.id();
        let mut operands = vec![id];
        operands.extend(&members);
        inst(&mut self.globals, OP_TYPE_STRUCT, &operands);
        self.name(id, abi.name);
        for (index, field) in abi.fields.iter().enumerate() {
            let mut operands = vec![id, index as u32];
            operands.extend(string(field.name));
            inst(&mut self.names, OP_MEMBER_NAME, &operands);
            if !laid {
                continue;
            }
            // The whole of the layout contract, in one instruction per field: Rust said where
            // this member is, and SPIR-V is told rather than asked.
            inst(
                &mut self.decorations,
                OP_MEMBER_DECORATE,
                &[id, index as u32, DEC_OFFSET, field.offset as u32],
            );
            // A matrix member needs its own two, and column-major is what the host writes.
            if let Ty::Mat(_, rows) = field.ty {
                inst(
                    &mut self.decorations,
                    OP_MEMBER_DECORATE,
                    &[id, index as u32, DEC_COL_MAJOR],
                );
                inst(
                    &mut self.decorations,
                    OP_MEMBER_DECORATE,
                    &[
                        id,
                        index as u32,
                        DEC_MATRIX_STRIDE,
                        if rows == 3 { 16 } else { 4 * rows as u32 },
                    ],
                );
            }
        }
        id
    }

    /// A pointer type, by pointee type rather than id, so a device one can be told how far
    /// apart its elements are. `OpPtrAccessChain` — the only pointer arithmetic SPIR-V has, and
    /// what `p[i]` on a device pointer becomes — is invalid without that stride.
    fn ptr(&mut self, storage: u32, ty: Ty) -> u32 {
        let memory = storage == PHYSICAL_STORAGE_BUFFER || storage == PUSH_CONSTANT;
        let pointee = self.type_of(ty, memory);
        let id = self.pointer(storage, pointee);
        if storage == PHYSICAL_STORAGE_BUFFER && !self.strided.contains(&id) {
            self.decorate(id, DEC_ARRAY_STRIDE, &[ty.size() as u32]);
            self.strided.push(id);
        }
        id
    }

    fn pointer(&mut self, storage: u32, pointee: u32) -> u32 {
        if let Some((.., id)) = self
            .pointers
            .iter()
            .find(|(s, p, _)| *s == storage && *p == pointee)
        {
            return *id;
        }
        let id = self.id();
        inst(&mut self.globals, OP_TYPE_POINTER, &[id, storage, pointee]);
        self.pointers.push((storage, pointee, id));
        id
    }

    // -- constants --

    fn constant(&mut self, ty: Ty, bits: u64) -> u32 {
        let type_id = self.ty(ty);
        if let Some((.., id)) = self
            .scalar_constants
            .iter()
            .find(|(t, b, _)| *t == type_id && *b == bits)
        {
            return *id;
        }
        let id = self.id();
        match ty {
            Ty::Scalar(Prim::Bool) => {
                let op = match bits {
                    0 => OP_CONSTANT_FALSE,
                    _ => OP_CONSTANT_TRUE,
                };
                inst(&mut self.globals, op, &[type_id, id]);
            }
            Ty::Scalar(Prim::I64 | Prim::U64) => inst(
                &mut self.globals,
                OP_CONSTANT,
                &[type_id, id, bits as u32, (bits >> 32) as u32],
            ),
            _ => inst(&mut self.globals, OP_CONSTANT, &[type_id, id, bits as u32]),
        }
        self.scalar_constants.push((type_id, bits, id));
        id
    }

    /// The bit pattern a scalar of this type takes for a literal.
    fn literal(&mut self, ty: Ty, value: Const) -> u32 {
        let bits = match (ty, value) {
            (_, Const::Bool(b)) => b as u64,
            (Ty::Scalar(Prim::F32), Const::Int(i)) => (i as f32).to_bits() as u64,
            (Ty::Scalar(Prim::F32), Const::Float(f)) => (f as f32).to_bits() as u64,
            (Ty::Scalar(Prim::F16), Const::Float(f)) => half(f as f32) as u64,
            (Ty::Scalar(Prim::F16), Const::Int(i)) => half(i as f32) as u64,
            (_, Const::Int(i)) => i as u64,
            (_, Const::Float(f)) => f as i64 as u64,
        };
        self.constant(ty, bits)
    }
}

/// IEEE half, for the one place `f16` reaches a constant. Ties round to even like the hardware.
fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    sign | ((exponent as u16) << 10) | (mantissa >> 13) as u16
}

// -- the module ---------------------------------------------------------------------------

pub fn module(entries: &[&'static GpuFn]) -> Vec<u8> {
    let words = words(entries);
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub fn words(entries: &[&'static GpuFn]) -> Vec<u32> {
    let mut m = M {
        next: 1,
        glsl: 0,
        capabilities: Vec::new(),
        extensions: Vec::new(),
        entry_points: Vec::new(),
        execution_modes: Vec::new(),
        names: Vec::new(),
        decorations: Vec::new(),
        globals: Vec::new(),
        functions: Vec::new(),
        types: Vec::new(),
        pointers: Vec::new(),
        scalar_constants: Vec::new(),
        function_types: Vec::new(),
        strided: Vec::new(),
        builtins: Vec::new(),
        push: 0,
        textures: 0,
        storage: 0,
        samplers: 0,
        texture_image: 0,
        storage_image: 0,
        sampler_type: 0,
        sampled_image: 0,
    };
    m.glsl = m.id();
    m.capability(1);
    // Addresses are the whole model here: every root pointer is a device address and every
    // struct a shader walks is reached through one.
    m.capability(5347);
    m.extension("SPV_KHR_physical_storage_buffer");

    let mut functions: Vec<&'static GpuFn> = Vec::new();
    for entry in entries {
        for f in entry.reachable() {
            if !functions.iter().any(|g| std::ptr::eq(*g, f)) {
                functions.push(f);
            }
        }
    }

    let mut ids: Vec<(&'static GpuFn, u32)> = Vec::new();
    for f in &functions {
        let id = m.id();
        m.name(id, f.name);
        ids.push((f, id));
    }
    for (f, id) in &ids {
        function(&mut m, f, *id, &ids);
    }

    let mut out = vec![MAGIC, VERSION, GENERATOR, m.next, 0];
    for capability in &m.capabilities.clone() {
        inst(&mut out, OP_CAPABILITY, &[*capability]);
    }
    for extension in &m.extensions {
        let mut operands = Vec::new();
        operands.extend(string(extension));
        inst(&mut out, 10, &operands);
    }
    let mut operands = vec![m.glsl];
    operands.extend(string("GLSL.std.450"));
    inst(&mut out, OP_EXT_INST_IMPORT, &operands);
    // Logical addressing for everything the module declares, physical for the pointers it is
    // handed. `PhysicalStorageBuffer64` is the only combination Vulkan allows here.
    inst(&mut out, OP_MEMORY_MODEL, &[5348, 1]);
    out.extend(m.entry_points);
    out.extend(m.execution_modes);
    out.extend(m.names);
    out.extend(m.decorations);
    out.extend(m.globals);
    out.extend(m.functions);
    out
}

// -- functions ----------------------------------------------------------------------------

struct W<'a> {
    f: &'a GpuFn,
    types: ir::Types,
    /// The `OpVariable` backing each local.
    locals: Vec<u32>,
    /// The value of each parameter: an `OpFunctionParameter`, or a load from a global.
    params: Vec<u32>,
    /// Where `break` and `continue` go, innermost last.
    loops: Vec<(u32, u32)>,
    /// The globals this entry point touched, for its interface list.
    interface: Vec<u32>,
    /// One id per specialization constant, by parameter and field index.
    constants: Vec<(u32, u32, u32)>,
    /// Set for an entry point, whose `return` writes to output variables first.
    outputs: Vec<(u32, u32)>,
    /// Whether the block being written already has its terminator. A basic block has exactly
    /// one, at the end, and `return` in the middle of a Rust body would otherwise produce two.
    terminated: bool,
    body: Vec<u32>,
    ids: &'a [(&'static GpuFn, u32)],
}

/// The storage class is part of the answer: the member a vertex stage writes as `Position` is
/// the one a fragment stage reads as `FragCoord`, and Vulkan rejects either in the other's
/// place.
fn builtin_number(b: Builtin, storage: u32) -> u32 {
    match b {
        Builtin::Position if storage == INPUT => 15,
        Builtin::Position => 0,
        // `VertexIndex` and `InstanceIndex`, not the GL-style `VertexId`/`InstanceId`, which
        // Vulkan does not allow. These are the ones that count from the draw's own base, which
        // is also what Metal's `[[vertex_id]]` gives.
        Builtin::VertexId => 42,
        Builtin::InstanceId => 43,
        Builtin::BaseInstance => 4425,
        Builtin::RenderTargetArrayIndex => 9,
        Builtin::FrontFacing => 17,
        Builtin::ThreadIdInGroup => 27,
        Builtin::ThreadId => 28,
        Builtin::GroupId => 26,
        Builtin::ThreadIndexInGroup => 29,
    }
}

/// The builtin's own type, which is not always the one the parameter asks for.
fn builtin_ty(b: Builtin) -> Ty {
    match b {
        Builtin::Position => Ty::Vec(Prim::F32, 4),
        Builtin::VertexId
        | Builtin::InstanceId
        | Builtin::BaseInstance
        | Builtin::RenderTargetArrayIndex
        | Builtin::ThreadIndexInGroup => Ty::U32,
        Builtin::FrontFacing => Ty::BOOL,
        _ => Ty::Vec(Prim::U32, 3),
    }
}

fn builtin_variable(m: &mut M, b: Builtin, storage: u32) -> u32 {
    // Keyed by storage class as well as number: `Position` written by a vertex stage and
    // `FragCoord` read by a fragment one are the same builtin and two different variables.
    if let Some((.., id)) = m
        .builtins
        .iter()
        .find(|(other, class, _)| *other == b && *class == storage)
    {
        return *id;
    }
    if b == Builtin::BaseInstance {
        m.capability(4427);
        m.extension("SPV_KHR_shader_draw_parameters");
    }
    if b == Builtin::RenderTargetArrayIndex {
        m.capability(69);
    }
    let ty = m.ty(builtin_ty(b));
    let pointer = m.pointer(storage, ty);
    let id = m.id();
    inst(&mut m.globals, OP_VARIABLE, &[pointer, id, storage]);
    m.decorate(id, DEC_BUILTIN, &[builtin_number(b, storage)]);
    m.builtins.push((b, storage, id));
    id
}

/// The push constant block: one address and the two descriptor-array bases, matching
/// `nga_push`. Made once and shared by every entry point.
fn push_block(m: &mut M) -> u32 {
    if m.push != 0 {
        return m.push;
    }
    let u64_id = m.ty(Ty::Scalar(Prim::U64));
    let u32_id = m.ty(Ty::U32);
    let id = m.id();
    inst(&mut m.globals, OP_TYPE_STRUCT, &[id, u64_id, u32_id, u32_id]);
    m.decorate(id, DEC_BLOCK, &[]);
    for (index, offset) in [(0u32, 0u32), (1, 8), (2, 12)] {
        inst(
            &mut m.decorations,
            OP_MEMBER_DECORATE,
            &[id, index, DEC_OFFSET, offset],
        );
    }
    m.name(id, "nga_push");
    let pointer = m.pointer(PUSH_CONSTANT, id);
    let variable = m.id();
    inst(&mut m.globals, OP_VARIABLE, &[pointer, variable, PUSH_CONSTANT]);
    m.push = variable;
    variable
}

fn function(m: &mut M, f: &'static GpuFn, id: u32, ids: &[(&'static GpuFn, u32)]) {
    let types = ir::infer(f);
    let entry = f.stage != Stage::Helper;
    let ret = match entry {
        // An entry point returns nothing; its result leaves through an output variable.
        true => Ty::Void,
        false => f.ret,
    };
    let ret_id = m.ty(ret);

    let parameter_types: Vec<u32> = match entry {
        true => Vec::new(),
        false => f.params.iter().map(|p| m.ty(p.ty)).collect(),
    };
    let function_type = {
        let mut signature = vec![ret_id];
        signature.extend(&parameter_types);
        match m.function_types.iter().find(|(other, _)| *other == signature) {
            Some((_, id)) => *id,
            None => {
                let id = m.id();
                let mut operands = vec![id];
                operands.extend(&signature);
                inst(&mut m.globals, OP_TYPE_FUNCTION, &operands);
                m.function_types.push((signature, id));
                id
            }
        }
    };

    let mut w = W {
        f,
        types,
        locals: Vec::new(),
        params: Vec::new(),
        loops: Vec::new(),
        interface: Vec::new(),
        constants: Vec::new(),
        outputs: Vec::new(),
        terminated: false,
        body: Vec::new(),
        ids,
    };

    inst(&mut w.body, OP_FUNCTION, &[ret_id, id, 0, function_type]);
    if !entry {
        for ty in &parameter_types {
            let param = m.id();
            inst(&mut w.body, OP_FUNCTION_PARAMETER, &[*ty, param]);
            w.params.push(param);
        }
    }
    let first = m.id();
    label(&mut w, first);

    // Every `OpVariable` in the `Function` class has to be in the first block, so the locals
    // are declared before anything else happens.
    for (index, _) in f.locals.iter().enumerate() {
        let ty = m.ty(w.types.locals[index]);
        let pointer = m.pointer(FUNCTION, ty);
        let variable = m.id();
        inst(&mut w.body, OP_VARIABLE, &[pointer, variable, FUNCTION]);
        m.name(variable, f.locals[index].name);
        w.locals.push(variable);
    }

    if entry {
        entry_prologue(m, &mut w);
    }

    stmt(m, &mut w, f.body);
    terminate(&mut w, entry);
    inst(&mut w.body, OP_FUNCTION_END, &[]);
    m.functions.extend(&w.body);

    if entry {
        let execution = match f.stage {
            Stage::Vertex => 0,
            Stage::Fragment => 4,
            _ => 5,
        };
        let mut operands = vec![execution, id];
        operands.extend(string(f.name));
        operands.extend(&w.interface);
        inst(&mut m.entry_points, OP_ENTRY_POINT, &operands);
        match f.stage {
            Stage::Fragment => inst(&mut m.execution_modes, OP_EXECUTION_MODE, &[id, 7]),
            Stage::Compute([x, y, z]) => {
                inst(&mut m.execution_modes, OP_EXECUTION_MODE, &[id, 17, x, y, z])
            }
            _ => {}
        }
    }
}

/// Loads an entry point's parameters out of the globals they arrive in, and declares the
/// outputs its `return` will write.
fn entry_prologue(m: &mut M, w: &mut W) {
    let mut location = 0u32;
    for (index, param) in w.f.params.iter().enumerate() {
        let value = match param.binding {
            ir::Binding::Constants => {
                let Ty::Struct(abi) = param.ty else {
                    panic!("`{}` is not a #[derive(Constants)] struct", param.name);
                };
                // Each field is one `OpSpecConstant`, decorated with the field's own index.
                // Nothing composes them: a reference to `options.gamma` is the constant itself,
                // so the driver still sees a specialization it can fold.
                for (field_index, field) in abi.fields.iter().enumerate() {
                    let type_id = m.ty(field.ty);
                    let id = m.id();
                    match field.ty {
                        Ty::Scalar(Prim::Bool) => inst(&mut m.globals, 48, &[type_id, id]),
                        Ty::Scalar(Prim::I64 | Prim::U64) => {
                            inst(&mut m.globals, OP_SPEC_CONSTANT, &[type_id, id, 0, 0])
                        }
                        _ => inst(&mut m.globals, OP_SPEC_CONSTANT, &[type_id, id, 0]),
                    }
                    m.decorate(id, DEC_SPEC_ID, &[field_index as u32]);
                    m.name(id, field.name);
                    w.constants.push((index as u32, field_index as u32, id));
                }
                0
            }
            ir::Binding::Root => {
                let block = push_block(m);
                if !w.interface.contains(&block) {
                    w.interface.push(block);
                }
                let u64_id = m.ty(Ty::Scalar(Prim::U64));
                let pointer = m.pointer(PUSH_CONSTANT, u64_id);
                let zero = m.constant(Ty::U32, 0);
                let chain = m.id();
                inst(&mut w.body, OP_ACCESS_CHAIN, &[pointer, chain, block, zero]);
                let address = m.id();
                inst(&mut w.body, OP_LOAD, &[u64_id, address, chain]);
                let ty = m.ty(param.ty);
                let value = m.id();
                inst(&mut w.body, OP_CONVERT_U_TO_PTR, &[ty, value, address]);
                value
            }
            ir::Binding::Builtin(b) => {
                let variable = builtin_variable(m, b, INPUT);
                if !w.interface.contains(&variable) {
                    w.interface.push(variable);
                }
                let ty = m.ty(builtin_ty(b));
                let loaded = m.id();
                inst(&mut w.body, OP_LOAD, &[ty, loaded, variable]);
                convert(m, w, loaded, builtin_ty(b), param.ty)
            }
            // A fragment stage takes its varyings as a struct, one input variable per member.
            _ => match param.ty {
                Ty::Struct(abi) => {
                    let mut members = Vec::new();
                    for field in abi.fields {
                        let variable = stage_variable(m, w, field.ty, field.name, &mut location, INPUT);
                        let ty = m.ty(field.ty);
                        let loaded = m.id();
                        inst(&mut w.body, OP_LOAD, &[ty, loaded, variable]);
                        members.push(loaded);
                    }
                    let ty = m.ty(param.ty);
                    let value = m.id();
                    let mut operands = vec![ty, value];
                    operands.extend(&members);
                    inst(&mut w.body, OP_COMPOSITE_CONSTRUCT, &operands);
                    value
                }
                other => {
                    let variable = stage_variable(m, w, other, param.name, &mut location, INPUT);
                    let ty = m.ty(other);
                    let loaded = m.id();
                    inst(&mut w.body, OP_LOAD, &[ty, loaded, variable]);
                    loaded
                }
            },
        };
        w.params.push(value);
    }

    // Outputs, in the order `Return` will write them.
    let mut location = 0u32;
    match w.f.ret {
        Ty::Void => {}
        Ty::Struct(abi) => {
            for field in abi.fields {
                let variable = stage_variable(m, w, field.ty, field.name, &mut location, OUTPUT);
                w.outputs.push((variable, m.ty(field.ty)));
            }
        }
        // A vertex stage that returns a bare `f32x4` is returning clip space and nothing else.
        other => {
            let name = match w.f.stage {
                Stage::Vertex => "position",
                _ => "result",
            };
            let variable = stage_variable(m, w, other, name, &mut location, OUTPUT);
            w.outputs.push((variable, m.ty(other)));
        }
    }
}

/// One varying. Interface members with reserved names become builtins; everything else takes
/// the next location.
fn stage_variable(m: &mut M, w: &mut W, ty: Ty, name: &str, location: &mut u32, storage: u32) -> u32 {
    let id = match name {
        "position" => builtin_variable(m, Builtin::Position, storage),
        "render_target_array_index" => {
            let id = builtin_variable(m, Builtin::RenderTargetArrayIndex, storage);
            if storage == INPUT {
                m.decorate(id, DEC_FLAT, &[]);
            }
            id
        }
        _ => {
            let type_id = m.ty(ty);
            let pointer = m.pointer(storage, type_id);
            let id = m.id();
            inst(&mut m.globals, OP_VARIABLE, &[pointer, id, storage]);
            m.decorate(id, DEC_LOCATION, &[*location]);
            // An integer varying has nothing to interpolate, and a fragment stage that reads
            // one without saying `Flat` is invalid rather than merely wrong.
            if storage == INPUT
                && matches!(ty, Ty::Scalar(p) | Ty::Vec(p, _) if !matches!(p, Prim::F16 | Prim::F32))
            {
                m.decorate(id, DEC_FLAT, &[]);
            }
            m.name(id, name);
            *location += 1;
            id
        }
    };
    if !w.interface.contains(&id) {
        w.interface.push(id);
    }
    id
}

/// The last block needs a terminator, and the source only writes one where it returns. A
/// function whose result never arrives is unreachable rather than void, which is a distinction
/// SPIR-V insists on.
fn terminate(w: &mut W, entry: bool) {
    if w.terminated {
        return;
    }
    match w.f.ret == Ty::Void || entry {
        true => inst(&mut w.body, OP_RETURN, &[]),
        false => inst(&mut w.body, 255, &[]),
    }
    w.terminated = true;
}

// -- statements ---------------------------------------------------------------------------

fn stmt(m: &mut M, w: &mut W, node: u32) {
    if node == NONE {
        return;
    }
    match w.f.nodes[node as usize] {
        Node::Block(stmts) => {
            for &s in stmts {
                if w.terminated {
                    return;
                }
                stmt(m, w, s);
            }
        }
        Node::Let { local, value } => {
            let want = w.types.locals[local as usize];
            let value = coerced(m, w, value, want);
            let variable = w.locals[local as usize];
            inst(&mut w.body, OP_STORE, &[variable, value]);
        }
        Node::Store { place, value } => {
            let want = w.types.nodes[place as usize];
            let value = coerced(m, w, value, want);
            let target = place_of(m, w, place);
            store(m, w, target, value);
        }
        Node::If { cond, then, els } => {
            let cond = value(m, w, cond);
            let then_label = m.id();
            let else_label = m.id();
            let merge = m.id();
            inst(&mut w.body, OP_SELECTION_MERGE, &[merge, 0]);
            inst(
                &mut w.body,
                OP_BRANCH_CONDITIONAL,
                &[cond, then_label, if els == NONE { merge } else { else_label }],
            );
            w.terminated = true;
            label(w, then_label);
            stmt(m, w, then);
            branch(w, merge);
            if els != NONE {
                label(w, else_label);
                stmt(m, w, els);
                branch(w, merge);
            }
            label(w, merge);
        }
        Node::While { cond, body } => {
            let header = m.id();
            let test = m.id();
            let inner = m.id();
            let continue_label = m.id();
            let merge = m.id();
            branch(w, header);
            label(w, header);
            inst(&mut w.body, OP_LOOP_MERGE, &[merge, continue_label, 0]);
            branch(w, test);
            label(w, test);
            let cond = value(m, w, cond);
            inst(&mut w.body, OP_BRANCH_CONDITIONAL, &[cond, inner, merge]);
            w.terminated = true;
            label(w, inner);
            w.loops.push((merge, continue_label));
            stmt(m, w, body);
            w.loops.pop();
            branch(w, continue_label);
            label(w, continue_label);
            branch(w, header);
            label(w, merge);
        }
        Node::Loop { body } => {
            let header = m.id();
            let inner = m.id();
            let continue_label = m.id();
            let merge = m.id();
            branch(w, header);
            label(w, header);
            inst(&mut w.body, OP_LOOP_MERGE, &[merge, continue_label, 0]);
            branch(w, inner);
            label(w, inner);
            w.loops.push((merge, continue_label));
            stmt(m, w, body);
            w.loops.pop();
            branch(w, continue_label);
            label(w, continue_label);
            branch(w, header);
            label(w, merge);
        }
        Node::Range {
            local,
            start,
            end,
            body,
        } => {
            let ty = w.types.locals[local as usize];
            let variable = w.locals[local as usize];
            let start = coerced(m, w, start, ty);
            inst(&mut w.body, OP_STORE, &[variable, start]);
            let header = m.id();
            let test = m.id();
            let inner = m.id();
            let continue_label = m.id();
            let merge = m.id();
            branch(w, header);
            label(w, header);
            inst(&mut w.body, OP_LOOP_MERGE, &[merge, continue_label, 0]);
            branch(w, test);
            label(w, test);
            let type_id = m.ty(ty);
            let current = m.id();
            inst(&mut w.body, OP_LOAD, &[type_id, current, variable]);
            let end = coerced(m, w, end, ty);
            let bool_id = m.ty(Ty::BOOL);
            let cond = m.id();
            let op = match signed(ty) {
                true => OP_S_LESS_THAN,
                false => OP_U_LESS_THAN,
            };
            inst(&mut w.body, op, &[bool_id, cond, current, end]);
            inst(&mut w.body, OP_BRANCH_CONDITIONAL, &[cond, inner, merge]);
            w.terminated = true;
            label(w, inner);
            w.loops.push((merge, continue_label));
            stmt(m, w, body);
            w.loops.pop();
            branch(w, continue_label);
            label(w, continue_label);
            let current = m.id();
            inst(&mut w.body, OP_LOAD, &[type_id, current, variable]);
            let one = m.constant(ty, 1);
            let next = m.id();
            inst(&mut w.body, OP_I_ADD, &[type_id, next, current, one]);
            inst(&mut w.body, OP_STORE, &[variable, next]);
            branch(w, header);
            label(w, merge);
        }
        Node::Break => {
            let (merge, _) = *w.loops.last().expect("`break` outside a loop");
            branch(w, merge);
            w.terminated = true;
        }
        Node::Continue => {
            let (_, continue_label) = *w.loops.last().expect("`continue` outside a loop");
            branch(w, continue_label);
            w.terminated = true;
        }
        Node::Discard => {
            inst(&mut w.body, OP_KILL, &[]);
            w.terminated = true;
        }
        Node::Return(result) => {
            if result == NONE {
                inst(&mut w.body, OP_RETURN, &[]);
            } else if w.outputs.is_empty() {
                let ret = w.f.ret;
                let result = coerced(m, w, result, ret);
                inst(&mut w.body, OP_RETURN_VALUE, &[result]);
            } else {
                // An entry point's result goes to its output variables and the return is bare.
                let ret = w.f.ret;
                let result = coerced(m, w, result, ret);
                for (index, (variable, ty)) in w.outputs.clone().iter().enumerate() {
                    let member = match w.f.ret {
                        Ty::Struct(_) => {
                            let id = m.id();
                            inst(
                                &mut w.body,
                                OP_COMPOSITE_EXTRACT,
                                &[*ty, id, result, index as u32],
                            );
                            id
                        }
                        _ => result,
                    };
                    inst(&mut w.body, OP_STORE, &[*variable, member]);
                }
                inst(&mut w.body, OP_RETURN, &[]);
            }
            w.terminated = true;
        }
        _ => {
            value(m, w, node);
        }
    }
}

/// A branch, unless the block is already finished — the `then` arm of an `if` that returns
/// needs no jump to the merge block.
fn branch(w: &mut W, target: u32) {
    if !w.terminated {
        inst(&mut w.body, OP_BRANCH, &[target]);
    }
    w.terminated = true;
}

fn label(w: &mut W, id: u32) {
    inst(&mut w.body, OP_LABEL, &[id]);
    w.terminated = false;
}

// -- places -------------------------------------------------------------------------------

/// A pointer to somewhere a value lives, and the storage class it lives in, which is what
/// decides whether a load needs the `Aligned` operand.
#[derive(Clone, Copy)]
struct Place {
    id: u32,
    ty: Ty,
    storage: u32,
}

/// Rebuilds an aggregate at the other layout. The two type ids differ only in decorations, so
/// the value has to be taken apart and put back together; there is no cast for it. Scalars and
/// vectors have one id either way and pass straight through.
fn relayout(m: &mut M, w: &mut W, id: u32, ty: Ty, laid: bool) -> u32 {
    let members: Vec<Ty> = match ty {
        Ty::Struct(abi) => abi.fields.iter().map(|f| f.ty).collect(),
        Ty::Array(element, count) => (0..count).map(|_| *element).collect(),
        _ => return id,
    };
    let mut built = Vec::new();
    for (index, member) in members.iter().enumerate() {
        let from = m.type_of(*member, !laid);
        let extracted = m.id();
        inst(
            &mut w.body,
            OP_COMPOSITE_EXTRACT,
            &[from, extracted, id, index as u32],
        );
        built.push(relayout(m, w, extracted, *member, laid));
    }
    let to = m.type_of(ty, laid);
    let out = m.id();
    let mut operands = vec![to, out];
    operands.extend(&built);
    inst(&mut w.body, OP_COMPOSITE_CONSTRUCT, &operands);
    out
}

fn memory(storage: u32) -> bool {
    storage == PHYSICAL_STORAGE_BUFFER || storage == PUSH_CONSTANT
}

fn store(m: &mut M, w: &mut W, place: Place, value: u32) {
    match place.storage {
        PHYSICAL_STORAGE_BUFFER => {
            let value = relayout(m, w, value, place.ty, true);
            inst(
                &mut w.body,
                OP_STORE,
                &[place.id, value, MEMORY_ALIGNED, place.ty.align() as u32],
            )
        }
        _ => inst(&mut w.body, OP_STORE, &[place.id, value]),
    }
}

fn load(m: &mut M, w: &mut W, place: Place) -> u32 {
    let ty = m.type_of(place.ty, memory(place.storage));
    let id = m.id();
    match place.storage {
        // Every access through a device address needs its alignment stated. Without it the
        // module is invalid, not merely slow.
        PHYSICAL_STORAGE_BUFFER => inst(
            &mut w.body,
            OP_LOAD,
            &[ty, id, place.id, MEMORY_ALIGNED, place.ty.align() as u32],
        ),
        _ => inst(&mut w.body, OP_LOAD, &[ty, id, place.id]),
    }
    match memory(place.storage) {
        true => relayout(m, w, id, place.ty, false),
        false => id,
    }
}

fn place_of(m: &mut M, w: &mut W, node: u32) -> Place {
    match w.f.nodes[node as usize] {
        Node::Local(index) => Place {
            id: w.locals[index as usize],
            ty: w.types.locals[index as usize],
            storage: FUNCTION,
        },
        Node::Field { base, name } => {
            let base_ty = w.types.nodes[base as usize];
            let ty = w.types.nodes[node as usize];
            let (pointer, storage) = match base_ty {
                // A device pointer is already the address of the struct.
                Ty::Ptr { .. } => (value(m, w, base), PHYSICAL_STORAGE_BUFFER),
                _ => {
                    let place = place_of(m, w, base);
                    (place.id, place.storage)
                }
            };
            let Ty::Struct(abi) = base_ty.deref() else {
                panic!("`.{name}` is not a place");
            };
            let index = abi.fields.iter().position(|f| f.name == name).unwrap() as u32;
            let index = m.constant(Ty::U32, index as u64);
            let pointer_type = m.ptr(storage, ty);
            let id = m.id();
            inst(
                &mut w.body,
                OP_ACCESS_CHAIN,
                &[pointer_type, id, pointer, index],
            );
            Place { id, ty, storage }
        }
        Node::Index { base, index } => {
            let base_ty = w.types.nodes[base as usize];
            let ty = w.types.nodes[node as usize];
            let index = value(m, w, index);
            match base_ty {
                // Pointer arithmetic, the one form `OpAccessChain` cannot express.
                Ty::Ptr { .. } => {
                    let base = value(m, w, base);
                    let pointer_type = m.ptr(PHYSICAL_STORAGE_BUFFER, ty);
                    let id = m.id();
                    inst(
                        &mut w.body,
                        OP_PTR_ACCESS_CHAIN,
                        &[pointer_type, id, base, index],
                    );
                    Place {
                        id,
                        ty,
                        storage: PHYSICAL_STORAGE_BUFFER,
                    }
                }
                _ => {
                    let place = place_of(m, w, base);
                    let pointer_type = m.ptr(place.storage, ty);
                    let id = m.id();
                    inst(
                        &mut w.body,
                        OP_ACCESS_CHAIN,
                        &[pointer_type, id, place.id, index],
                    );
                    Place {
                        id,
                        ty,
                        storage: place.storage,
                    }
                }
            }
        }
        other => panic!("{other:?} is not a place"),
    }
}

// -- expressions --------------------------------------------------------------------------

fn signed(ty: Ty) -> bool {
    matches!(
        ty.prim(),
        Prim::I8 | Prim::I16 | Prim::I32 | Prim::I64
    )
}

fn floating(ty: Ty) -> bool {
    matches!(ty.prim(), Prim::F16 | Prim::F32)
}

/// A value at the type the surrounding operation needs, converting or broadcasting where the
/// IR's own types fall short. SPIR-V has no implicit conversion at all, so this is not an
/// optimisation.
fn coerced(m: &mut M, w: &mut W, node: u32, want: Ty) -> u32 {
    let have = w.types.nodes[node as usize];
    let id = value(m, w, node);
    convert(m, w, id, have, want)
}

fn convert(m: &mut M, w: &mut W, id: u32, have: Ty, want: Ty) -> u32 {
    if have == want || want == Ty::Void || want == Ty::Infer {
        return id;
    }
    let convertible = matches!(have, Ty::Scalar(_) | Ty::Vec(..)) && matches!(want, Ty::Scalar(_) | Ty::Vec(..));
    if !convertible {
        return id;
    }
    // A scalar meeting a vector broadcasts, which is the only implicit widening either
    // language agrees on and the one Rust's operator overloads allow.
    if let (Ty::Scalar(_), Ty::Vec(p, n)) = (have, want) {
        let scalar = convert(m, w, id, have, Ty::Scalar(p));
        let ty = m.ty(want);
        let out = m.id();
        let mut operands = vec![ty, out];
        operands.extend(std::iter::repeat_n(scalar, n as usize));
        inst(&mut w.body, OP_COMPOSITE_CONSTRUCT, &operands);
        return out;
    }
    let ty = m.ty(want);
    let out = m.id();
    let op = match (floating(have), floating(want)) {
        (true, true) => OP_F_CONVERT,
        (true, false) if signed(want) => OP_CONVERT_F_TO_S,
        (true, false) => OP_CONVERT_F_TO_U,
        (false, true) if signed(have) => OP_CONVERT_S_TO_F,
        (false, true) => OP_CONVERT_U_TO_F,
        (false, false) if signed(want) => OP_S_CONVERT,
        (false, false) => OP_U_CONVERT,
    };
    // A same-width signed/unsigned change is a reinterpretation, and `OpUConvert` refuses it.
    if !floating(have) && !floating(want) && have.size() == want.size() {
        inst(&mut w.body, 124, &[ty, out, id]);
        return out;
    }
    inst(&mut w.body, op, &[ty, out, id]);
    out
}

/// Whether a node names memory. A local does; so does anything reached from one, or through a
/// device pointer. A parameter that is not a pointer is a value in a register and has no
/// address at all, which is why `input.color` on a fragment's stage-in struct is an extract
/// rather than an access chain.
fn is_place(w: &W, node: u32) -> bool {
    match w.f.nodes[node as usize] {
        Node::Local(_) => true,
        Node::Param(index) => matches!(w.f.params[index as usize].ty, Ty::Ptr { .. }),
        Node::Field { base, .. } | Node::Index { base, .. } => {
            matches!(w.types.nodes[base as usize], Ty::Ptr { .. }) || is_place(w, base)
        }
        _ => false,
    }
}

fn value(m: &mut M, w: &mut W, node: u32) -> u32 {
    let ty = w.types.nodes[node as usize];
    match w.f.nodes[node as usize] {
        Node::Local(_) => {
            let place = place_of(m, w, node);
            load(m, w, place)
        }
        Node::Index { base, index } => {
            if is_place(w, node) {
                let place = place_of(m, w, node);
                return load(m, w, place);
            }
            // A composite held in a register: `OpVectorExtractDynamic` is the only indexing
            // SPIR-V offers without an address.
            let base = value(m, w, base);
            let index = value(m, w, index);
            let type_id = m.ty(ty);
            let id = m.id();
            inst(&mut w.body, 77, &[type_id, id, base, index]);
            id
        }
        Node::Param(index) => w.params[index as usize],
        Node::Const(value) => m.literal(ty, value),
        Node::Field { base, name } => {
            if let Node::Param(index) = w.f.nodes[base as usize]
                && w.f.params[index as usize].binding == ir::Binding::Constants
            {
                let Ty::Struct(abi) = w.f.params[index as usize].ty else {
                    unreachable!("a constants parameter is a struct")
                };
                let field = abi.fields.iter().position(|f| f.name == name).unwrap() as u32;
                return w
                    .constants
                    .iter()
                    .find(|(p, f, _)| *p == index && *f == field)
                    .expect("every constant is declared in the prologue")
                    .2;
            }
            let base_ty = w.types.nodes[base as usize];
            match base_ty.deref() {
                // A swizzle is a shuffle of a value, never a place.
                Ty::Vec(..) | Ty::PackedVec(..) => {
                    let source = value(m, w, base);
                    let type_id = m.ty(ty);
                    let id = m.id();
                    if name.len() == 1 {
                        let index = component(name.as_bytes()[0]);
                        inst(&mut w.body, OP_COMPOSITE_EXTRACT, &[type_id, id, source, index]);
                    } else {
                        let mut operands = vec![type_id, id, source, source];
                        operands.extend(name.bytes().map(component));
                        inst(&mut w.body, OP_VECTOR_SHUFFLE, &operands);
                    }
                    id
                }
                Ty::Struct(abi) if !is_place(w, node) => {
                    let index = abi.fields.iter().position(|f| f.name == name).unwrap() as u32;
                    let source = value(m, w, base);
                    let type_id = m.ty(ty);
                    let id = m.id();
                    inst(&mut w.body, OP_COMPOSITE_EXTRACT, &[type_id, id, source, index]);
                    id
                }
                _ => {
                    let place = place_of(m, w, node);
                    load(m, w, place)
                }
            }
        }
        Node::Unary(op, operand) => {
            let operand_ty = w.types.nodes[operand as usize];
            let operand = value(m, w, operand);
            let type_id = m.ty(ty);
            let id = m.id();
            let op = match (op, floating(operand_ty), operand_ty.prim()) {
                (UnOp::Neg, true, _) => OP_F_NEGATE,
                (UnOp::Neg, false, _) => OP_S_NEGATE,
                (UnOp::Not, _, Prim::Bool) => OP_LOGICAL_NOT,
                (UnOp::Not, ..) => OP_NOT,
            };
            inst(&mut w.body, op, &[type_id, id, operand]);
            id
        }
        Node::Binary(op, lhs, rhs) => binary(m, w, node, op, lhs, rhs),
        Node::Cast { value: source, to } => coerced(m, w, source, to),
        Node::Construct { ty, args } => {
            let element = match ty {
                Ty::Vec(p, _) | Ty::PackedVec(p, _) => Ty::Scalar(p),
                Ty::Array(element, _) => *element,
                other => other,
            };
            let mut components: Vec<u32> = args.iter().map(|&a| coerced(m, w, a, element)).collect();
            // `f32x4::splat(x)` is one component standing for all of them.
            if let Ty::Vec(_, n) | Ty::PackedVec(_, n) = ty {
                while components.len() < n as usize {
                    components.push(components[0]);
                }
            }
            let type_id = m.ty(ty);
            let id = m.id();
            let mut operands = vec![type_id, id];
            operands.extend(&components);
            inst(&mut w.body, OP_COMPOSITE_CONSTRUCT, &operands);
            id
        }
        Node::StructLit { ty, fields } => {
            let Ty::Struct(abi) = ty else {
                unreachable!("a struct literal is a struct")
            };
            let mut members = Vec::new();
            for field in abi.fields {
                let Some(&(_, source)) = fields.iter().find(|(name, _)| *name == field.name) else {
                    panic!("`{}` has no value for `{}`", abi.name, field.name);
                };
                members.push(coerced(m, w, source, field.ty));
            }
            let type_id = m.ty(ty);
            let id = m.id();
            let mut operands = vec![type_id, id];
            operands.extend(&members);
            inst(&mut w.body, OP_COMPOSITE_CONSTRUCT, &operands);
            id
        }
        Node::Call { func, args } => {
            let callee = w.f.calls[func as usize];
            let arguments: Vec<u32> = args
                .iter()
                .enumerate()
                .map(|(i, &a)| coerced(m, w, a, callee.params[i].ty))
                .collect();
            let target = w
                .ids
                .iter()
                .find(|(f, _)| std::ptr::eq(*f, callee))
                .expect("every callee is in the module")
                .1;
            let type_id = m.ty(callee.ret);
            let id = m.id();
            let mut operands = vec![type_id, id, target];
            operands.extend(&arguments);
            inst(&mut w.body, OP_FUNCTION_CALL, &operands);
            id
        }
        Node::Intrinsic { op, args } => intrinsic(m, w, node, op, args),
        other => panic!("{other:?} is not a SPIR-V expression"),
    }
}

fn component(letter: u8) -> u32 {
    match letter {
        b'x' => 0,
        b'y' => 1,
        b'z' => 2,
        _ => 3,
    }
}

fn binary(m: &mut M, w: &mut W, node: u32, op: BinOp, lhs: u32, rhs: u32) -> u32 {
    let want = match (op.compare() || op.logical(), matches!(op, BinOp::Shl | BinOp::Shr)) {
        (_, true) => Ty::Infer,
        (true, _) => w.types.nodes[lhs as usize],
        (false, _) => w.types.nodes[node as usize],
    };
    let operand_ty = match want {
        Ty::Infer => w.types.nodes[lhs as usize],
        other => other,
    };
    let lhs = coerced(m, w, lhs, want);
    let rhs = coerced(m, w, rhs, want);
    let float = floating(operand_ty);
    let sign = signed(operand_ty);
    let opcode = match op {
        BinOp::Add if float => OP_F_ADD,
        BinOp::Add => OP_I_ADD,
        BinOp::Sub if float => OP_F_SUB,
        BinOp::Sub => OP_I_SUB,
        BinOp::Mul if float => OP_F_MUL,
        BinOp::Mul => OP_I_MUL,
        BinOp::Div if float => OP_F_DIV,
        BinOp::Div if sign => OP_S_DIV,
        BinOp::Div => OP_U_DIV,
        BinOp::Rem if float => OP_F_REM,
        BinOp::Rem if sign => OP_S_REM,
        BinOp::Rem => OP_U_MOD,
        BinOp::And => OP_BITWISE_AND,
        BinOp::Or => OP_BITWISE_OR,
        BinOp::Xor => OP_BITWISE_XOR,
        BinOp::Shl => OP_SHIFT_LEFT_LOGICAL,
        BinOp::Shr if sign => OP_SHIFT_RIGHT_ARITHMETIC,
        BinOp::Shr => OP_SHIFT_RIGHT_LOGICAL,
        BinOp::AndAnd => OP_LOGICAL_AND,
        BinOp::OrOr => OP_LOGICAL_OR,
        BinOp::Eq if float => OP_F_ORD_EQUAL,
        BinOp::Eq => OP_I_EQUAL,
        BinOp::Ne if float => OP_F_ORD_NOT_EQUAL,
        BinOp::Ne => OP_I_NOT_EQUAL,
        BinOp::Lt if float => OP_F_ORD_LESS_THAN,
        BinOp::Lt if sign => OP_S_LESS_THAN,
        BinOp::Lt => OP_U_LESS_THAN,
        BinOp::Le if float => OP_F_ORD_LESS_THAN_EQUAL,
        BinOp::Le if sign => OP_S_LESS_THAN_EQUAL,
        BinOp::Le => OP_U_LESS_THAN_EQUAL,
        BinOp::Gt if float => OP_F_ORD_GREATER_THAN,
        BinOp::Gt if sign => OP_S_GREATER_THAN,
        BinOp::Gt => OP_U_GREATER_THAN,
        BinOp::Ge if float => OP_F_ORD_GREATER_THAN_EQUAL,
        BinOp::Ge if sign => OP_S_GREATER_THAN_EQUAL,
        BinOp::Ge => OP_U_GREATER_THAN_EQUAL,
    };
    let type_id = m.ty(w.types.nodes[node as usize]);
    let id = m.id();
    inst(&mut w.body, opcode, &[type_id, id, lhs, rhs]);
    id
}

fn intrinsic(m: &mut M, w: &mut W, node: u32, op: Intrinsic, args: &[u32]) -> u32 {
    let ty = w.types.nodes[node as usize];

    // The handle intrinsics come first: their arguments have types the arithmetic path below
    // would try to make sense of.
    match op {
        Intrinsic::Sample
        | Intrinsic::SampleLevel
        | Intrinsic::Load
        | Intrinsic::Store
        | Intrinsic::Dimensions => return image(m, w, node, op, args),
        Intrinsic::AtomicAdd
        | Intrinsic::AtomicMin
        | Intrinsic::AtomicMax
        | Intrinsic::AtomicExchange => return atomic(m, w, op, args),
        _ => {}
    }

    if op == Intrinsic::Barrier {
        // Workgroup scope, with both memory classes made visible: the same guarantee
        // `threadgroup_barrier(mem_device)` gives on Metal.
        let scope = m.constant(Ty::U32, 2);
        let semantics = m.constant(Ty::U32, 0x2 | 0x40 | 0x100);
        inst(&mut w.body, OP_CONTROL_BARRIER, &[scope, scope, semantics]);
        return 0;
    }
    if op == Intrinsic::Unpack {
        return value(m, w, args[0]);
    }

    let glsl = |m: &mut M, w: &mut W, number: u32, operands: &[u32]| {
        let type_id = m.ty(ty);
        let id = m.id();
        let mut all = vec![type_id, id, m.glsl, number];
        all.extend(operands);
        inst(&mut w.body, OP_EXT_INST, &all);
        id
    };

    let first = w.types.nodes[args[0] as usize];
    let float = matches!(first, Ty::Void) || floating(first);
    let sign = !matches!(first, Ty::Void) && signed(first);

    // Everything else takes its arguments at the result's type, which is what `infer` recorded.
    let element = match ty {
        Ty::Vec(p, _) if matches!(op, Intrinsic::Dot | Intrinsic::Length | Intrinsic::Distance) => Ty::Scalar(p),
        other => other,
    };
    let _ = element;
    let a: Vec<u32> = match op {
        Intrinsic::Dot | Intrinsic::Length | Intrinsic::Distance | Intrinsic::Cross | Intrinsic::Normalize => {
            args.iter().map(|&arg| value(m, w, arg)).collect()
        }
        Intrinsic::Mix => {
            let mut out = vec![coerced(m, w, args[0], ty), coerced(m, w, args[1], ty)];
            out.push(coerced(m, w, args[2], ty));
            out
        }
        Intrinsic::Select => vec![
            coerced(m, w, args[0], ty),
            value(m, w, args[1]),
            coerced(m, w, args[2], ty),
        ],
        _ => args.iter().map(|&arg| coerced(m, w, arg, ty)).collect(),
    };

    match op {
        Intrinsic::Sqrt => glsl(m, w, GLSL_SQRT, &a),
        Intrinsic::InverseSqrt => glsl(m, w, GLSL_INVERSE_SQRT, &a),
        Intrinsic::Abs if float => glsl(m, w, GLSL_F_ABS, &a),
        Intrinsic::Abs => glsl(m, w, GLSL_S_ABS, &a),
        Intrinsic::Sign => glsl(m, w, GLSL_F_SIGN, &a),
        Intrinsic::Floor => glsl(m, w, GLSL_FLOOR, &a),
        Intrinsic::Ceil => glsl(m, w, GLSL_CEIL, &a),
        Intrinsic::Round => glsl(m, w, GLSL_ROUND, &a),
        Intrinsic::Fract => glsl(m, w, GLSL_FRACT, &a),
        Intrinsic::Sin => glsl(m, w, GLSL_SIN, &a),
        Intrinsic::Cos => glsl(m, w, GLSL_COS, &a),
        Intrinsic::Tan => glsl(m, w, GLSL_TAN, &a),
        Intrinsic::Asin => glsl(m, w, GLSL_ASIN, &a),
        Intrinsic::Acos => glsl(m, w, GLSL_ACOS, &a),
        Intrinsic::Atan => glsl(m, w, GLSL_ATAN, &a),
        Intrinsic::Atan2 => glsl(m, w, GLSL_ATAN2, &a),
        Intrinsic::Exp => glsl(m, w, GLSL_EXP, &a),
        Intrinsic::Exp2 => glsl(m, w, GLSL_EXP2, &a),
        Intrinsic::Log => glsl(m, w, GLSL_LOG, &a),
        Intrinsic::Log2 => glsl(m, w, GLSL_LOG2, &a),
        Intrinsic::Pow => glsl(m, w, GLSL_POW, &a),
        Intrinsic::Min if float => glsl(m, w, GLSL_F_MIN, &a),
        Intrinsic::Min if sign => glsl(m, w, GLSL_S_MIN, &a),
        Intrinsic::Min => glsl(m, w, GLSL_U_MIN, &a),
        Intrinsic::Max if float => glsl(m, w, GLSL_F_MAX, &a),
        Intrinsic::Max if sign => glsl(m, w, GLSL_S_MAX, &a),
        Intrinsic::Max => glsl(m, w, GLSL_U_MAX, &a),
        Intrinsic::Clamp if float => glsl(m, w, GLSL_F_CLAMP, &a),
        Intrinsic::Clamp if sign => glsl(m, w, GLSL_S_CLAMP, &a),
        Intrinsic::Clamp => glsl(m, w, GLSL_U_CLAMP, &a),
        Intrinsic::Mix => glsl(m, w, GLSL_F_MIX, &a),
        // Rust reads `x.step(edge)`; GLSL takes the edge first.
        Intrinsic::Step => glsl(m, w, GLSL_STEP, &[a[1], a[0]]),
        Intrinsic::SmoothStep => glsl(m, w, GLSL_SMOOTH_STEP, &[a[1], a[2], a[0]]),
        Intrinsic::Length => glsl(m, w, GLSL_LENGTH, &a),
        Intrinsic::Distance => glsl(m, w, GLSL_DISTANCE, &a),
        Intrinsic::Cross => glsl(m, w, GLSL_CROSS, &a),
        Intrinsic::Normalize => glsl(m, w, GLSL_NORMALIZE, &a),
        Intrinsic::Reflect => glsl(m, w, GLSL_REFLECT, &a),
        Intrinsic::Dot => {
            let type_id = m.ty(ty);
            let id = m.id();
            inst(&mut w.body, OP_DOT, &[type_id, id, a[0], a[1]]);
            id
        }
        Intrinsic::Any | Intrinsic::All => {
            let type_id = m.ty(ty);
            let id = m.id();
            let opcode = match op {
                Intrinsic::Any => OP_ANY,
                _ => OP_ALL,
            };
            inst(&mut w.body, opcode, &[type_id, id, a[0]]);
            id
        }
        Intrinsic::Select => {
            let type_id = m.ty(ty);
            let id = m.id();
            inst(&mut w.body, OP_SELECT, &[type_id, id, a[1], a[0], a[2]]);
            id
        }
        Intrinsic::Transpose => {
            let type_id = m.ty(ty);
            let id = m.id();
            inst(&mut w.body, OP_TRANSPOSE, &[type_id, id, a[0]]);
            id
        }
        Intrinsic::Mul => {
            let left = w.types.nodes[args[0] as usize];
            let right = w.types.nodes[args[1] as usize];
            let a: Vec<u32> = args.iter().map(|&arg| value(m, w, arg)).collect();
            let opcode = match (left, right) {
                (Ty::Mat(..), Ty::Vec(..)) => OP_MATRIX_TIMES_VECTOR,
                (Ty::Mat(..), Ty::Mat(..)) => OP_MATRIX_TIMES_MATRIX,
                _ => panic!("mul({left:?}, {right:?})"),
            };
            let type_id = m.ty(ty);
            let id = m.id();
            inst(&mut w.body, opcode, &[type_id, id, a[0], a[1]]);
            id
        }
        other => panic!("{other:?} has no SPIR-V form"),
    }
}

fn atomic(m: &mut M, w: &mut W, op: Intrinsic, args: &[u32]) -> u32 {
    let place = match w.types.nodes[args[0] as usize] {
        // The pointer is the value here: `p.atomic_add(1)` on a device pointer.
        Ty::Ptr { .. } => Place {
            id: value(m, w, args[0]),
            ty: Ty::U32,
            storage: PHYSICAL_STORAGE_BUFFER,
        },
        _ => place_of(m, w, args[0]),
    };
    let operand = coerced(m, w, args[1], Ty::U32);
    let scope = m.constant(Ty::U32, 1);
    let semantics = m.constant(Ty::U32, 0);
    let type_id = m.ty(Ty::U32);
    let id = m.id();
    let opcode = match op {
        Intrinsic::AtomicAdd => OP_ATOMIC_I_ADD,
        Intrinsic::AtomicMin => OP_ATOMIC_U_MIN,
        Intrinsic::AtomicMax => OP_ATOMIC_U_MAX,
        _ => OP_ATOMIC_EXCHANGE,
    };
    inst(
        &mut w.body,
        opcode,
        &[type_id, id, place.id, scope, semantics, operand],
    );
    id
}


/// The three descriptor arrays the device binds, matching the Vulkan half of the old prelude:
/// sampled images at set 0, storage images at set 1, samplers at set 2, each at binding 0. A
/// handle's low word is the index into one of them.
fn descriptor_array(m: &mut M, kind: HandleKind) -> (u32, u32) {
    m.capability(5302);
    m.extension("SPV_EXT_descriptor_indexing");
    let set = match kind {
        HandleKind::Sampler => 2,
        HandleKind::Texture2dRW | HandleKind::Texture3dRW => 1,
        _ => 0,
    };
    let cached = match set {
        1 => (m.storage, m.storage_image),
        2 => (m.samplers, m.sampler_type),
        _ => (m.textures, m.texture_image),
    };
    if cached.0 != 0 {
        return cached;
    }

    let element = match set {
        2 => {
            let id = m.id();
            inst(&mut m.globals, OP_TYPE_SAMPLER, &[id]);
            id
        }
        _ => {
            let float = m.ty(Ty::F32);
            let id = m.id();
            // 2D, no depth, not arrayed, single-sampled. A sampled image says so; a storage
            // image says "used without a sampler" and carries an unknown format, which is what
            // lets one array serve every format the host binds into it.
            let sampled = match set {
                1 => 2,
                _ => 1,
            };
            inst(&mut m.globals, OP_TYPE_IMAGE, &[id, float, 1, 0, 0, 0, sampled, 0]);
            if set == 1 {
                m.capability(55);
                m.capability(56);
            }
            id
        }
    };
    let array = m.id();
    inst(&mut m.globals, OP_TYPE_RUNTIME_ARRAY, &[array, element]);
    let pointer = m.pointer(UNIFORM_CONSTANT, array);
    let variable = m.id();
    inst(&mut m.globals, OP_VARIABLE, &[pointer, variable, UNIFORM_CONSTANT]);
    m.decorate(variable, DEC_DESCRIPTOR_SET, &[set]);
    m.decorate(variable, DEC_BINDING, &[0]);
    m.name(
        variable,
        match set {
            1 => "nga_storage",
            2 => "nga_samplers",
            _ => "nga_textures",
        },
    );
    match set {
        1 => {
            m.storage = variable;
            m.storage_image = element;
        }
        2 => {
            m.samplers = variable;
            m.sampler_type = element;
        }
        _ => {
            m.textures = variable;
            m.texture_image = element;
        }
    }
    (variable, element)
}

/// One descriptor out of an array, indexed by the handle's low word.
fn descriptor(m: &mut M, w: &mut W, kind: HandleKind, handle: u32) -> u32 {
    let (array, element) = descriptor_array(m, kind);
    if !w.interface.contains(&array) {
        w.interface.push(array);
    }
    let u32_id = m.ty(Ty::U32);
    let index = m.id();
    inst(&mut w.body, OP_U_CONVERT, &[u32_id, index, handle]);
    let pointer = m.pointer(UNIFORM_CONSTANT, element);
    let chain = m.id();
    inst(&mut w.body, OP_ACCESS_CHAIN, &[pointer, chain, array, index]);
    let loaded = m.id();
    inst(&mut w.body, OP_LOAD, &[element, loaded, chain]);
    loaded
}

fn image(m: &mut M, w: &mut W, node: u32, op: Intrinsic, args: &[u32]) -> u32 {
    let ty = w.types.nodes[node as usize];
    let Ty::Handle(kind) = w.types.nodes[args[0] as usize] else {
        panic!("an image intrinsic takes a handle");
    };
    let handle = value(m, w, args[0]);
    let image_id = descriptor(m, w, kind, handle);
    match op {
        Intrinsic::Sample | Intrinsic::SampleLevel => {
            let Ty::Handle(sampler_kind) = w.types.nodes[args[1] as usize] else {
                panic!("sampling takes a sampler");
            };
            let sampler_handle = value(m, w, args[1]);
            let sampler = descriptor(m, w, sampler_kind, sampler_handle);
            let uv = value(m, w, args[2]);
            let (_, element) = descriptor_array(m, kind);
            if m.sampled_image == 0 {
                let id = m.id();
                inst(&mut m.globals, OP_TYPE_SAMPLED_IMAGE, &[id, element]);
                m.sampled_image = id;
            }
            let sampled = m.id();
            inst(
                &mut w.body,
                OP_SAMPLED_IMAGE,
                &[m.sampled_image, sampled, image_id, sampler],
            );
            let type_id = m.ty(ty);
            let id = m.id();
            match op {
                Intrinsic::SampleLevel => {
                    let level = coerced(m, w, args[3], Ty::F32);
                    inst(
                        &mut w.body,
                        OP_IMAGE_SAMPLE_EXPLICIT_LOD,
                        &[type_id, id, sampled, uv, 0x2, level],
                    );
                }
                _ => inst(
                    &mut w.body,
                    OP_IMAGE_SAMPLE_IMPLICIT_LOD,
                    &[type_id, id, sampled, uv],
                ),
            }
            id
        }
        Intrinsic::Load => {
            let at = value(m, w, args[1]);
            let type_id = m.ty(ty);
            let id = m.id();
            inst(&mut w.body, OP_IMAGE_READ, &[type_id, id, image_id, at]);
            id
        }
        Intrinsic::Store => {
            let at = value(m, w, args[1]);
            let source = coerced(m, w, args[2], Ty::Vec(Prim::F32, 4));
            inst(&mut w.body, OP_IMAGE_WRITE, &[image_id, at, source]);
            0
        }
        _ => {
            let type_id = m.ty(Ty::Vec(Prim::U32, 2));
            let id = m.id();
            inst(&mut w.body, OP_IMAGE_QUERY_SIZE, &[type_id, id, image_id]);
            id
        }
    }
}
