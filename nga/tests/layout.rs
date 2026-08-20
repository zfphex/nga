//! Replaces the MSL `static_assert(__builtin_offsetof(...))` the Metal port relies on.
//!
//! Slang has no `offsetof`, and offline SPIR-V compilation means the shader is built before
//! `TypeAbi` exists, so the check cannot live in the shader or in `build.rs`. It runs here.
//!
//! The source of truth is the SPIR-V itself, not `slangc -reflection-json`: reflection only
//! describes the binding interface (push constants, descriptor bindings). Structs reached
//! through a `PhysicalStorageBuffer` pointer — which is every struct in this design — appear
//! only as `OpMemberDecorate ... Offset` in the module.

use nga::*;
use std::collections::HashMap;
use std::mem::offset_of;

#[gpu_struct]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[gpu_struct]
struct Data {
    vertices: Gpu<'static, Vertex>,
    mvp: f32x4x4,
}

const OP_NAME: u16 = 5;
const OP_MEMBER_NAME: u16 = 6;
const OP_MEMBER_DECORATE: u16 = 72;
const DECORATION_OFFSET: u32 = 35;

/// `(struct name, field name) -> byte offset`, read straight out of the module.
fn spirv_offsets(spirv: &[u8]) -> HashMap<(String, String), usize> {
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(words[0], 0x0723_0203, "not a SPIR-V module");

    let mut names: HashMap<u32, String> = HashMap::new();
    let mut members: HashMap<(u32, u32), String> = HashMap::new();
    let mut offsets: HashMap<(u32, u32), usize> = HashMap::new();

    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        assert!(count > 0, "zero-length SPIR-V instruction");
        let operands = &words[i + 1..i + count];
        match opcode {
            OP_NAME => {
                names.insert(operands[0], decode(&operands[1..]));
            }
            OP_MEMBER_NAME => {
                members.insert((operands[0], operands[1]), decode(&operands[2..]));
            }
            OP_MEMBER_DECORATE if operands.get(2) == Some(&DECORATION_OFFSET) => {
                offsets.insert((operands[0], operands[1]), operands[3] as usize);
            }
            _ => {}
        }
        i += count;
    }

    offsets
        .into_iter()
        .filter_map(|((id, member), offset)| {
            let struct_name = names.get(&id)?;
            let field = members.get(&(id, member))?;
            // Slang suffixes layout variants, e.g. `Data_natural`.
            let struct_name = struct_name.split('_').next().unwrap_or(struct_name);
            Some(((struct_name.to_string(), field.clone()), offset))
        })
        .collect()
}

fn decode(words: &[u32]) -> String {
    let bytes: Vec<u8> = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .take_while(|&byte| byte != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[test]
fn rust_and_slang_agree_on_field_offsets() {
    let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.spv"));
    let found = spirv_offsets(spirv);
    assert!(
        !found.is_empty(),
        "no member offsets in the module — the check would pass vacuously"
    );

    let mut checked = 0;
    for abi in [<Vertex as DeviceAbi>::ABI, <Data as DeviceAbi>::ABI] {
        for (name, offset, _) in abi.fields {
            let key = (abi.name.to_string(), name.to_string());
            let Some(&slang) = found.get(&key) else {
                continue;
            };
            assert_eq!(
                slang, *offset,
                "{}::{name} is at {offset} in Rust but {slang} in Slang — \
                 add padding to the Slang struct (see nga::slang::declare)",
                abi.name
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 4,
        "only {checked} fields were cross-checked; the shader and the Rust structs have drifted apart"
    );
}

/// The padding synthesiser is what makes a mismatch fixable rather than merely detectable,
/// so it is checked directly: `Data` must gain 8 bytes between the pointer and the matrix.
#[test]
fn padding_is_synthesised_from_rust_offsets() {
    let declared = nga::slang::declare(&<Data as DeviceAbi>::ABI);
    assert!(
        declared.contains("uint _nga_pad0[2];"),
        "expected 8 bytes of padding before `mvp`, got:\n{declared}"
    );
    let vertices = declared.find("vertices").unwrap();
    let pad = declared.find("_nga_pad0").unwrap();
    let mvp = declared.find("mvp").unwrap();
    assert!(
        vertices < pad && pad < mvp,
        "padding is in the wrong place:\n{declared}"
    );
}

#[test]
fn rust_layout_is_what_the_shader_assumes() {
    assert_eq!(size_of::<Vertex>(), 24);
    assert_eq!(offset_of!(Vertex, color), 12);
    assert_eq!(size_of::<Data>(), 80);
    assert_eq!(offset_of!(Data, mvp), 16);
}

/// The other half of the M1 gate, and the mitigation for the fact that `metal.rs` cannot be
/// compiled on a Windows host at all: slangc's Metal backend runs anywhere, so the *layout*
/// half of the Metal port is checkable here even though the Objective-C half is not.
///
/// slangc packs every member of a natural-layout struct (`packed_float3`,
/// `array<packed_float4, 4>`), so the padding synthesised from the Rust offsets lands each
/// field on the same byte on both targets. If that ever stops being true, this fails long
/// before anyone gets to a Mac.
#[test]
fn metal_output_packs_the_same_way() {
    let metal = include_str!(concat!(env!("OUT_DIR"), "/triangle.metal"));

    let vertex = metal
        .split_once("struct Vertex_natural_0")
        .expect("no Vertex struct in the Metal output")
        .1;
    let vertex = &vertex[..vertex.find("};").unwrap()];
    assert!(
        vertex.matches("packed_float3").count() == 2,
        "float3 did not pack on the Metal target, so `[f32; 3]` would be 16 bytes there:\n{vertex}"
    );

    let data = metal
        .split_once("struct Data_natural_0")
        .expect("no Data struct in the Metal output")
        .1;
    let data = &data[..data.find("};").unwrap()];
    // The 8 bytes `declare()` inserts between the pointer and the matrix must survive codegen;
    // without them the matrix would sit at offset 8 on Metal and 16 in Rust.
    assert!(
        data.contains("_pad0"),
        "the padding member was stripped from the Metal struct:\n{data}"
    );
    // The matrix lowers to a named storage struct rather than an inline member, so the packing
    // has to be checked where that struct is declared.
    let storage = metal
        .split_once("struct _MatrixStorage_float4x4_ColMajornatural_0")
        .expect("no float4x4 storage struct in the Metal output")
        .1;
    let storage = &storage[..storage.find("};").unwrap()];
    assert!(
        storage.contains("packed_float4"),
        "float4x4 did not lower to a packed column array:\n{storage}"
    );
}

/// The prelude is the one file that forks per backend, so the fork is the one place a layout
/// mismatch can hide. A bindless handle is eight bytes on both targets — a `uint2` descriptor
/// index on Vulkan, an `MTLResourceID` on Metal — and `material.slang` exists to prove it.
///
/// Vulkan's half is checked against the SPIR-V; Metal's is checked against the MSL, which
/// slangc emits on any host.
#[test]
fn handles_are_eight_bytes_on_both_targets() {
    let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/material.spv"));
    let found = spirv_offsets(spirv);
    for (field, expected) in [("albedo", 0), ("samp", 8), ("target", 16), ("uv", 24)] {
        let key = ("Material".to_string(), field.to_string());
        let &offset = found
            .get(&key)
            .unwrap_or_else(|| panic!("Material::{field} is not in the module"));
        assert_eq!(
            offset, expected,
            "Material::{field} moved to {offset}; a Vulkan handle is no longer 8 bytes"
        );
    }

    let metal = include_str!(concat!(env!("OUT_DIR"), "/material.metal"));
    let material = metal
        .split_once("struct Material_natural_0")
        .expect("no Material struct in the Metal output")
        .1;
    let material = &material[..material.find("};").unwrap()];
    // On Metal the handles become argument-buffer members outright. If slangc ever emitted a
    // wrapper struct or an index instead, the offsets would stop matching the Vulkan side.
    for spelling in ["texture2d<float, access::sample>", "sampler", "access::read_write"] {
        assert!(
            material.contains(spelling),
            "expected `{spelling}` in the Metal Material struct:\n{material}"
        );
    }
}
