//! Replaces the MSL `static_assert(__builtin_offsetof(...))` the Metal port relies on.
//!
//! Slang has no `offsetof`, and offline SPIR-V compilation means the shader is built before
//! `TypeAbi` exists, so the check cannot live in the shader or in `build.rs`. It runs here.
//!
//! The source of truth is the SPIR-V itself, not `slangc -reflection-json`: reflection only
//! describes the binding interface (push constants, descriptor bindings). Structs reached
//! through a `PhysicalStorageBuffer` pointer — which is every struct in this design — appear
//! only as `OpMemberDecorate ... Offset` in the module.

use nga_vk::*;
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
                 add padding to the Slang struct (see nga_vk::slang::declare)",
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
    let declared = nga_vk::slang::declare(&<Data as DeviceAbi>::ABI);
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
