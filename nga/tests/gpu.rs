//! `#[gpu]` end to end: the body compiles as Rust, the IR beside it says the same thing, and
//! both backends agree about what it means.

use nga::ir::{self, Stage, Ty};
use nga::*;
use std::mem::offset_of;

#[gpu_struct]
struct Scene {
    pixels: GpuMut<'static, [f32x4]>,
    width: u32,
    height: u32,
    tint: f32x4,
}

#[gpu(compute(8, 8, 1))]
fn clear(mut root: GpuMut<'static, Scene>, #[thread_id] id: u32x3) {
    if id.x >= root.width || id.y >= root.height {
        return;
    }
    let index = id.x + id.y * root.width;
    let mut colour = f32x4::new(id.x as f32 / 64.0, id.y as f32 / 64.0, 0.0, 1.0);
    colour = colour * root.tint;
    root.pixels[index] = colour;
}

#[gpu]
fn luma(colour: f32x4) -> f32 {
    colour.xyz().dot(f32x3::new(0.2126, 0.7152, 0.0722))
}

#[gpu(compute(64))]
fn grey(mut root: GpuMut<'static, Scene>, #[thread_id] id: u32x3) {
    let mut total = 0.0;
    for i in 0..4 {
        total += luma(root.pixels[id.x * 4 + i]);
    }
    root.pixels[id.x] = f32x4::splat(total * 0.25);
}

#[gpu_struct]
struct Varyings {
    position: f32x4,
    colour: f32x3,
}

#[gpu(vertex)]
fn vertex_main(root: Gpu<'static, Scene>, #[vertex_id] vid: u32) -> Varyings {
    let uv = f32x2::new((vid & 1) as f32, (vid >> 1) as f32);
    Varyings {
        position: f32x4::new(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0),
        colour: f32x3::new(uv.x, uv.y, root.tint.z),
    }
}

#[gpu(fragment)]
fn fragment_main(input: Varyings) -> f32x4 {
    let mut c = input.colour;
    if c.x > 0.5 {
        c = c * 0.5;
    }
    f32x4::new(c.x, c.y, c.z, 1.0)
}

/// A pointer then a matrix. `mvp` is sixteen-aligned and the pointer is eight bytes, so eight
/// bytes of hole sit between them, and that hole is the whole reason this struct exists. Both
/// emitters have to reproduce it: SPIR-V as an `Offset` decoration, MSL as a padding member.
#[gpu_struct(align(16))]
struct Transform {
    scene: Gpu<'static, Scene>,
    mvp: f32x4x4,
}

#[gpu(compute(64))]
fn transform(root: Gpu<'static, Transform>, #[thread_id] id: u32x3) {
    let scene = root.scene;
    let mut out = scene.pixels;
    out[id.x] = root.mvp.mul(scene.tint);
}

#[test]
fn shape() {
    assert_eq!(CLEAR.name, "clear");
    assert_eq!(CLEAR.stage, Stage::Compute([8, 8, 1]));
    assert_eq!(CLEAR.params.len(), 2);
    assert_eq!(CLEAR.params[0].binding, ir::Binding::Root);
    assert_eq!(
        CLEAR.params[1].binding,
        ir::Binding::Builtin(ir::Builtin::ThreadId)
    );
    assert_eq!(LUMA.ret, Ty::F32);
    assert_eq!(GREY.calls.len(), 1);
    assert_eq!(GREY.calls[0].name, "luma");
}

#[test]
fn types() {
    let types = ir::infer(&CLEAR);
    // `index`, then `colour`.
    assert_eq!(types.locals[0], Ty::U32);
    assert_eq!(types.locals[1], Ty::Vec(ir::Prim::F32, 4));

    let types = ir::infer(&GREY);
    assert_eq!(types.locals[0], Ty::F32);
    assert_eq!(types.locals[1], Ty::I32);
}

#[test]
fn metal() {
    let source = nga::msl::module(&[&CLEAR, &GREY, &VERTEX_MAIN, &FRAGMENT_MAIN]);
    println!("{source}");
    assert!(source.contains("kernel void clear("));
    assert!(source.contains("float luma("));
    // `Scene` is memory, so its members land on the offsets Rust chose.
    assert!(source.contains("struct Scene {"));
}

/// The ahead-of-time path: both targets written from one host, which is also how the Metal
/// output gets checked from a machine that cannot build for Apple.
#[test]
fn dump_writes_both_targets() {
    let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../target/generated"));
    let (spirv, metal) = nga::dump(&dir, "gpu-test", &[&CLEAR, &GREY, &VERTEX_MAIN, &FRAGMENT_MAIN]).unwrap();
    assert!(std::fs::metadata(&spirv).unwrap().len() > 0);
    assert!(std::fs::read_to_string(&metal).unwrap().contains("kernel void clear("));
}

/// Emits the module and runs `spirv-val` over it. The SDK is not a build dependency, so a
/// machine without one skips the check rather than failing it.
#[test]
fn vulkan() {
    let words = nga::spirv::words(&[&CLEAR, &GREY, &VERTEX_MAIN, &FRAGMENT_MAIN]);
    assert_eq!(words[0], 0x0723_0203);
    let path = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../target"))
        .join("gpu-test.spv");
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();

    let Ok(sdk) = std::env::var("VULKAN_SDK") else {
        eprintln!("no VULKAN_SDK; skipping validation");
        return;
    };
    let validator = std::path::PathBuf::from(sdk).join("Bin").join("spirv-val");
    let Ok(output) = std::process::Command::new(&validator)
        .args(["--target-env", "vulkan1.3", "--scalar-block-layout"])
        .arg(&path)
        .output()
    else {
        eprintln!("no spirv-val; skipping validation");
        return;
    };
    assert!(
        output.status.success(),
        "spirv-val rejected the module:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn reachable() {
    let order: Vec<&str> = GREY.reachable().iter().map(|f| f.name).collect();
    assert_eq!(order, ["luma", "grey"]);
}


// -- struct layout -------------------------------------------------------------------------
//
// `#[gpu_struct]` records what Rust chose and both emitters write those numbers out rather
// than arranging for them to happen. So this does not catch a shader compiler picking a layout
// behind our back, the way it had to when shaders were Slang. It catches an emitter that
// stopped using the offsets it was handed.

const OP_NAME: u16 = 5;
const OP_MEMBER_NAME: u16 = 6;
const OP_MEMBER_DECORATE: u16 = 72;
const DECORATION_OFFSET: u32 = 35;

/// `(struct name, field name) -> byte offset`, read straight out of the module. Reflection
/// would not do: a struct reached through a `PhysicalStorageBuffer` pointer, which is every
/// struct in this design, is not part of the binding interface and appears only here.
fn spirv_offsets(words: &[u32]) -> std::collections::HashMap<(String, String), usize> {
    let mut names = std::collections::HashMap::new();
    let mut members = std::collections::HashMap::new();
    let mut offsets = std::collections::HashMap::new();

    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        assert!(count > 0, "zero-length SPIR-V instruction");
        let operands = &words[i + 1..i + count];
        match opcode {
            OP_NAME => {
                names.insert(operands[0], text(&operands[1..]));
            }
            OP_MEMBER_NAME => {
                members.insert((operands[0], operands[1]), text(&operands[2..]));
            }
            OP_MEMBER_DECORATE if operands.get(2) == Some(&DECORATION_OFFSET) => {
                offsets.insert((operands[0], operands[1]), operands[3] as usize);
            }
            _ => {}
        }
        i += count;
    }

    let mut out = std::collections::HashMap::new();
    for ((id, member), offset) in offsets {
        let (Some(name), Some(field)) = (names.get(&id), members.get(&(id, member))) else {
            continue;
        };
        out.insert((name.clone(), field.clone()), offset);
    }
    out
}

fn text(words: &[u32]) -> String {
    let bytes: Vec<u8> = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .take_while(|&byte| byte != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[test]
fn the_spirv_offsets_are_the_rust_offsets() {
    let found = spirv_offsets(&nga::spirv::words(&[&CLEAR, &TRANSFORM]));
    assert!(!found.is_empty(), "no member offsets, so this would pass vacuously");

    let mut checked = 0;
    for abi in [<Scene as DeviceAbi>::ABI, <Transform as DeviceAbi>::ABI] {
        for field in abi.fields {
            let key = (abi.name.to_string(), field.name.to_string());
            let Some(&emitted) = found.get(&key) else {
                continue;
            };
            assert_eq!(
                emitted, field.offset,
                "{}::{} is at {} in Rust but {emitted} in the module",
                abi.name, field.name, field.offset
            );
            checked += 1;
        }
    }
    assert!(checked >= 6, "only {checked} fields were cross-checked");
}

#[test]
fn the_msl_pads_to_the_same_offsets() {
    let source = nga::msl::module(&[&CLEAR, &TRANSFORM]);
    let body = source.split_once("struct Transform {").expect("no Transform struct").1;
    let body = &body[..body.find("};").unwrap()];

    assert!(
        body.contains("uint _nga_pad0[2];"),
        "expected 8 bytes of padding before `mvp`, got:
{body}"
    );
    let scene = body.find("scene").unwrap();
    let pad = body.find("_nga_pad0").unwrap();
    let mvp = body.find("mvp").unwrap();
    assert!(scene < pad && pad < mvp, "the padding is in the wrong place:
{body}");

    assert_eq!(offset_of!(Transform, mvp), 16);
    assert_eq!(size_of::<Transform>(), 80);
}
