//! `#[gpu]` end to end: the body compiles as Rust, the IR beside it says the same thing, and
//! both backends agree about what it means.

use nga::ir::{self, Stage, Ty};
use nga::*;

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
