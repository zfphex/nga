//! `#[gpu_mod]` end to end: the body compiles as Rust, the IR beside it says the same thing,
//! and both backends agree about what it means.

use nga::ir::{self, Stage, Ty};
use nga::*;
use shader::*;
use std::mem::{align_of, offset_of, size_of};

/// One tag for the whole module: every struct in it is shared with the GPU and every function
/// is a shader, so nothing inside carries a tag of its own except the stage of an entry point.
#[gpu_mod]
mod shader {
    use nga::*;

    pub struct Scene {
        pub pixels: GpuMut<'static, [f32x4]>,
        pub width: u32,
        pub height: u32,
        pub tint: f32x4,
    }

    #[compute(8, 8, 1)]
    fn clear(mut root: GpuMut<'static, Scene>, #[thread_id] id: u32x3) {
        if id.x >= root.width || id.y >= root.height {
            return;
        }
        let index = id.x + id.y * root.width;
        let mut colour = f32x4::new(id.x as f32 / 64.0, id.y as f32 / 64.0, 0.0, 1.0);
        colour = colour * root.tint;
        root.pixels[index] = colour;
    }

    fn luma(colour: f32x4) -> f32 {
        colour.xyz().dot(f32x3::new(0.2126, 0.7152, 0.0722))
    }

    #[compute(64)]
    fn grey(mut root: GpuMut<'static, Scene>, #[thread_id] id: u32x3) {
        let mut total = 0.0;
        for i in 0..4 {
            total += luma(root.pixels[id.x * 4 + i]);
        }
        root.pixels[id.x] = f32x4::splat(total * 0.25);
    }

    pub struct Varyings {
        pub position: f32x4,
        pub colour: f32x3,
    }

    #[vertex]
    fn vertex_main(root: Gpu<'static, Scene>, #[vertex_id] vid: u32) -> Varyings {
        let uv = f32x2::new((vid & 1) as f32, (vid >> 1) as f32);
        Varyings {
            position: f32x4::new(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0),
            colour: f32x3::new(uv.x, uv.y, root.tint.z),
        }
    }

    #[fragment]
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
    #[repr(C, align(16))]
    pub struct Transform {
        pub scene: Gpu<'static, Scene>,
        pub mvp: f32x4x4,
    }

    #[compute(64)]
    fn transform(root: Gpu<'static, Transform>, #[thread_id] id: u32x3) {
        let scene = root.scene;
        let mut out = scene.pixels;
        out[id.x] = root.mvp.mul(scene.tint);
    }
}

#[gpu_mod]
mod layered {
    use nga::*;

    pub struct VertexOut {
        pub position: f32x4,
        pub render_target_array_index: u32,
    }

    #[vertex]
    fn vertex(#[vertex_id] vid: u32) -> VertexOut {
        VertexOut {
            position: f32x4::new(0.0, 0.0, 0.0, 1.0),
            render_target_array_index: vid,
        }
    }
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

/// An inner attribute belongs to the module, not to the item under it.
#[gpu_mod]
mod inner_attribute {
    #![allow(clippy::needless_range_loop)]
    use nga::*;

    pub struct Value {
        pub x: f32,
    }

    #[compute(1)]
    fn double(mut root: GpuMut<'static, Value>) {
        root.x = root.x * 2.0;
    }
}

#[test]
fn an_inner_attribute_is_the_modules() {
    assert_eq!(inner_attribute::ENTRIES.len(), 1);
}

/// `#[host]` is the way out: the item passes through untouched, so a host-side type or helper
/// can live next to the shader it belongs to.
#[gpu_mod]
mod mixed {
    use nga::*;

    pub struct Params {
        pub scale: f32,
    }

    #[host]
    pub struct Buffers {
        pub staging: Vec<f32>,
    }

    #[host]
    pub fn fill(count: usize) -> Buffers {
        Buffers { staging: vec![0.0; count] }
    }

    #[compute(32)]
    fn scale(mut root: GpuMut<'static, Params>, #[thread_id] id: u32x3) {
        root.scale = root.scale * id.x as f32;
    }
}

/// What the three Metal-only examples do: a host module that shares a struct with a
/// hand-written `.metal` shader nests the struct in a `#[gpu_mod]` of its own, so the host code
/// around it is never asked to be shader code.
mod hand_written {
    use nga::*;

    #[gpu_mod]
    mod data {
        use nga::*;

        pub struct Corner {
            pub position: [f32; 3],
            pub uv: f32x2,
        }

        #[repr(C, align(16))]
        pub struct Camera {
            pub corners: Gpu<'static, Corner>,
            pub mvp: f32x4x4,
        }
    }

    use data::*;

    pub fn corner() -> Corner {
        Corner { position: [0.0; 3], uv: f32x2::splat(0.0) }
    }

    pub fn camera_size() -> usize {
        <Camera as DeviceAbi>::ABI.size
    }
}

#[test]
fn a_nested_gpu_mod_shares_structs_with_a_hand_written_shader() {
    assert_eq!(hand_written::corner().position.len(), 3);
    // The pointer, eight bytes of hole, then the matrix.
    assert_eq!(hand_written::camera_size(), 80);
}

#[test]
fn host_items_pass_through() {
    assert_eq!(mixed::fill(3).staging.len(), 3);
    assert_eq!(mixed::ENTRIES.len(), 1);
    // The shader half of the same module is still a shader.
    assert_eq!(<mixed::Params as DeviceAbi>::ABI.fields.len(), 1);
}

/// The entry list is derived from the stage attributes, so a shader cannot be left out of it.
#[test]
fn entries() {
    let names: Vec<&str> = shader::ENTRIES.iter().map(|f| f.name).collect();
    assert_eq!(names, ["clear", "grey", "vertex_main", "fragment_main", "transform"]);
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

#[test]
fn layered_vertex_output_is_a_backend_builtin() {
    let source = nga::msl::module(&[&layered::VERTEX]);
    assert!(
        source.contains("uint render_target_array_index [[render_target_array_index]]"),
        "Metal output is missing render_target_array_index:\n{source}"
    );

    let words = nga::spirv::words(&[&layered::VERTEX]);
    let mut capability = false;
    let mut builtin = false;
    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        let operands = &words[i + 1..i + count];
        capability |= opcode == 17 && operands.first() == Some(&69);
        builtin |= opcode == 71 && operands.len() >= 3 && operands[1..3] == [11, 9];
        i += count;
    }
    assert!(capability, "SPIR-V is missing ShaderLayer");
    assert!(builtin, "SPIR-V is missing BuiltIn Layer");
}

#[gpu_mod]
mod packed {
    use nga::*;

    pub struct Mesh {
        pub normal: packed_f32x3,
        pub count: u32,
        pub transform: f32x3x3,
    }

    #[compute(64)]
    fn packed_math(root: Gpu<'static, Mesh>, #[thread_id] id: u32x3) {
        let n = root.transform.mul(root.normal.unpack());
        let scale = n.length().max(1e-4);
        let mut acc = n.x * id.x as f32;
        for i in 0..root.count {
            acc += scale * i as f32;
        }
        let _keep = acc;
    }
}

#[test]
fn explicit_packed_vec3_and_native_matrix_are_msl() {
    assert_eq!(
        (
            size_of::<packed_f32x3>(),
            align_of::<packed_f32x3>(),
            size_of::<f32x3>(),
            align_of::<f32x3>(),
            size_of::<f32x3x3>(),
            align_of::<f32x3x3>(),
        ),
        (12, 4, 16, 16, 48, 16),
    );
    let types = ir::infer(&packed::PACKED_MATH);
    assert!(
        types.locals.iter().any(|ty| *ty == Ty::U32),
        "0..u32 must type the loop variable as u32, got {:?}",
        types.locals
    );

    let source = nga::msl::module(&[&packed::PACKED_MATH]);
    assert!(
        source.contains("packed_float3 normal"),
        "an explicit packed vector must stay twelve bytes:\n{source}"
    );
    assert!(
        source.contains("float3x3 transform"),
        "f32x3x3 must use Metal's native matrix ABI:\n{source}"
    );
    assert!(
        source.contains("static_assert(sizeof(Mesh) == "),
        "Mesh must assert its Rust size:\n{source}"
    );
    assert!(
        source.contains("float3(root->normal.x, root->normal.y, root->normal.z)"),
        "packed_float3 must unpack explicitly before arithmetic:\n{source}"
    );
    assert!(
        source.contains("0.0001f"),
        "1e-4 must emit as a float, not the integer 1:\n{source}"
    );
    assert!(
        source.contains("for (uint "),
        "a u32 range bound must emit an unsigned loop:\n{source}"
    );
}

#[test]
fn metal_compile_errors_drop_candidate_notes() {
    let source = "void main() {\n    return VertexOut { (m * n) };\n}\n";
    let log = "\
program_source:2:12: error: cannot convert between vector and non-scalar values ('metal::float3x3' and 'packed_float3')
    return VertexOut { (m * n) };
           ^
/System/Library/PrivateFrameworks/GPUCompiler.framework/include/metal/metal_matrix:2340:34: note: candidate template ignored: deduced conflicting types for parameter 'T'
METAL_FUNC matrix<T, Cols, Rows> operator*(const thread  matrix<T, Cols, Rows> &a, T b)
                                 ^
program_source:2:5: warning: comparison of integers of different signs: 'int' and 'uint'
 (domain: MTLLibraryErrorDomain, code: 3)";
    let formatted = nga::msl::format_compile_error(source, log);
    assert!(formatted.contains("cannot convert"));
    assert!(formatted.contains("return VertexOut"));
    assert!(formatted.contains("comparison of integers"));
    assert!(
        !formatted.contains("candidate template ignored"),
        "system-header candidate notes should be dropped:\n{formatted}"
    );
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
// `#[gpu_mod]` records what Rust chose and both emitters write those numbers out rather
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
