use crate::TypeAbi;

/// The Metal port ships an MSL prelude full of `nga_texture2d`-style one-member argument-buffer
/// structs, because MSL has no way to spell a bare `MTLResourceID`. Vulkan needs none of that:
/// a bindless handle is a `uint` index and the heaps are ordinary descriptor arrays.
pub const SLANG_PRELUDE: &str = r#"
// Metal needs one-member argument-buffer structs here because MSL cannot spell a bare
// MTLResourceID. Vulkan needs none of that: a handle is a `uint` index into a descriptor
// array, and the arrays are laid out by VK_EXT_descriptor_buffer. Metal's single heap
// becomes three, because sampler descriptors cannot share a buffer with image descriptors.
[[vk::binding(0, 0)]] Texture2D<float4> nga_textures[];
[[vk::binding(0, 1)]] RWTexture2D<float4> nga_storage[];
[[vk::binding(0, 2)]] SamplerState nga_samplers[];

struct nga_push {
    uint64_t root;
    uint resources;
    uint samplers;
};

[[vk::push_constant]] ConstantBuffer<nga_push> push;

struct nga_draw { uint vertex_count; uint instance_count; uint vertex_start; uint base_instance; };
struct nga_draw_indexed { uint index_count; uint instance_count; uint index_start; int base_vertex; uint base_instance; };
struct nga_dispatch { uint threadgroups[3]; };
struct nga_dispatch_threads { uint threads[3]; uint group[3]; };
"#;

#[macro_export]
macro_rules! slang_header {
    ($($ty:ty),* $(,)?) => {
        $crate::slang::header(&[$(<$ty as $crate::DeviceAbi>::ABI),*])
    };
}

pub use crate::slang_header as header;

/// Slang has no `offsetof` and, under scalar layout, no `float4x4` alignment rule matching
/// Rust's `align(16)`. The Rust layout is therefore authoritative: this walks the recorded
/// offsets and inserts explicit padding so the Slang struct lands on the same bytes.
pub fn header(types: &[TypeAbi]) -> String {
    let mut out = String::new();
    for ty in types {
        out += &declare(ty);
    }
    out
}

pub fn declare(ty: &TypeAbi) -> String {
    let body = ty.decl.trim_start_matches('\n');
    let open = body
        .find('{')
        .expect("a gpu_struct decl always has a brace");
    let mut out = String::from(&body[..=open]);
    out.push('\n');

    let mut cursor = 0usize;
    let mut pad = 0usize;
    for (line, (name, offset, size)) in body[open + 1..]
        .lines()
        .filter(|line| line.contains(';'))
        .zip(ty.fields)
    {
        if *offset > cursor {
            out += &padding(*offset - cursor, &mut pad);
        }
        debug_assert!(
            line.contains(name),
            "decl line `{line}` does not declare `{name}`"
        );
        out += line;
        out.push('\n');
        cursor = offset + size;
    }
    if ty.size > cursor {
        out += &padding(ty.size - cursor, &mut pad);
    }
    out += "};\n";
    out
}

fn padding(bytes: usize, index: &mut usize) -> String {
    let name = format!("_nga_pad{index}");
    *index += 1;
    match bytes % 4 {
        0 => format!("    uint {name}[{}];\n", bytes / 4),
        _ => format!("    uint8_t {name}[{bytes}];\n"),
    }
}

pub fn dump(name: &str, contents: &str) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../target/generated");
    std::fs::create_dir_all(dir).expect("could not create the generated Slang directory");
    std::fs::write(format!("{dir}/{name}.slang"), contents)
        .expect("could not write the generated Slang header");
}
