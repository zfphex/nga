//! Slang is the only shading language. `slangc` emits SPIR-V for the Vulkan backend and MSL
//! for the Metal one from the same source, so there is one prelude and one header emitter.

use crate::TypeAbi;

/// The prelude every shader is compiled against, and the only file that knows there are two
/// backends. `build.rs` defines `NGA_METAL` or `NGA_VULKAN`; nothing downstream forks again.
///
/// Two things the old MSL prelude needed are gone. The `constexpr sampler` set went, because a
/// sampler is now an ordinary heap slot on both backends. The hand-written one-member argument
/// buffer structs went too, because `DescriptorHandle<T>` says the same thing and slangc emits
/// the argument-buffer member itself.
///
/// What survives the fork is the handle *size*: eight bytes either way, which is the only
/// property the shared struct layout depends on.
pub const SLANG_PRELUDE: &str = include_str!("prelude.slang");

#[macro_export]
macro_rules! slang_header {
    ($($ty:ty),* $(,)?) => {
        $crate::slang::header(&[$(<$ty as $crate::DeviceAbi>::ABI),*])
    };
}

pub use crate::slang_header as header;

pub fn header(types: &[TypeAbi]) -> String {
    let mut out = String::new();
    for ty in types {
        out += &declare(ty);
    }
    out
}

/// The Rust layout is authoritative and the Slang one is made to match, never the other way
/// round.
///
/// Slang has no `offsetof` and no alignment attribute that reproduces Rust's `align(16)` on
/// `float4x4`, so alignment cannot be requested — but it can be made irrelevant. slangc's
/// natural layout packs every member (`packed_float3`, `array<packed_float4, 4>`), which means
/// inserting explicit `uint` padding is enough to land each field on the byte offset
/// `#[gpu_struct]` recorded from Rust. Verified on both targets: SPIR-V reports the same
/// `OpMemberDecorate Offset` values, and the Metal output keeps the padding members in place.
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
