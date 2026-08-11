use crate::TypeAbi;

pub const MSL_PRELUDE: &str = r#"
#include <metal_stdlib>
using namespace metal;

[[maybe_unused]] constexpr sampler nga_nearest_clamp(filter::nearest, address::clamp_to_edge);
[[maybe_unused]] constexpr sampler nga_linear_clamp(filter::linear, mip_filter::linear, address::clamp_to_edge);
[[maybe_unused]] constexpr sampler nga_linear_repeat(filter::linear, mip_filter::linear, address::repeat);
[[maybe_unused]] constexpr sampler nga_aniso_repeat(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(16));

struct nga_texture2d { texture2d<float> t; };
struct nga_texture2d_array { texture2d_array<float> t; };
struct nga_texturecube { texturecube<float> t; };
struct nga_sampler { sampler s; };

static_assert(sizeof(nga_texture2d) == 8, "nga_texture2d is not one resource id");
static_assert(sizeof(nga_texture2d_array) == 8, "nga_texture2d_array is not one resource id");
static_assert(sizeof(nga_texturecube) == 8, "nga_texturecube is not one resource id");
static_assert(sizeof(nga_sampler) == 8, "nga_sampler is not one resource id");
"#;

#[macro_export]
macro_rules! msl_header {
    ($($ty:ty),* $(,)?) => {
        $crate::msl::header(&[$(<$ty as $crate::DeviceAbi>::ABI),*])
    };
}

pub use crate::msl_header as header;

pub fn header(types: &[TypeAbi]) -> String {
    let mut out = String::new();
    for ty in types {
        out += ty.decl;
    }
    for ty in types {
        let name = ty.name;
        out += "\n";
        out += &format!(
            "static_assert(sizeof({name}) == {}, \"{name} has the wrong size\");\n",
            ty.size
        );
        out += &format!(
            "static_assert(alignof({name}) == {}, \"{name} has the wrong alignment\");\n",
            ty.align
        );
        for (field, offset) in ty.fields {
            out += &format!(
                "static_assert(__builtin_offsetof({name}, {field}) == {offset}, \"{name}::{field} is at the wrong offset\");\n"
            );
        }
    }
    out
}

pub fn assert_matches(path: &str, contents: &str) {
    if std::fs::read_to_string(path).ok().as_deref() == Some(contents) {
        return;
    }
    if std::env::var_os("NGA_BLESS").is_some() {
        std::fs::write(path, contents).expect("could not write the generated MSL header");
        eprintln!("blessed {path}");
        return;
    }
    panic!("{path} no longer matches the Rust structs; re-run with NGA_BLESS=1 to update it");
}
