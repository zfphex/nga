use std::path::PathBuf;

/// The same file `msl.rs` exposes as `msl::PRELUDE`, so a hand-written shader and a generated
/// one cannot disagree about what a handle or the root binding is.
const PRELUDE: &str = include_str!("src/prelude.metal");

/// What the hand-written shaders need on top of it. Mesh pipelines and indirect command
/// buffers have no `#[gpu_mod]` spelling, which is why those shaders are still MSL, and the
/// types
/// they encode through appear nowhere else.
const EXTRA: &str = r#"
#include <metal_mesh>

struct nga_commands { command_buffer cb; };
struct nga_pipeline { render_pipeline_state p; };
struct nga_exec_range { uint location; atomic_uint length; };

static_assert(sizeof(nga_commands) == 8, "nga_commands is not one resource id");
static_assert(sizeof(nga_pipeline) == 8, "nga_pipeline is not one resource id");
static_assert(sizeof(nga_exec_range) == 8, "nga_exec_range does not match MTLIndirectCommandBufferExecutionRange");
"#;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shaders = manifest.join("shaders");
    println!("cargo:rerun-if-changed={}", shaders.display());
    println!("cargo:rerun-if-changed=src/prelude.metal");

    // Everything under `shaders/` is a feature `#[gpu_mod]` is not asked to reach: tile shaders
    // and imageblocks have no Vulkan equivalent, and encoding an indirect command buffer from a
    // shader has no Rust spelling. They are copied through on Apple and ignored elsewhere,
    // which is what `#[cfg(target_vendor = "apple")]` does to the tests that use them.
    if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() != Ok("apple") {
        return;
    }

    let Ok(entries) = std::fs::read_dir(&shaders) else {
        return;
    };

    for entry in entries {
        let source = entry.unwrap().path();
        if source.extension().is_none_or(|ext| ext != "metal") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", source.display());
        let stem = source.file_stem().unwrap().to_str().unwrap();
        let body = std::fs::read_to_string(&source).unwrap();
        std::fs::write(
            out.join(format!("{stem}.metal")),
            format!("{PRELUDE}{EXTRA}\n{body}"),
        )
        .unwrap();
    }
}
