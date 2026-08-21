//! Every shader in the crate, through both backends. The SPIR-V is handed to `spirv-val`; the
//! MSL cannot be compiled from a non-Apple host, so what is checked there is that it is emitted
//! at all and names its entry points.

use nga_shaders::*;

fn modules() -> Vec<(&'static str, &'static [&'static nga::ir::GpuFn])> {
    vec![
        ("fill", fill::ENTRIES),
        ("gradient", gradient::ENTRIES),
        ("sample", sample::ENTRIES),
        ("material", material::ENTRIES),
        ("triangle", triangle::ENTRIES),
        ("raster", raster::ENTRIES),
        ("sync", sync::ENTRIES),
        ("bindless", bindless::ENTRIES),
        ("gpu_driven", gpu_driven::ENTRIES),
    ]
}

#[test]
fn every_module_validates() {
    let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../target/generated"));
    let validator = std::env::var("VULKAN_SDK")
        .map(|sdk| std::path::PathBuf::from(sdk).join("Bin").join("spirv-val"));

    for (name, entries) in modules() {
        let (spirv, metal) = nga::dump(&dir, name, entries).unwrap();
        let source = std::fs::read_to_string(&metal).unwrap();
        for entry in entries {
            assert!(
                source.contains(&format!(" {}(", entry.name)),
                "{name}: the MSL has no `{}`",
                entry.name
            );
        }

        let Ok(validator) = &validator else {
            continue;
        };
        let Ok(output) = std::process::Command::new(validator)
            .args(["--target-env", "vulkan1.3", "--scalar-block-layout"])
            .arg(&spirv)
            .output()
        else {
            continue;
        };
        assert!(
            output.status.success(),
            "{name}: spirv-val rejected the module:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
