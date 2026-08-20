use std::path::PathBuf;

/// The same file `slang.rs` exposes as `SLANG_PRELUDE`, so the two cannot drift apart.
const PRELUDE: &str = include_str!("src/prelude.slang");
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shaders = manifest.join("shaders");
    println!("cargo:rerun-if-changed={}", shaders.display());

    let apple = std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple");

    let slangc = match std::env::var("VULKAN_SDK") {
        Ok(sdk) if !apple => PathBuf::from(sdk).join("Bin").join("slangc.exe"),
        _ => PathBuf::from("slangc"),
    };

    let Ok(entries) = std::fs::read_dir(&shaders) else {
        return;
    };

    for entry in entries {
        let source = entry.unwrap().path();
        if source.extension().is_none_or(|ext| ext != "slang") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", source.display());
        let stem = source.file_stem().unwrap().to_str().unwrap();

        // Every shader is compiled against the prelude, which is the only place that knows
        // there are two backends. Prepending beats an include path: there is no file to find.
        let combined = out.join(format!("{stem}.prelude.slang"));
        std::fs::write(
            &combined,
            format!("{PRELUDE}
{}", std::fs::read_to_string(&source).unwrap()),
        )
        .unwrap();
        let source = combined;

        // Only the host platform's target is built. Both come from this one source, which is
        // the whole point of the crate merge.
        // The Metal source is emitted on every host, not just macOS. slangc needs no Apple
        // toolchain to produce it, and it is the only way to check the Metal-side struct
        // layout from a machine that cannot compile `metal.rs` at all — see tests/layout.rs.
        let metal = Command::new(&slangc)
            .arg(&source)
            .arg("-DNGA_METAL=1")
            .args(["-target", "metal"])
            .arg("-o")
            .arg(out.join(format!("{stem}.metal")))
            .output()
            .unwrap_or_else(|e| panic!("could not run {}: {e}", slangc.display()));
        if !metal.status.success() {
            panic!(
                "slangc metal target failed on {}:
{}{}",
                source.display(),
                String::from_utf8_lossy(&metal.stdout),
                String::from_utf8_lossy(&metal.stderr)
            );
        }

        let mut command = Command::new(&slangc);
        command.arg(&source);
        // Slang predefines no target macro of its own, so the prelude gets one from here.
        // It is the only thing standing between one shader source and two binding models.
        command.arg(if apple { "-DNGA_METAL=1" } else { "-DNGA_VULKAN=1" });
        if apple {
            command.args(["-target", "metal"]);
            command.arg("-o").arg(out.join(format!("{stem}.metal")));
        } else {
            // Scalar layout is what makes `float3` occupy 12 bytes, matching the
            // `packed_float3` slangc emits for the Metal target and Rust's `[f32; 3]`.
            // Without it every three-component vector silently grows to 16.
            command.args(["-target", "spirv"]);
            command.args(["-profile", "spirv_1_5"]);
            command.arg("-emit-spirv-directly");
            command.arg("-fvk-use-scalar-layout");
            // Without this slangc renames a module's only entry point to `main`, so a
            // single-kernel shader stops matching the name the host asks for.
            command.arg("-fvk-use-entrypoint-name");
            command.args(["-bindless-space-index", "0"]);
            command.arg("-o").arg(out.join(format!("{stem}.spv")));
        }
        command.args([
            "-reflection-json",
            out.join(format!("{stem}.json")).to_str().unwrap(),
        ]);

        let output = command
            .output()
            .unwrap_or_else(|e| panic!("could not run {}: {e}", slangc.display()));

        if !output.status.success() {
            panic!(
                "slangc failed on {}:\n{}{}",
                source.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
