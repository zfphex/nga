use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shaders = manifest.join("shaders");
    println!("cargo:rerun-if-changed={}", shaders.display());

    let slangc = match std::env::var("VULKAN_SDK") {
        Ok(sdk) => PathBuf::from(sdk).join("Bin").join("slangc.exe"),
        Err(_) => PathBuf::from("slangc"),
    };

    for entry in std::fs::read_dir(&shaders).expect("nga-vk/shaders is missing") {
        let source = entry.unwrap().path();
        if source.extension().is_none_or(|ext| ext != "slang") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", source.display());
        let stem = source.file_stem().unwrap().to_str().unwrap();
        let spirv = out.join(format!("{stem}.spv"));

        // Scalar layout is what makes `float3` occupy 12 bytes, matching MSL `packed_float3`
        // and Rust `[f32; 3]`. Without it every three-component vector silently grows to 16.
        let output = Command::new(&slangc)
            .arg(&source)
            .args(["-target", "spirv"])
            .args(["-profile", "spirv_1_5"])
            .arg("-emit-spirv-directly")
            .arg("-fvk-use-scalar-layout")
            .arg("-o")
            .arg(&spirv)
            .args([
                "-reflection-json",
                out.join(format!("{stem}.json")).to_str().unwrap(),
            ])
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
