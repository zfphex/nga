//! A mesh pipeline drawing the same triangle `raster.rs` draws through the vertex path, so the
//! two can be compared pixel for pixel. Headless, and skipped where the backend has no mesh
//! pipeline — which is what `Feature::MeshShaders` is for.

// The shader is MSL — see `shaders/mesh.metal` for why — so there is nothing to include on a
// backend that wants SPIR-V. The `Feature::MeshShaders` check below is the runtime half.
#![cfg(target_vendor = "apple")]

use nga::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Data {
    vertices: Gpu<'static, Vertex>,
    mvp: f32x4x4,
}

const SHADERS: &[u8] = nga::shader!("mesh");

const SIZE: u32 = 128;

#[test]
fn a_mesh_pipeline_emits_a_triangle() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };
    if !device.supports(Feature::MeshShaders) {
        eprintln!("{} has no mesh pipeline; skipping", device.name());
        return Ok(());
    }

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let pipeline = shaders.mesh(&MeshDesc::new(
        "",
        "mesh_main",
        "mesh_fragment",
        &[ColorTarget::new(Format::Rgba8Unorm)],
        0,
    ))?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut heap = GpuHeap::new(&device, 16 * 1024 * 1024)?;
    let target = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE)
            .usage(Usage::COLOR | Usage::COPY_SRC),
    )?;

    let white = [1.0, 1.0, 1.0];
    let mesh = Arena::new(&device, 64 * 1024, Memory::Device)?;
    let vertices = mesh
        .alloc_slice(&[
            Vertex {
                position: [0.0, 0.8, 0.0],
                color: white,
            },
            Vertex {
                position: [-0.8, -0.8, 0.0],
                color: white,
            },
            Vertex {
                position: [0.8, -0.8, 0.0],
                color: white,
            },
        ])
        .unwrap();
    let vertices: Gpu<'static, Vertex> = Gpu::new(vertices.gpu.0.get());

    let readback = Arena::new(&device, 1024 * 1024, Memory::Readback)?;
    let pixels = readback
        .alloc_scratch::<u8>((SIZE * SIZE * 4) as usize)
        .unwrap();
    let dest: GpuMut<'_, [u8]> = GpuMut::new(pixels.gpu.0.get());

    let mut frame = ctx.begin_frame()?;
    let data = frame
        .arena(0)
        .alloc(Data {
            vertices,
            mvp: f32x4x4([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        })
        .unwrap();
    let root = data.gpu;

    let mut recorder = frame.recorder();
    {
        let pass = Pass::new().color_texture(0, &target, Some([0.0, 0.0, 0.0, 1.0]));
        let raster = recorder.raster(&pass, SIZE, SIZE);
        raster.pipeline(pipeline);
        // One mesh threadgroup of three threads, one thread per vertex. No object stage, so the
        // object threadgroup size is unused.
        raster.draw_mesh(root, [1, 1, 1], [1, 1, 1], [3, 1, 1]);
    }
    {
        let compute = recorder.compute();
        compute.barrier(Stage::COLOR_OUT, Stage::TRANSFER, Hazard::TEXTURES);
        compute.read_texture(&target, &readback, dest);
    }
    drop(recorder);
    frame.submit(None)?;
    device.wait_idle()?;

    let bytes: &[u8] = &pixels;
    let luma = |x: u32, y: u32| bytes[((y * SIZE + x) * 4) as usize];
    assert_eq!(luma(2, 2), 0, "the top-left corner is outside the triangle");
    assert_eq!(
        luma(SIZE / 2, SIZE / 2),
        255,
        "the centre should be covered by the triangle"
    );

    // The same window `raster.rs` asserts, because it is the same triangle.
    let covered = bytes.chunks_exact(4).filter(|p| p[0] > 128).count();
    let total = (SIZE * SIZE) as usize;
    assert!(
        (total / 4..total / 2).contains(&covered),
        "{covered} of {total} pixels covered; that is not the triangle"
    );

    heap.free_texture(target);
    Ok(())
}
