//! MSAA: rasterise at four samples into a multisample attachment, resolve into a single-sample
//! texture and read that back. The proof is the edge — a resolved edge has partial coverage in
//! it, a single-sample edge is only ever 0 or 255.

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

const SHADERS: &[u8] = nga::shader!("triangle");

const SIZE: u32 = 128;
const IDENTITY: f32x4x4 = f32x4x4([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]);

/// Draws the triangle at `samples` and returns the frame, resolving first when `samples > 1`.
fn render(samples: u32) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let device = Device::new(cfg!(debug_assertions))?;

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let pipeline = shaders.raster(
        &RasterDesc::new(
            "vertex_main",
            "fragment_main",
            &[ColorTarget::new(Format::Rgba8Unorm)],
        )
        .depth(Format::D32Float)
        .samples(samples),
    )?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut heap = GpuHeap::new(&device, 16 * 1024 * 1024)?;
    // What the pass draws into, and what it ends up in. At one sample they are the same
    // texture and there is no resolve.
    let resolved = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE)
            .usage(Usage::COLOR | Usage::COPY_SRC),
    )?;
    let multisample = match samples {
        1 => None,
        _ => Some(heap.alloc_texture(
            &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE).samples(samples),
        )?),
    };
    // Depth is here for its resolve, which is the half colour cannot exercise: colour averages
    // and depth has to be told which sample to keep.
    let depth =
        heap.alloc_texture(&TextureDesc::depth(Format::D32Float, SIZE, SIZE).samples(samples))?;
    let depth_resolved = match samples {
        1 => None,
        _ => Some(heap.alloc_texture(&TextureDesc::depth(Format::D32Float, SIZE, SIZE))?),
    };

    // A slanted edge, so the resolve has something to average. White, so coverage is the only
    // thing the pixel values can be measuring.
    let white = [1.0, 1.0, 1.0];
    let mesh = Arena::new(&device, 64 * 1024, Memory::Device)?;
    let vertices = mesh
        .alloc_slice(&[
            Vertex {
                position: [-0.9, 0.8, 0.0],
                color: white,
            },
            Vertex {
                position: [-0.9, -0.8, 0.0],
                color: white,
            },
            Vertex {
                position: [0.7, -0.8, 0.0],
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
    let root = frame
        .arena(0)
        .alloc(Data {
            vertices,
            mvp: IDENTITY,
        })
        .unwrap()
        .gpu;

    let mut recorder = frame.recorder();
    {
        let mut pass = match &multisample {
            Some(target) => Pass::new()
                .color_texture(0, target, Some([0.0, 0.0, 0.0, 1.0]))
                .resolve_texture(0, &resolved),
            None => Pass::new().color_texture(0, &resolved, Some([0.0, 0.0, 0.0, 1.0])),
        }
        .depth(&depth, Some(1.0), false);
        if let Some(target) = &depth_resolved {
            pass = pass.depth_resolve(target, ResolveMode::SampleZero);
        }
        let raster = recorder.raster(&pass, SIZE, SIZE);
        raster.pipeline(pipeline);
        raster.depth(&DepthState::new(CompareOp::Less, true));
        raster.draw(root, 0, 3);
    }
    {
        let compute = recorder.compute();
        compute.barrier(Stage::COLOR_OUT, Stage::TRANSFER, Hazard::TEXTURES);
        compute.read_texture(&resolved, &readback, dest);
    }
    drop(recorder);
    frame.submit(None)?;
    device.wait_idle()?;

    let bytes = pixels.to_vec();
    if let Some(target) = depth_resolved {
        heap.free_texture(target);
    }
    heap.free_texture(depth);
    if let Some(target) = multisample {
        heap.free_texture(target);
    }
    heap.free_texture(resolved);
    Ok(bytes)
}

#[test]
fn a_resolved_edge_has_partial_coverage() -> Result<(), Box<dyn std::error::Error>> {
    if Device::new(false).is_err() {
        eprintln!("no supported device; skipping");
        return Ok(());
    }

    let partial = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .filter(|p| (16..240).contains(&p[0]))
            .count()
    };

    let aliased = render(1)?;
    let resolved = render(4)?;

    assert_eq!(
        partial(&aliased),
        0,
        "a single-sample edge cannot be partially covered"
    );
    // The hypotenuse crosses ~128 pixels, and four samples give it three intermediate levels.
    assert!(
        partial(&resolved) > 50,
        "{} partially covered pixels is not a resolved edge",
        partial(&resolved)
    );
    // The resolve must not have changed what is covered, only how the edge reads.
    let covered = |bytes: &[u8]| bytes.chunks_exact(4).filter(|p| p[0] > 128).count();
    let (a, b) = (covered(&aliased), covered(&resolved));
    assert!(
        a.abs_diff(b) < (SIZE * SIZE) as usize / 50,
        "{a} covered at one sample against {b} at four; that is a different triangle"
    );
    Ok(())
}
