//! M4's exit criterion, headless: rasterise a triangle into a texture, read it back and check
//! the pixels. Per CLAUDE.md this is how rendering gets validated with an image file, not a
//! screenshot, and having it headless means it runs in CI and needs no window.

use nga::*;
use nga_shaders::triangle::{self, Data, Vertex};



const SIZE: u32 = 128;

#[test]
fn raster_draws_into_a_texture() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };

    let mut shaders = Shaders::new(&device, &nga::compile(triangle::ENTRIES))?;
    let pipeline = shaders.raster(&RasterDesc::new(
        "vertex",
        "fragment",
        &[ColorTarget::new(Format::Rgba8Unorm)],
    ))?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut heap = GpuHeap::new(&device, 16 * 1024 * 1024)?;
    let target = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE)
            .usage(Usage::COLOR | Usage::COPY_SRC),
    )?;

    // A triangle covering the middle of the frame, white, so the coverage test does not
    // depend on how the three vertex colours interpolate.
    let white = f32x3([1.0, 1.0, 1.0]);
    let mesh = Arena::new(&device, 64 * 1024, Memory::Device)?;
    let vertices = mesh
        .alloc_slice(&[
            Vertex {
                position: f32x3([0.0, 0.8, 0.0]),
                color: white,
            },
            Vertex {
                position: f32x3([-0.8, -0.8, 0.0]),
                color: white,
            },
            Vertex {
                position: f32x3([0.8, -0.8, 0.0]),
                color: white,
            },
        ])
        .unwrap();
    let vertices: Gpu<'static, [Vertex]> = Gpu::new(vertices.gpu.0.get());

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
        raster.draw(root, 0, 3);
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
        luma(SIZE - 3, 2),
        0,
        "the top-right corner is outside the triangle"
    );
    assert_eq!(
        luma(SIZE / 2, SIZE / 2),
        255,
        "the centre should be covered by the triangle"
    );

    let covered = bytes.chunks_exact(4).filter(|p| p[0] > 128).count();
    let total = (SIZE * SIZE) as usize;
    // The triangle spans 80% of the frame in each axis, so it covers ~32% of it.
    assert!(
        (total / 4..total / 2).contains(&covered),
        "{covered} of {total} pixels covered; that is not the triangle"
    );

    heap.free_texture(target);
    Ok(())
}
