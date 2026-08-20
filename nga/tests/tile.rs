//! A tile pipeline, headless: draw the triangle `raster.rs` draws, then run a compute function
//! over the imageblock before it is ever stored. The centre pixel is white in `raster.rs` and
//! half that here, which is the whole claim.

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

const SHADERS: &[u8] = nga::shader!("tile");

const SIZE: u32 = 128;
/// One `TileColor`: a `half4`.
const IMAGEBLOCK_BYTES: usize = 8;
const TILE: u32 = 16;

#[test]
fn a_tile_pipeline_edits_the_imageblock() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };
    if !device.supports(Feature::TileShaders) {
        eprintln!("{} has no tile pipeline; skipping", device.name());
        return Ok(());
    }

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let draw = shaders.raster(&RasterDesc::new(
        "tile_vertex",
        "tile_fragment",
        &[ColorTarget::new(Format::Rgba8Unorm)],
    ))?;
    let halve = shaders.tile(&TileDesc::new("tile_halve", &[Format::Rgba8Unorm]))?;
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
        let pass = Pass::new()
            .color_texture(0, &target, Some([0.0, 0.0, 0.0, 1.0]))
            .tile(TILE, TILE, IMAGEBLOCK_BYTES);
        let raster = recorder.raster(&pass, SIZE, SIZE);
        raster.pipeline(draw);
        raster.draw(root, 0, 3);
        raster.pipeline(halve);
        raster.dispatch_tile(root, TILE, TILE);
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
    assert_eq!(
        luma(2, 2),
        0,
        "the corner was black and half of black is black"
    );
    let centre = luma(SIZE / 2, SIZE / 2);
    assert!(
        (120..=136).contains(&centre),
        "the centre is {centre}; the tile pass should have halved 255"
    );

    heap.free_texture(target);
    Ok(())
}
