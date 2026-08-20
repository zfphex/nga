//! An indirect command buffer, headless. A compute shader encodes three draws, picking the
//! pipeline and the per-draw root pointer itself, and a render pass replays them.
//!
//! `shaders/icb.metal` is MSL for the same reason `shaders/mesh.metal` is; see its header.

#![cfg(target_vendor = "apple")]

use nga::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

/// Written by the encoding shader, one per command. The host only sizes the buffer.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DrawData {
    vertices: u64,
    offset: [f32; 2],
    _pad: [u32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EncodeData {
    vertices: Gpu<'static, Vertex>,
    draws: GpuMut<'static, DrawData>,
    commands: Commands,
    pipeline: PipelineHandle,
    count: u32,
}

const SHADERS: &[u8] = nga::shader!("icb");

const SIZE: u32 = 128;
const DRAWS: u32 = 3;

#[test]
fn a_shader_encodes_the_draws_it_replays() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };
    if !device.supports(Feature::IndirectCommands) {
        eprintln!(
            "{} has no indirect command buffers; skipping",
            device.name()
        );
        return Ok(());
    }

    let mut shaders = Shaders::new(&device, SHADERS)?;
    // `indirect` is what gives the pipeline a resource ID for the shader to name.
    let drawn = shaders.raster(
        &RasterDesc::new(
            "icb_vertex",
            "icb_fragment",
            &[ColorTarget::new(Format::Rgba8Unorm)],
        )
        .indirect(),
    )?;
    let encode = shaders.compute("encode", None)?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let commands = IndirectCommands::draws(&device, DRAWS, false)?;

    let mut heap = GpuHeap::new(&device, 16 * 1024 * 1024)?;
    let target = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE)
            .usage(Usage::COLOR | Usage::COPY_SRC),
    )?;
    let mut draws = heap.alloc_buffer(size_of::<DrawData>() * DRAWS as usize)?;
    let draw_block: GpuMut<'static, DrawData> = GpuMut::new(draws.gpu_mut::<DrawData>().0.get());

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
        .alloc(EncodeData {
            vertices,
            draws: draw_block,
            commands: commands.handle(),
            pipeline: drawn.handle(),
            count: DRAWS,
        })
        .unwrap();
    let root = data.gpu;

    let mut recorder = frame.recorder();
    {
        let compute = recorder.compute();
        // An unreset slot replays whatever it last held, so clearing comes first.
        compute.reset(&commands, commands.all());
        compute.pipeline(encode);
        compute.dispatch(root, DRAWS as usize, 64);
        compute.barrier(Stage::COMPUTE, Stage::INDIRECT, Hazard::DRAW_ARGUMENTS);
    }
    {
        let pass = Pass::new().color_texture(0, &target, Some([0.0, 0.0, 0.0, 1.0]));
        let raster = recorder.raster(&pass, SIZE, SIZE);
        raster.execute(&commands, commands.all());
    }
    {
        let compute = recorder.compute();
        compute.barrier(Stage::COLOR_OUT, Stage::TRANSFER, Hazard::TEXTURES);
        compute.read_texture(&target, &readback, dest);
    }
    drop(recorder);
    frame.submit(None)?;
    device.wait_idle()?;

    // Each triangle is a quarter the size of the one `raster.rs` draws, about 2%% of the frame,
    // and there are three of them, spread across the width. A backend that replays nothing
    // leaves the clear behind; one that replays garbage fills the frame.
    let bytes: &[u8] = &pixels;
    let total = (SIZE * SIZE) as usize;
    let covered = bytes.chunks_exact(4).filter(|p| p[0] > 128).count();
    assert!(
        (total / 100..total / 4).contains(&covered),
        "{covered} of {total} pixels drawn; that is not three small triangles"
    );

    // One column per encoded draw, at the offsets the shader wrote.
    for slot in 0..DRAWS {
        let x = ((-0.6 + 0.4 * slot as f32) * 0.5 + 0.5) * SIZE as f32;
        let column = (0..SIZE)
            .filter(|&y| bytes[((y * SIZE + x as u32) * 4) as usize] > 128)
            .count();
        assert!(column > 0, "draw {slot} left no pixels at x = {x}");
    }

    heap.free_texture(target);
    heap.free_buffer(draws);
    Ok(())
}
