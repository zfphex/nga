//! An indirect command buffer, headless. A compute shader encodes three draws, picking the
//! pipeline and the per-draw root pointer itself, and a render pass replays them.

use nga::*;
#[cfg(target_vendor = "apple")]
use shader::{DrawData, EncodeData, Vertex};

#[gpu_mod]
mod shader {
    use nga::*;

    pub struct Vertex {
        pub position: [f32; 3],
        pub color: [f32; 3],
    }

    /// Written by the encoding shader, one per command. The host only sizes the buffer.
    pub struct DrawData {
        pub vertices: Gpu<[Vertex]>,
        pub offset: f32x2,
        pub color: f32x4,
    }

    pub struct EncodeData {
        pub vertices: Gpu<[Vertex]>,
        pub draws: GpuRw<[DrawData]>,
        pub commands: Commands,
        pub pipeline: PipelineHandle,
        pub count: u32,
    }

    pub struct VertexOut {
        pub position: f32x4,
        pub color: f32x4,
    }

    #[compute(64)]
    fn encode(root: Gpu<EncodeData>, #[thread_id] i: u32x3) {
        if i.x >= root.count {
            return;
        }
        let mut draws = root.draws;
        draws[i.x] = DrawData {
            vertices: root.vertices,
            offset: f32x2::new(-0.6 + 0.4 * i.x as f32, 0.0),
            color: f32x4::new(1.0, 1.0, 1.0, 1.0),
        };
        command_set_pipeline(root.commands, i.x, root.pipeline);
        command_set_vertex_buffer(root.commands, i.x, draws[i.x], 1);
        command_draw(root.commands, i.x, false, 0, 3, 1, 0);
    }

    #[vertex]
    fn icb_vertex(#[buffer(1)] data: Gpu<DrawData>, #[vertex_id] vid: u32) -> VertexOut {
        let source = data.vertices[vid];
        VertexOut {
            position: f32x4::new(
                source.position[0] * 0.25 + data.offset.x,
                source.position[1] * 0.25 + data.offset.y,
                source.position[2],
                1.0,
            ),
            color: data.color,
        }
    }

    #[fragment]
    fn icb_fragment(input: VertexOut) -> f32x4 {
        input.color
    }
}

#[gpu_mod]
mod counted {
    use nga::*;

    pub struct EncodeData {
        pub commands: Commands,
    }

    #[compute(1)]
    fn encode(root: Gpu<EncodeData>) {
        command_draw(root.commands, 0, false, 0, 3, 1, 0);
    }
}

#[cfg(target_vendor = "apple")]
const SIZE: u32 = 128;
#[cfg(target_vendor = "apple")]
const DRAWS: u32 = 3;

#[test]
fn icb_shader_is_generated_from_rust() {
    let source = msl::module(shader::ENTRIES);
    assert!(source.contains("struct nga_commands { command_buffer cb; };"));
    assert!(source.contains("nga_command_set_pipeline("));
    assert!(source.contains("nga_command_set_vertex_buffer("));
    assert!(source.contains("nga_command_draw("));
    assert!(source.contains("device const DrawData* data [[buffer(1)]]"));
}

#[test]
fn inherited_counted_draw_is_metal_work_and_a_vulkan_no_op() {
    let source = msl::module(&[&counted::ENCODE]);
    assert!(source.contains("nga_command_draw("));

    let words = spirv::words(&[&counted::ENCODE]);
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    assert!(
        bytes
            .windows(b"encode\0".len())
            .any(|name| name == b"encode\0"),
        "the Vulkan entry point was dropped instead of treating command_draw as a no-op"
    );
}

#[cfg(target_vendor = "apple")]
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

    let mut shaders = device.shaders(shader::MODULE)?;
    let drawn = shaders
        .raster(shader::ICB_VERTEX, shader::ICB_FRAGMENT)
        .color(Format::Rgba8Unorm)
        .indirect()
        .build()?;
    let encode = shaders.compute(shader::ENCODE)?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let commands = IndirectCommands::draws(&device, DRAWS, false)?;

    let mut heap = GpuHeap::new(&device, 16 * 1024 * 1024)?;
    let target = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE)
            .usage(Usage::COLOR | Usage::COPY_SRC),
    )?;
    let draws = heap.alloc_buffer(size_of::<DrawData>() * DRAWS as usize)?;
    let draw_block: GpuRw<[DrawData]> = draws.gpu_rw();

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
    let vertices: Gpu<[Vertex]> = vertices.gpu;

    let readback = Arena::new(&device, 1024 * 1024, Memory::Readback)?;
    // The render and copy below initialise every byte before the host reads it.
    let pixels = unsafe {
        readback
            .alloc_scratch::<u8>((SIZE * SIZE * 4) as usize)
            .unwrap()
    };
    let dest: GpuRw<[u8]> = pixels.gpu.as_rw();

    let mut frame = ctx.begin_frame()?;
    let mut recorder = frame.recorder();
    {
        let compute = recorder.compute();
        // An unreset slot replays whatever it last held, so clearing comes first.
        compute.reset(&commands, commands.all());
        compute.dispatch(
            encode,
            EncodeData {
                vertices,
                draws: draw_block,
                commands: commands.handle(),
                pipeline: drawn.handle(),
                count: DRAWS,
            },
            Dispatch::x(DRAWS as usize, 64),
        )?;
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
