#![cfg(target_vendor = "apple")]

use nga::*;

#[gpu_mod]
mod shader {
    use nga::*;

    pub struct Paint {
        pub target: Texture2dRW,
    }

    #[compute(1)]
    fn paint(root: Gpu<Paint>) {
        root.target
            .offset(1)
            .store(u32x2::new(0, 0), f32x4::new(0.0, 1.0, 0.0, 1.0));
    }

    pub struct RenderData {
        pub positions: Gpu<[f32x4]>,
    }

    pub struct VertexOut {
        pub position: f32x4,
    }

    #[vertex]
    fn vertex(root: Gpu<RenderData>, #[vertex_id] index: u32) -> VertexOut {
        VertexOut {
            position: root.positions[index],
        }
    }

    #[fragment]
    fn fragment(_input: VertexOut) -> f32x4 {
        f32x4::new(0.0, 1.0, 0.0, 1.0)
    }

    pub struct CopyData {
        pub positions: Gpu<[f32x4]>,
        pub source: Texture2d,
        pub sampler: Sampler,
    }

    fn sample_source(source: Texture2d, sampler: Sampler, uv: f32x2) -> f32x4 {
        source.sample(sampler, uv)
    }

    fn sample_copy(root: Gpu<CopyData>, uv: f32x2) -> f32x4 {
        sample_source(root.source, root.sampler, uv)
    }

    #[fragment]
    fn copy(root: Gpu<CopyData>, input: VertexOut) -> f32x4 {
        sample_copy(root, input.position.xy() / 4.0)
    }
}

#[test]
#[ignore = "requires Metal 4"]
fn indexed_draws_resource_ids_and_timestamp_reuse() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(true)?;
    let mut shaders = device.shaders(shader::MODULE)?;
    let draw = shaders
        .raster(shader::VERTEX, shader::FRAGMENT)
        .color(Format::Rgba8Unorm)
        .build()?;
    let copy = shaders
        .raster(shader::VERTEX, shader::COPY)
        .color(Format::Rgba8Unorm)
        .build()?;
    let paint = shaders.compute(shader::PAINT)?;
    let heap = GpuHeap::new(&device, 1024 * 1024)?;
    let target = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, 4, 4)
            .usage(Usage::COLOR | Usage::STORAGE | Usage::SAMPLED | Usage::COPY_SRC),
    )?;
    let other = heap
        .alloc_texture(&TextureDesc::texture_2d(Format::Rgba8Unorm, 4, 4).usage(Usage::STORAGE))?;
    let positions = heap.alloc_slice(&[
        f32x4::new(0.0, 0.0, 0.0, 0.0),
        f32x4::new(-1.0, -1.0, 0.0, 1.0),
        f32x4::new(3.0, -1.0, 0.0, 1.0),
        f32x4::new(-1.0, 3.0, 0.0, 1.0),
    ])?;
    let indices = heap.alloc_slice(&[0u16, 1, 2, 2, 3, 4])?;
    let mut resources = ResourceHeap::new(&device, 4)?;
    let range = resources.alloc_range(4)?;
    let texture = resources.write_storage(range, 2, &other);
    resources.write_storage(range, 3, &target);
    let copied = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, 4, 4).usage(Usage::COLOR | Usage::COPY_SRC),
    )?;
    let source = resources.write_texture(range, 0, &target);
    let sampler_state = device.sampler(&SamplerDesc::nearest())?;
    let sampler = resources.write_sampler(range, 1, &sampler_state);
    let readback = Arena::new(&device, 4096, Memory::Readback)?;
    let pixels = readback.alloc_slice(&[0u8; 64]).unwrap();
    let mut context = Context::new(&device, &shaders, 4096, 1)?;
    let mut timers = Timers::new(&device, 2)?;

    for frame_index in 0..(FRAMES_IN_FLIGHT * 3) {
        let mut frame = context.begin_frame()?;
        timers.begin(&frame);
        if frame_index >= FRAMES_IN_FLIGHT {
            let expected = frame_index < FRAMES_IN_FLIGHT * 2;
            assert_eq!(timers.written(0), expected);
            assert_eq!(timers.written(1), expected);
            assert!(timers.ns(0, 1).is_finite());
        }
        let recorder = frame.recorder();
        if frame_index < FRAMES_IN_FLIGHT {
            recorder.compute().timestamp(&timers, 0);
        }
        let pass = Pass::new().color_texture(0, &target, Some([1.0, 0.0, 0.0, 1.0]));
        // Both signs must select the same three valid vertices.
        let (first_index, vertex_offset) = if frame_index % 2 == 0 {
            (0, 1)
        } else {
            (3, -1)
        };
        recorder.raster(&pass).draw_multi_indexed::<_, u16>(
            draw,
            shader::RenderData {
                positions: positions.gpu(),
            },
            &indices,
            &[MultiDrawIndexed {
                first_index,
                index_count: 3,
                vertex_offset,
            }],
        )?;
        recorder
            .compute()
            .barrier(Stage::COLOR_OUT, Stage::PIXEL, Hazard::TEXTURES);
        let pass = Pass::new().color_texture(0, &copied, Some([1.0, 0.0, 0.0, 1.0]));
        let raster = recorder.raster(&pass);
        raster.heap(&resources);
        raster.draw(
            copy,
            shader::CopyData {
                positions: unsafe {
                    Gpu::new(positions.gpu::<f32x4>().get() + size_of::<f32x4>() as u64)
                },
                source,
                sampler,
            },
            0..3,
        )?;
        let compute = recorder.compute();
        compute.barrier(Stage::COLOR_OUT, Stage::TRANSFER, Hazard::TEXTURES);
        compute.read_texture(&copied, &readback, pixels.gpu.as_rw());
        if frame_index < FRAMES_IN_FLIGHT {
            compute.timestamp(&timers, 1);
        }
        frame.submit(None)?;
        device.wait_idle()?;
        for pixel in pixels.chunks_exact(4) {
            assert_eq!(pixel, [0, 255, 0, 255]);
        }
    }

    // Clear before the storage write so stale render output cannot satisfy the assertion.
    let mut frame = context.begin_frame()?;
    let recorder = frame.recorder();
    let pass = Pass::new().color_texture(0, &target, Some([1.0, 0.0, 0.0, 1.0]));
    recorder.raster(&pass);
    let compute = recorder.compute();
    compute.barrier(Stage::COLOR_OUT, Stage::COMPUTE, Hazard::TEXTURES);
    compute.heap(&resources);
    compute.dispatch(paint, shader::Paint { target: texture }, Dispatch::x(1))?;
    compute.barrier(Stage::COMPUTE, Stage::TRANSFER, Hazard::TEXTURES);
    compute.read_texture(&target, &readback, pixels.gpu.as_rw());
    frame.submit(None)?;
    device.wait_idle()?;
    assert_eq!(&pixels[..4], &[0, 255, 0, 255]);
    assert_eq!(&pixels[4..8], &[255, 0, 0, 255]);
    let mut frame = context.begin_frame()?;
    let compute = frame.recorder().compute();
    let invalid_fill = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compute.fill_buffer(&indices, 0, 4, 0x01020304);
    }));
    assert!(
        invalid_fill.is_err(),
        "word fills must not silently truncate in release builds"
    );
    Ok(())
}
