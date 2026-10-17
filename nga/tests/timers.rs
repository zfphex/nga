use nga::*;

#[gpu_mod]
mod shader {
    use nga::*;

    pub struct VertexOut {
        pub position: f32x4,
    }

    #[vertex]
    fn vertex(#[vertex_id] vid: u32) -> VertexOut {
        VertexOut {
            position: f32x4::new(
                (vid % 2) as f32 * 4.0 - 1.0,
                (vid / 2) as f32 * 4.0 - 1.0,
                0.0,
                1.0,
            ),
        }
    }

    #[fragment]
    fn fragment(_input: VertexOut) -> f32x4 {
        f32x4::splat(1.0)
    }
}

#[test]
#[ignore = "requires a GPU"]
fn pass_timestamps_and_skipped_slots() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(true)?;
    if !device.supports(Feature::Timestamps) {
        return Ok(());
    }
    let mut shaders = device.shaders(shader::MODULE)?;
    let pipeline = shaders
        .raster(shader::VERTEX, shader::FRAGMENT)
        .color(Format::Rgba8Unorm)
        .build()?;
    let heap = GpuHeap::new(&device, 1024 * 1024)?;
    let buffer = heap.alloc_buffer(64 * 1024)?;
    let texture = heap
        .alloc_texture(&TextureDesc::texture_2d(Format::Rgba8Unorm, 64, 64).usage(Usage::COLOR))?;
    let mut context = Context::new(&device, &shaders, 4096, 1)?;
    let mut timers = Timers::new(&device, 5)?;
    for index in 0..FRAMES_IN_FLIGHT * 5 {
        timers.enabled = !(FRAMES_IN_FLIGHT..FRAMES_IN_FLIGHT * 2).contains(&index);
        let mut frame = context.begin_frame()?;
        timers.begin(&frame);
        let expected = index >= FRAMES_IN_FLIGHT * 4;
        for (start, end) in [(0, 1), (2, 3)] {
            assert_eq!(timers.checked_ms(start, end).is_some(), expected);
            if expected {
                assert_eq!(timers.checked_ns(start, start), Some(0.0));
                let duration = timers.checked_ns(start, end).unwrap();
                assert!(duration.is_finite() && duration >= 0.0);
                if duration > 0.0 {
                    assert_eq!(timers.checked_ns(end, start), None);
                }
            }
        }
        assert_eq!(timers.checked_ms(0, 4), None);
        assert_eq!(timers.checked_ms(4, 0), None);
        assert_eq!(timers.checked_ms(5, 0), None);
        assert_eq!(timers.checked_ms(0, usize::MAX), None);
        let recorder = frame.recorder();
        if index < FRAMES_IN_FLIGHT * 2 || (FRAMES_IN_FLIGHT * 3..FRAMES_IN_FLIGHT * 4).contains(&index) {
            let compute = recorder.compute();
            compute.timestamp(&timers, 0);
            compute.fill_buffer(&buffer, 0, buffer.len(), 0);
            compute.timestamp(&timers, 1);
            let pass = Pass::new().color_texture(0, &texture, Some([0.0; 4]));
            let raster = recorder.raster(&pass);
            raster.timestamp(&timers, 2);
            raster.draw(pipeline, (), 0..3)?;
            raster.timestamp(&timers, 3);
            if !timers.enabled { recorder.timestamp(&timers, 4); }
        }
        frame.submit(None)?;
    }
    device.wait_idle()?;
    assert_eq!(device.validation_errors(), 0);
    Ok(())
}
