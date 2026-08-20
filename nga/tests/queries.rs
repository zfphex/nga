//! Timestamps and occlusion counters, headless. Both read back what a frame `FRAMES_IN_FLIGHT`
//! ago recorded, so the loop below runs long enough for the first frame's numbers to come back
//! and checks them against a triangle whose coverage `raster.rs` already pins.

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

#[test]
fn a_pass_reports_its_time_and_its_coverage() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };
    assert!(device.supports(Feature::Timestamps));
    assert!(device.supports(Feature::Occlusion));

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let pipeline = shaders.raster(
        &RasterDesc::new(
            "vertex_main",
            "fragment_main",
            &[ColorTarget::new(Format::Rgba8Unorm)],
        )
        .depth(Format::D32Float),
    )?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut heap = GpuHeap::new(&device, 16 * 1024 * 1024)?;
    let target = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE).usage(Usage::COLOR),
    )?;
    let depth = heap.alloc_texture(&TextureDesc::depth(Format::D32Float, SIZE, SIZE))?;

    // The same triangle `raster.rs` draws: 80% of the frame in each axis, so a little under a
    // third of the pixels pass the depth test.
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

    let mut timers = Timers::new(&device, 2)?;
    let mut occlusion = Occlusion::new(&device, 1)?;

    // Two frames per slot, so the last iteration reads results the GPU has certainly finished
    // writing rather than the zeroes a freshly reset pool holds.
    let frames = FRAMES_IN_FLIGHT * 2;
    for _ in 0..frames {
        let mut frame = ctx.begin_frame()?;
        timers.begin(&frame);
        occlusion.begin(&frame);

        let data = frame
            .arena(0)
            .alloc(Data {
                vertices,
                mvp: IDENTITY,
            })
            .unwrap();
        let root = data.gpu;

        let mut recorder = frame.recorder();
        {
            let pass = Pass::new()
                .color_texture(0, &target, Some([0.0, 0.0, 0.0, 1.0]))
                .depth(&depth, Some(1.0), true)
                .occlusion(&occlusion);
            let raster = recorder.raster(&pass, SIZE, SIZE);
            raster.timestamp(&timers, 0);
            raster.pipeline(pipeline);
            raster.depth(&DepthState::new(CompareOp::LessEqual, true));
            raster.occlusion(0);
            raster.draw(root, 0, 3);
            raster.occlusion_off();
            raster.timestamp(&timers, 1);
        }
        drop(recorder);
        frame.submit(None)?;
    }
    device.wait_idle()?;

    // One more `begin` to fetch what the last completed frame left behind.
    let mut frame = ctx.begin_frame()?;
    timers.begin(&frame);
    occlusion.begin(&frame);
    drop(frame.recorder());
    frame.submit(None)?;
    device.wait_idle()?;

    let covered = occlusion.samples(0);
    let total = (SIZE * SIZE) as u64;
    assert!(
        (total / 4..total / 2).contains(&covered),
        "{covered} of {total} samples passed; that is not the triangle"
    );

    assert!(
        timers.written(0) && timers.written(1),
        "neither timestamp landed"
    );
    // One triangle can start and finish inside a single tick, so the only claim worth making
    // is that the pair is ordered and the number is a duration rather than a clock reading.
    let ms = timers.ms(0, 1);
    assert!(ms < 1000.0, "{ms} ms for one triangle is not a duration");

    heap.free_texture(target);
    heap.free_texture(depth);
    Ok(())
}
