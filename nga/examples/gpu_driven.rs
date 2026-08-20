use minwin::*;
use nga::*;

#[gpu_struct]
struct Instance {
    center: f32x2,
    angle: f32,
    scale: f32,
    color: f32x4,
}

#[gpu_struct]
struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

#[gpu_struct]
struct SimData {
    instances: GpuMut<'static, Instance>,
    time: f32,
    count: u32,
}

#[gpu_struct]
struct EncodeData {
    instances: Gpu<'static, Instance>,
    draws: GpuMut<'static, DrawArgs>,
    count: GpuMut<'static, AtomicU32>,
    cull_radius: f32,
    total: u32,
}

#[gpu_struct(align(16))]
struct DrawData {
    instances: Gpu<'static, Instance>,
    mvp: f32x4x4,
}

#[gpu_struct(align(16))]
struct BoundaryData {
    mvp: f32x4x4,
    radius: f32,
    segments: u32,
}

const SHADERS: &[u8] = nga::shader!("gpu_driven");
const COUNT: usize = 4096;
const GROUP: usize = 64;
const SEGMENTS: u32 = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window(
        "nga-vk M5 — gpu driven",
        None,
        900,
        900,
        true,
        WindowStyle::Standard,
    );

    let device = Device::new(cfg!(debug_assertions))?;
    eprintln!("device: {}", device.name());

    let (mut width, mut height) = window.scaled_size();
    // The platform window handle: an `HWND` on Windows, the `NSView` on macOS.
    #[cfg(not(target_vendor = "apple"))]
    let handle = window.hwnd;
    #[cfg(target_vendor = "apple")]
    let handle = window.ns_view as isize;
    let mut surface = Surface::new(&device, handle, width as u32, height as u32)?;

    slang::dump(
        "gpu_driven",
        &format!(
            "{SLANG_PRELUDE}{}",
            slang::header!(
                Instance,
                DrawArgs,
                SimData,
                EncodeData,
                DrawData,
                BoundaryData
            )
        ),
    );

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let simulate = shaders.compute("simulate", None)?;
    let encode = shaders.compute("encode", None)?;
    let draw = shaders.raster(&RasterDesc::new(
        "vertex_main",
        "fragment_main",
        &[ColorTarget::new(surface.format())],
    ))?;
    let boundary = shaders.raster(&RasterDesc {
        topology: Topology::TriangleStrip,
        ..RasterDesc::new(
            "boundary_main",
            "boundary_fragment",
            &[ColorTarget {
                blend: BlendMode::Alpha,
                ..ColorTarget::new(surface.format())
            }],
        )
    })?;

    let mut heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;
    let instances = heap.alloc_buffer(COUNT * size_of::<Instance>())?;
    let draws = heap.alloc_buffer(COUNT * size_of::<DrawArgs>())?;
    let counter = heap.alloc_buffer(size_of::<u32>())?;

    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut time = 0.0f32;
    let mut resized = false;

    while window.open() {
        if window.pressed(Key::Escape) {
            window.close();
        }

        window.draw(|win| {
            let (new_width, new_height) = win.scaled_size();
            if (new_width, new_height) != (width, height) || resized {
                (width, height) = (new_width, new_height);
                surface
                    .rebuild(&device, width as u32, height as u32)
                    .unwrap();
                resized = false;
            }
            if surface.width() == 0 || surface.height() == 0 {
                return;
            }

            let Some(drawable) = surface.acquire(&device) else {
                resized = true;
                return;
            };

            let aspect = surface.width() as f32 / surface.height() as f32;
            let mvp = f32x4x4([
                [1.0 / aspect, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            let cull_radius = 0.55 + 0.35 * (time * 0.6).sin();

            let mut frame = ctx.begin_frame().unwrap();
            let mut rec = frame.recorder();

            let sim_root = rec
                .arena()
                .alloc(SimData {
                    instances: GpuMut::new(instances.address()),
                    time,
                    count: COUNT as u32,
                })
                .unwrap()
                .gpu;
            let encode_root = rec
                .arena()
                .alloc(EncodeData {
                    instances: Gpu::new(instances.address()),
                    draws: GpuMut::new(draws.address()),
                    count: GpuMut::new(counter.address()),
                    cull_radius,
                    total: COUNT as u32,
                })
                .unwrap()
                .gpu;
            let draw_root = rec
                .arena()
                .alloc(DrawData {
                    instances: Gpu::new(instances.address()),
                    mvp,
                })
                .unwrap()
                .gpu;
            let boundary_root = rec
                .arena()
                .alloc(BoundaryData {
                    mvp,
                    radius: cull_radius,
                    segments: SEGMENTS,
                })
                .unwrap()
                .gpu;

            {
                let compute = rec.compute();
                // Metal resets the indirect command buffer's execution range with a dedicated
                // `reset` command. Vulkan's equivalent is just zeroing the count word.
                compute.fill_buffer(&counter, 0, size_of::<u32>(), 0);
                // Every draw slot the encode pass does not claim this frame has to read as
                // `instance_count == 0`: Vulkan stops at the count, but the Metal backend
                // replays all `COUNT` slots because there is no GPU-side count to stop at.
                compute.fill_buffer(&draws, 0, draws.len(), 0);
                compute.barrier(Stage::TRANSFER, Stage::COMPUTE, Hazard::ALL);

                compute.pipeline(simulate);
                compute.dispatch(sim_root, COUNT, GROUP);
                compute.barrier(Stage::COMPUTE, Stage::COMPUTE, Hazard::ALL);

                compute.pipeline(encode);
                compute.dispatch(encode_root, COUNT, GROUP);
                compute.barrier(Stage::COMPUTE, Stage::INDIRECT, Hazard::ALL);
            }

            {
                let pass = Pass::new().color(0, &drawable, Some([0.02, 0.02, 0.04, 1.0]));
                let raster = rec.raster(&pass, surface.width(), surface.height());
                raster.pipeline(draw);
                raster.draw_indirect_count(draw_root, &draws, &counter, COUNT as u32);

                raster.pipeline(boundary);
                raster.draw(boundary_root, 0, SEGMENTS * 2 + 2);
            }

            drop(rec);
            frame.submit(Some(&drawable)).unwrap();

            resized = surface.present(&device, &drawable);

            time += 0.016;
        });

        window.wait_for_vsync();
    }

    device.wait_idle()?;
    Ok(())
}
