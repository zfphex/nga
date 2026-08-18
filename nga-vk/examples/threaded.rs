use ash::vk;
use minwin::*;
use nga_vk::*;

#[gpu_struct]
struct Instance {
    center: f32x2,
    angle: f32,
    scale: f32,
    color: f32x4,
}

#[gpu_struct]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[gpu_struct]
struct SimData {
    instances: GpuMut<'static, Instance>,
    time: f32,
    count: u32,
}

#[gpu_struct]
struct BuildData {
    instances: Gpu<'static, Instance>,
    vertices: GpuMut<'static, Vertex>,
    count: u32,
    pad: u32,
}

#[gpu_struct(align(16))]
struct DrawData {
    vertices: Gpu<'static, Vertex>,
    mvp: f32x4x4,
}

const SHADERS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sync.spv"));

const THREADS: usize = 4;
const SLICE: usize = 256;
const COUNT: usize = THREADS * SLICE;
const GROUP: usize = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window(
        "nga-vk M4 — threaded",
        None,
        900,
        700,
        true,
        WindowStyle::Standard,
    );

    let device = Device::new(cfg!(debug_assertions))?;
    eprintln!("device: {}", device.name);

    let (mut width, mut height) = window.scaled_size();
    let mut swapchain = Swapchain::new(&device, window.hwnd, width as u32, height as u32)?;

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let simulate = shaders.compute("simulate", None)?;
    let build = shaders.compute("build_vertices", None)?;
    let draw = shaders.raster(&RasterDesc::new(
        "vertex_main",
        "fragment_main",
        &[ColorTarget::new(swapchain.format)],
    ))?;

    let mut heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;
    let instances = heap.alloc_buffer(FRAMES_IN_FLIGHT as usize * COUNT * size_of::<Instance>())?;
    let vertices =
        heap.alloc_buffer(FRAMES_IN_FLIGHT as usize * COUNT * 3 * size_of::<Vertex>())?;

    let mut ctx = Context::new(&device, &shaders, 64 * 1024, THREADS)?;

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
                swapchain
                    .rebuild(&device, width as u32, height as u32)
                    .unwrap();
                resized = false;
            }
            if swapchain.width == 0 || swapchain.height == 0 {
                return;
            }

            let Some(drawable) = swapchain.acquire(&device) else {
                resized = true;
                return;
            };

            let aspect = swapchain.width as f32 / swapchain.height as f32;
            let mvp = f32x4x4([
                [1.0 / aspect, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);

            let slot = (ctx.frame % FRAMES_IN_FLIGHT) as usize * COUNT;
            let (sw, sh) = (swapchain.width, swapchain.height);
            let mut frame = ctx.begin_frame().unwrap();
            let recorders = frame.recorders(&[THREADS]);

            std::thread::scope(|scope| {
                for (index, mut rec) in recorders.into_iter().enumerate() {
                    let (simulate, build, draw, drawable) = (simulate, build, draw, &drawable);
                    let instance_base =
                        instances.gpu + ((slot + index * SLICE) * size_of::<Instance>()) as u64;
                    let vertex_base =
                        vertices.gpu + ((slot + index * SLICE) * 3 * size_of::<Vertex>()) as u64;

                    scope.spawn(move || {
                        let sim_root = rec
                            .arena
                            .alloc(SimData {
                                instances: GpuMut::new(instance_base),
                                time: time + index as f32 * 0.35,
                                count: SLICE as u32,
                            })
                            .unwrap()
                            .gpu;
                        let build_root = rec
                            .arena
                            .alloc(BuildData {
                                instances: Gpu::new(instance_base),
                                vertices: GpuMut::new(vertex_base),
                                count: SLICE as u32,
                                pad: 0,
                            })
                            .unwrap()
                            .gpu;
                        let draw_root = rec
                            .arena
                            .alloc(DrawData {
                                vertices: Gpu::new(vertex_base),
                                mvp,
                            })
                            .unwrap()
                            .gpu;

                        {
                            let compute = rec.compute();
                            compute.pipeline(simulate);
                            compute.dispatch(sim_root, SLICE, GROUP);
                            compute.barrier(Stages::DISPATCH, Stages::DISPATCH, Visibility::Device);
                            compute.pipeline(build);
                            compute.dispatch(build_root, SLICE, GROUP);
                            compute.barrier(Stages::DISPATCH, Stages::VERTEX, Visibility::Device);
                        }

                        // Only the first thread clears; the rest load. `Raster::new` turns that
                        // into the right layout transition, which is what keeps four separately
                        // recorded command buffers from discarding each other's output.
                        let clear = match index {
                            0 => Some([0.03, 0.03, 0.05, 1.0]),
                            _ => None,
                        };
                        let pass = Pass::new().color(0, drawable, clear);
                        let raster = rec.raster(&pass, sw, sh);
                        raster.pipeline(draw);
                        raster.draw(draw_root, 0, (SLICE * 3) as u32);
                    });
                }
            });

            frame.submit(Some(&drawable)).unwrap();

            let indices = [drawable.index];
            let swapchains = [swapchain.swapchain];
            let waits = [drawable.released];
            let present = unsafe {
                device.swapchain_ext.queue_present(
                    device.queue(),
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&waits)
                        .swapchains(&swapchains)
                        .image_indices(&indices),
                )
            };
            if matches!(present, Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Ok(true)) {
                resized = true;
            }

            time += 0.016;
        });

        window.wait_for_vsync();
    }

    unsafe { device.device.device_wait_idle()? };
    Ok(())
}
