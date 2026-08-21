use minwin::*;
use nga::*;
use nga_shaders::sync::{self, Instance, Vertex, SimData, BuildData, DrawData};







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
    eprintln!("device: {}", device.name());

    let (mut width, mut height) = window.scaled_size();
    // The platform window handle: an `HWND` on Windows, the `NSView` on macOS.
    #[cfg(not(target_vendor = "apple"))]
    let handle = window.hwnd;
    #[cfg(target_vendor = "apple")]
    let handle = window.ns_view as isize;
    let mut surface = Surface::new(&device, handle, width as u32, height as u32)?;

    let mut shaders = Shaders::new(&device, &nga::compile(sync::ENTRIES))?;
    let simulate = shaders.compute("simulate", None)?;
    let build = shaders.compute("build_vertices", None)?;
    let draw = shaders.raster(&RasterDesc::new(
        "vertex",
        "fragment",
        &[ColorTarget::new(surface.format())],
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

            let slot = (ctx.frame() % FRAMES_IN_FLIGHT) as usize * COUNT;
            let (sw, sh) = (surface.width(), surface.height());
            let mut frame = ctx.begin_frame().unwrap();
            let recorders = frame.recorders(&[THREADS]);

            std::thread::scope(|scope| {
                for (index, mut rec) in recorders.into_iter().enumerate() {
                    let (simulate, build, draw, drawable) = (simulate, build, draw, &drawable);
                    let instance_base =
                        instances.address() + ((slot + index * SLICE) * size_of::<Instance>()) as u64;
                    let vertex_base =
                        vertices.address() + ((slot + index * SLICE) * 3 * size_of::<Vertex>()) as u64;

                    scope.spawn(move || {
                        let sim_root = rec
                            .arena()
                            .alloc(SimData {
                                instances: GpuMut::new(instance_base),
                                time: time + index as f32 * 0.35,
                                count: SLICE as u32,
                            })
                            .unwrap()
                            .gpu;
                        let build_root = rec
                            .arena()
                            .alloc(BuildData {
                                instances: Gpu::new(instance_base),
                                vertices: GpuMut::new(vertex_base),
                                count: SLICE as u32,
                                pad: 0,
                            })
                            .unwrap()
                            .gpu;
                        let draw_root = rec
                            .arena()
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
                            compute.barrier(Stage::COMPUTE, Stage::COMPUTE, Hazard::ALL);
                            compute.pipeline(build);
                            compute.dispatch(build_root, SLICE, GROUP);
                            compute.barrier(Stage::COMPUTE, Stage::VERTEX, Hazard::ALL);
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

            resized = surface.present(&device, &drawable);

            time += 0.016;
        });

        window.wait_for_vsync();
    }

    device.wait_idle()?;
    heap.free_buffer(vertices);
    heap.free_buffer(instances);
    Ok(())
}
