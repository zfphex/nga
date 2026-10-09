use minwin::*;
use nga::*;
#[path = "shaders/sync.rs"]
mod sync;
use sync::{BuildData, DrawData, Instance, SimData, Vertex};

const THREADS: usize = 4;
const SLICE: usize = 256;
const COUNT: usize = THREADS * SLICE;

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
    let mut surface = device.surface(window.handle(), width as u32, height as u32)?;
    let mut shaders = device.shaders(sync::MODULE)?;
    let simulate = shaders.compute(sync::SIMULATE)?;
    let build = shaders.compute(sync::BUILD_VERTICES)?;
    let draw = shaders
        .raster(sync::VERTEX, sync::FRAGMENT)
        .color(surface.format())
        .build()?;

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
            let mut frame = ctx.begin_frame().unwrap();
            let recorders = frame.recorders(&[THREADS]);

            std::thread::scope(|scope| {
                for (index, mut rec) in recorders.into_iter().enumerate() {
                    let (simulate, build, draw, drawable) = (simulate, build, draw, &drawable);
                    let instance_base = instances.address()
                        + ((slot + index * SLICE) * size_of::<Instance>()) as u64;
                    let vertex_base = vertices.address()
                        + ((slot + index * SLICE) * 3 * size_of::<Vertex>()) as u64;

                    scope.spawn(move || {
                        {
                            let compute = rec.compute();
                            compute
                                .dispatch(
                                    simulate,
                                    SimData {
                                        instances: unsafe { GpuRw::new(instance_base) },
                                        time: time + index as f32 * 0.35,
                                        count: SLICE as u32,
                                    },
                                    Dispatch::x(SLICE),
                                )
                                .unwrap();
                            compute.barrier(Stage::COMPUTE, Stage::COMPUTE, Hazard::BUFFERS);
                            compute
                                .dispatch(
                                    build,
                                    BuildData {
                                        instances: unsafe { Gpu::new(instance_base) },
                                        vertices: unsafe { GpuRw::new(vertex_base) },
                                        count: SLICE as u32,
                                        pad: 0,
                                    },
                                    Dispatch::x(SLICE),
                                )
                                .unwrap();
                            compute.barrier(Stage::COMPUTE, Stage::VERTEX, Hazard::BUFFERS);
                        }

                        // Only the first thread clears; the rest load. `Raster::new` turns that
                        // into the right layout transition, which is what keeps four separately
                        // recorded command buffers from discarding each other's output.
                        let clear = match index {
                            0 => Some([0.03, 0.03, 0.05, 1.0]),
                            _ => None,
                        };
                        let pass = Pass::new().color(0, drawable, clear);
                        let raster = rec.raster(&pass);
                        raster
                            .draw(
                                draw,
                                DrawData {
                                    vertices: unsafe { Gpu::new(vertex_base) },
                                    mvp,
                                },
                                0..(SLICE * 3) as u32,
                            )
                            .unwrap();
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
