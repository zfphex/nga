use minwin::*;
use nga::*;
#[path = "shaders/sync.rs"]
mod sync;
use sync::{BuildData, DrawData, Instance, SimData, Vertex};

const COUNT: usize = 1024;
const DEPTH_FORMAT: Format = Format::D32Float;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window(
        "nga-vk M4 — sync",
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
        .depth(DEPTH_FORMAT)
        .build()?;
    let depth_write = DepthState::new(CompareOp::Less, true);

    let heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;
    let instances = heap.alloc_buffer(COUNT * size_of::<Instance>())?;
    let vertices = heap.alloc_buffer(COUNT * 3 * size_of::<Vertex>())?;
    let instances_rw: GpuRw<[Instance]> = instances.gpu_rw();
    let instances_ro: Gpu<[Instance]> = instances.gpu();
    let vertices_rw: GpuRw<[Vertex]> = vertices.gpu_rw();
    let vertices_ro: Gpu<[Vertex]> = vertices.gpu();

    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut depth: Option<(HeapTexture, u32, u32)> = None;
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

            if depth
                .as_ref()
                .is_none_or(|&(_, w, h)| w != surface.width() || h != surface.height())
            {
                device.wait_idle().unwrap();
                depth = None;
                let texture = heap
                    .alloc_texture(&TextureDesc::depth(
                        DEPTH_FORMAT,
                        surface.width(),
                        surface.height(),
                    ))
                    .unwrap();
                depth = Some((texture, surface.width(), surface.height()));
            }

            let aspect = surface.width() as f32 / surface.height() as f32;
            let mvp = f32x4x4([
                [1.0 / aspect, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);

            let mut frame = ctx.begin_frame().unwrap();
            let rec = frame.recorder();

            let compute = rec.compute();
            compute
                .dispatch(
                    simulate,
                    SimData {
                        instances: instances_rw,
                        time,
                        count: COUNT as u32,
                    },
                    Dispatch::x(COUNT),
                )
                .unwrap();
            compute.barrier(Stage::COMPUTE, Stage::COMPUTE, Hazard::BUFFERS);
            compute
                .dispatch(
                    build,
                    BuildData {
                        instances: instances_ro,
                        vertices: vertices_rw,
                        count: COUNT as u32,
                        pad: 0,
                    },
                    Dispatch::x(COUNT),
                )
                .unwrap();
            // Metal signals a fence here and the raster encoder waits on it. Vulkan has no
            // cross-encoder fence inside one command buffer: a barrier on the queue is the
            // whole mechanism, so the compute -> vertex handoff is one more `barrier`.
            compute.barrier(Stage::COMPUTE, Stage::VERTEX, Hazard::BUFFERS);

            let pass = Pass::new()
                .color(0, &drawable, Some([0.03, 0.03, 0.05, 1.0]))
                .depth(&depth.as_ref().unwrap().0, Some(1.0), false);
            let raster = rec.raster(&pass);
            raster.depth(&depth_write);
            raster
                .draw(
                    draw,
                    DrawData {
                        vertices: vertices_ro,
                        mvp,
                    },
                    0..(COUNT * 3) as u32,
                )
                .unwrap();

            frame.submit(Some(&drawable)).unwrap();

            resized = surface.present(&device, &drawable);

            time += 0.016;
        });

        window.wait_for_vsync();
    }

    device.wait_idle()?;
    Ok(())
}
