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
const COUNT: usize = 1024;
const GROUP: usize = 64;
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
    let mut surface = Surface::new(&device, window.hwnd, width as u32, height as u32)?;

    slang::dump(
        "sync",
        &format!(
            "{SLANG_PRELUDE}{}",
            slang::header!(Instance, Vertex, SimData, BuildData, DrawData)
        ),
    );

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let simulate = shaders.compute("simulate", None)?;
    let build = shaders.compute("build_vertices", None)?;
    let draw = shaders.raster(&RasterDesc {
        depth: Some(DEPTH_FORMAT),
        ..RasterDesc::new(
            "vertex_main",
            "fragment_main",
            &[ColorTarget::new(surface.format())],
        )
    })?;
    let depth_write = DepthState::new(CompareOp::Less, true);

    let mut heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;
    let instances = heap.alloc_buffer(COUNT * size_of::<Instance>())?;
    let vertices = heap.alloc_buffer(COUNT * 3 * size_of::<Vertex>())?;
    let instances_rw: GpuMut<'static, Instance> = GpuMut::new(instances.address());
    let instances_ro: Gpu<'static, Instance> = Gpu::new(instances.address());
    let vertices_rw: GpuMut<'static, Vertex> = GpuMut::new(vertices.address());
    let vertices_ro: Gpu<'static, Vertex> = Gpu::new(vertices.address());

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
                if let Some((old, _, _)) = depth.take() {
                    heap.free_texture(old);
                }
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
            let mut rec = frame.recorder();
            let sim_root = rec
                .arena()
                .alloc(SimData {
                    instances: instances_rw,
                    time,
                    count: COUNT as u32,
                })
                .unwrap()
                .gpu;
            let build_root = rec
                .arena()
                .alloc(BuildData {
                    instances: instances_ro,
                    vertices: vertices_rw,
                    count: COUNT as u32,
                    pad: 0,
                })
                .unwrap()
                .gpu;
            let draw_root = rec
                .arena()
                .alloc(DrawData {
                    vertices: vertices_ro,
                    mvp,
                })
                .unwrap()
                .gpu;

            {
                let compute = rec.compute();
                compute.pipeline(simulate);
                compute.dispatch(sim_root, COUNT, GROUP);
                compute.barrier(Stage::COMPUTE, Stage::COMPUTE, Hazard::ALL);
                compute.pipeline(build);
                compute.dispatch(build_root, COUNT, GROUP);
                // Metal signals a fence here and the raster encoder waits on it. Vulkan has no
                // cross-encoder fence inside one command buffer: a barrier on the queue is the
                // whole mechanism, so the compute -> vertex handoff is one more `barrier`.
                compute.barrier(Stage::COMPUTE, Stage::VERTEX, Hazard::ALL);
            }

            {
                let pass = Pass::new()
                    .color(0, &drawable, Some([0.03, 0.03, 0.05, 1.0]))
                    .depth(&depth.as_ref().unwrap().0, Some(1.0), false);
                let raster = rec.raster(&pass, surface.width(), surface.height());
                raster.pipeline(draw);
                raster.depth(&depth_write);
                raster.draw(draw_root, 0, (COUNT * 3) as u32);
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
