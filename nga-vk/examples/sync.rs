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
const COUNT: usize = 1024;
const GROUP: usize = 64;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

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
    eprintln!("device: {}", device.name);

    let (mut width, mut height) = window.scaled_size();
    let mut swapchain = Swapchain::new(&device, window.hwnd, width as u32, height as u32)?;

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
        depth: DEPTH_FORMAT,
        ..RasterDesc::new(
            "vertex_main",
            "fragment_main",
            &[ColorTarget::new(swapchain.format)],
        )
    })?;
    let depth_write = depth_state(vk::CompareOp::LESS, true);

    let mut heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;
    let instances = heap.alloc_buffer(COUNT * size_of::<Instance>())?;
    let vertices = heap.alloc_buffer(COUNT * 3 * size_of::<Vertex>())?;
    let instances_rw: GpuMut<'static, Instance> = GpuMut::new(instances.gpu);
    let instances_ro: Gpu<'static, Instance> = Gpu::new(instances.gpu);
    let vertices_rw: GpuMut<'static, Vertex> = GpuMut::new(vertices.gpu);
    let vertices_ro: Gpu<'static, Vertex> = Gpu::new(vertices.gpu);

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

            if depth
                .as_ref()
                .is_none_or(|&(_, w, h)| w != swapchain.width || h != swapchain.height)
            {
                unsafe { device.device.device_wait_idle().unwrap() };
                if let Some((old, _, _)) = depth.take() {
                    unsafe {
                        device.device.destroy_image_view(old.view, None);
                        device.device.destroy_image(old.image, None);
                    }
                    heap.free(old.alloc);
                }
                let texture = heap
                    .alloc_texture(&TextureDesc::depth(
                        DEPTH_FORMAT,
                        swapchain.width,
                        swapchain.height,
                    ))
                    .unwrap();
                depth = Some((texture, swapchain.width, swapchain.height));
            }

            let aspect = swapchain.width as f32 / swapchain.height as f32;
            let mvp = f32x4x4([
                [1.0 / aspect, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);

            let mut frame = ctx.begin_frame().unwrap();
            let mut rec = frame.recorder();
            let sim_root = rec
                .arena
                .alloc(SimData {
                    instances: instances_rw,
                    time,
                    count: COUNT as u32,
                })
                .unwrap()
                .gpu;
            let build_root = rec
                .arena
                .alloc(BuildData {
                    instances: instances_ro,
                    vertices: vertices_rw,
                    count: COUNT as u32,
                    pad: 0,
                })
                .unwrap()
                .gpu;
            let draw_root = rec
                .arena
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
                compute.barrier(Stages::DISPATCH, Stages::DISPATCH, Visibility::Device);
                compute.pipeline(build);
                compute.dispatch(build_root, COUNT, GROUP);
                // Metal signals a fence here and the raster encoder waits on it. Vulkan has no
                // cross-encoder fence inside one command buffer: a barrier on the queue is the
                // whole mechanism, so the compute -> vertex handoff is one more `barrier`.
                compute.barrier(Stages::DISPATCH, Stages::VERTEX, Visibility::Device);
            }

            {
                let pass = Pass::new()
                    .color(0, &drawable, Some([0.03, 0.03, 0.05, 1.0]))
                    .depth(&depth.as_ref().unwrap().0, Some(1.0), false);
                let raster = rec.raster(&pass, swapchain.width, swapchain.height);
                raster.pipeline(draw);
                raster.depth(&depth_write);
                raster.draw(draw_root, 0, (COUNT * 3) as u32);
            }

            drop(rec);
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
