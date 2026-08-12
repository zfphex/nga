use minmetal::*;
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

const SHADERS: &str = include_str!("../shaders/sync.metal");

const COUNT: usize = 1024;
const GROUP: usize = 64;
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window("nga M4 — sync", None, 900, 700, true, WindowStyle::Standard);

    let device = Device::required_system_default()?;
    eprintln!("device: {}", device.name());

    let mut ctx = Context::new(&device, 64 * 1024, 64 * 1024 * 1024, 1024, 1)?;

    let source = format!(
        "{}\n{SHADERS}",
        msl::header!(Instance, Vertex, SimData, BuildData, DrawData)
    );
    msl::dump("sync", &format!("{MSL_PRELUDE}{source}"));

    let shaders = Shaders::new(&device, &source)?;
    let simulate = shaders.compute("simulate", None)?;
    let build = shaders.compute("build_vertices", None)?;
    let draw = shaders.raster(&RasterDesc::new(
        "vertex_main",
        "fragment_main",
        &[ColorTarget::new(PixelFormat::Bgra8Unorm)],
    ))?;
    let depth_write = depth_state(&device, CompareFunction::Less, true)?;
    let vertices_ready = device.new_fence()?;

    let instances = ctx.heap.alloc_buffer(COUNT * size_of::<Instance>())?;
    let vertices = ctx.heap.alloc_buffer(COUNT * 3 * size_of::<Vertex>())?;
    let instances_rw: GpuMut<'static, Instance> = GpuMut::new(instances.gpu);
    let instances_ro: Gpu<'static, Instance> = Gpu::new(instances.gpu);
    let vertices_rw: GpuMut<'static, Vertex> = GpuMut::new(vertices.gpu);
    let vertices_ro: Gpu<'static, Vertex> = Gpu::new(vertices.gpu);

    let scale = window.scale_factor();
    let (mut drawable_width, mut drawable_height) = window.scaled_size();
    let layer = unsafe {
        MetalLayer::attach_to_view(
            id(window.ns_view),
            &device,
            PixelFormat::Bgra8Unorm,
            drawable_width,
            drawable_height,
            scale,
        )?
    };

    let mut depth: Option<(HeapTexture, usize, usize)> = None;
    let mut time = 0.0f32;

    while window.open() {
        let _pool = AutoreleasePool::new();

        if window.pressed(Key::Escape) {
            window.close();
        }

        window.draw(|win| {
            (drawable_width, drawable_height) = win.scaled_size();
            layer.set_contents_scale(win.scale_factor());
            layer.set_drawable_size(drawable_width, drawable_height);

            let Some(drawable) = layer.next_drawable() else {
                return;
            };
            let Ok(target) = drawable.texture() else {
                return;
            };

            if depth
                .as_ref()
                .is_none_or(|&(_, w, h)| w != drawable_width || h != drawable_height)
            {
                if let Some((old, _, _)) = depth.take() {
                    ctx.free_later(Retired::Texture(old));
                }
                let desc = TextureDescriptor::texture_2d(
                    DEPTH_FORMAT,
                    drawable_width,
                    drawable_height,
                    false,
                );
                desc.set_usage(TextureUsage::RENDER_TARGET);
                let Ok(texture) = ctx.heap.alloc_texture(&desc) else {
                    return;
                };
                depth = Some((texture, drawable_width, drawable_height));
            }

            let aspect = drawable_width as f32 / drawable_height as f32;
            let mvp = f32x4x4([
                [1.0 / aspect, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);

            let mut frame = ctx.begin_frame();
            let mut rec = frame.recorder();
            let Some(sim_root) = rec.arena.alloc(SimData {
                instances: instances_rw,
                time,
                count: COUNT as u32,
            }) else {
                return;
            };
            let Some(build_root) = rec.arena.alloc(BuildData {
                instances: instances_ro,
                vertices: vertices_rw,
                count: COUNT as u32,
                pad: 0,
            }) else {
                return;
            };
            let Some(draw_root) = rec.arena.alloc(DrawData {
                vertices: vertices_ro,
                mvp,
            }) else {
                return;
            };

            {
                let Some(compute) = rec.compute() else {
                    return;
                };
                compute.pipeline(&simulate);
                compute.dispatch(sim_root.gpu, COUNT, GROUP);
                compute.barrier(
                    Stages::DISPATCH,
                    Stages::DISPATCH,
                    M4VisibilityOptions::DEVICE,
                );
                compute.pipeline(&build);
                compute.dispatch(build_root.gpu, COUNT, GROUP);
                compute.barrier(
                    Stages::DISPATCH,
                    Stages::DISPATCH,
                    M4VisibilityOptions::DEVICE,
                );
                compute.signal_fence(&vertices_ready, Stages::DISPATCH);
            }

            let pass = Pass::new()
                .color(0, &target, Some(ClearColor::new(0.03, 0.03, 0.05, 1.0)))
                .depth(&depth.as_ref().unwrap().0.texture, Some(1.0), false);
            let viewport = Viewport::new(
                0.0,
                0.0,
                drawable_width as f64,
                drawable_height as f64,
                0.0,
                1.0,
            );

            {
                let Some(raster) = rec.raster(&pass, viewport) else {
                    return;
                };
                raster.wait_fence(&vertices_ready, Stages::VERTEX);
                raster.pipeline(&draw);
                raster.depth(&depth_write);
                raster.draw(draw_root.gpu, PrimitiveType::Triangle, 0, COUNT * 3);
            }

            drop(rec);

            frame.submit(Some(&drawable));
            time += 0.016;
        });

        window.wait_for_vsync();
    }

    Ok(())
}
