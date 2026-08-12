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

const THREADS: usize = 4;
const SLICE: usize = 256;
const COUNT: usize = THREADS * SLICE;
const GROUP: usize = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window(
        "nga M4 — threaded",
        None,
        900,
        700,
        true,
        WindowStyle::Standard,
    );

    let device = Device::required_system_default()?;
    eprintln!("device: {}", device.name());

    let mut ctx = Context::new(&device, 64 * 1024, 64 * 1024 * 1024, 1024, 1, THREADS)?;

    let source = format!(
        "{}\n{SHADERS}",
        msl::header!(Instance, Vertex, SimData, BuildData, DrawData)
    );
    msl::dump("threaded", &format!("{MSL_PRELUDE}{source}"));

    let shaders = Shaders::new(&device, &source)?;
    let simulate = shaders.compute("simulate", None)?;
    let build = shaders.compute("build_vertices", None)?;
    let draw = shaders.raster(&RasterDesc::new(
        "vertex_main",
        "fragment_main",
        &[ColorTarget::new(PixelFormat::Bgra8Unorm)],
    ))?;

    let fences = (0..THREADS)
        .map(|_| device.new_fence())
        .collect::<Result<Vec<_>, _>>()?;
    let chain = (0..THREADS)
        .map(|_| device.new_fence())
        .collect::<Result<Vec<_>, _>>()?;

    let frames = FRAMES_IN_FLIGHT as usize;
    let instances = ctx.heap.alloc_buffer(frames * COUNT * size_of::<Instance>())?;
    let vertices = ctx
        .heap
        .alloc_buffer(frames * COUNT * 3 * size_of::<Vertex>())?;

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

            let aspect = drawable_width as f32 / drawable_height as f32;
            let mvp = f32x4x4([
                [1.0 / aspect, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            let viewport = Viewport::new(
                0.0,
                0.0,
                drawable_width as f64,
                drawable_height as f64,
                0.0,
                1.0,
            );

            let slot = (ctx.frame % FRAMES_IN_FLIGHT) as usize * COUNT;
            let mut frame = ctx.begin_frame();
            let recorders = frame.recorders(&[THREADS]);
            std::thread::scope(|scope| {
                for (index, mut rec) in recorders.into_iter().enumerate() {
                    let (target, simulate, build, draw) = (&target, &simulate, &build, &draw);
                    let (fence, chain) = (&fences[index], &chain);
                    let instances =
                        instances.gpu + ((slot + index * SLICE) * size_of::<Instance>()) as u64;
                    let vertices =
                        vertices.gpu + ((slot + index * SLICE) * 3 * size_of::<Vertex>()) as u64;
                    scope.spawn(move || {
                        let Some(sim_root) = rec.arena.alloc(SimData {
                            instances: GpuMut::new(instances),
                            time: time + index as f32 * 0.35,
                            count: SLICE as u32,
                        }) else {
                            return;
                        };
                        let Some(build_root) = rec.arena.alloc(BuildData {
                            instances: Gpu::new(instances),
                            vertices: GpuMut::new(vertices),
                            count: SLICE as u32,
                            pad: 0,
                        }) else {
                            return;
                        };
                        let Some(draw_root) = rec.arena.alloc(DrawData {
                            vertices: Gpu::new(vertices),
                            mvp,
                        }) else {
                            return;
                        };

                        {
                            let Some(compute) = rec.compute() else {
                                return;
                            };
                            compute.pipeline(simulate);
                            compute.dispatch(sim_root.gpu, SLICE, GROUP);
                            compute.barrier(
                                Stages::DISPATCH,
                                Stages::DISPATCH,
                                M4VisibilityOptions::DEVICE,
                            );
                            compute.pipeline(build);
                            compute.dispatch(build_root.gpu, SLICE, GROUP);
                            compute.barrier(
                                Stages::DISPATCH,
                                Stages::DISPATCH,
                                M4VisibilityOptions::DEVICE,
                            );
                            compute.signal_fence(fence, Stages::DISPATCH);
                        }

                        let clear = match index {
                            0 => Some(ClearColor::new(0.03, 0.03, 0.05, 1.0)),
                            _ => None,
                        };
                        let pass = Pass::new().color(0, target, clear);
                        let Some(raster) = rec.raster(&pass, viewport) else {
                            return;
                        };
                        raster.wait_fence(fence, Stages::VERTEX);
                        if index > 0 {
                            raster.wait_fence(&chain[index - 1], Stages::FRAGMENT);
                        }
                        raster.pipeline(draw);
                        raster.draw(draw_root.gpu, PrimitiveType::Triangle, 0, SLICE * 3);
                        raster.signal_fence(&chain[index], Stages::FRAGMENT);
                    });
                }
            });

            frame.submit(Some(&drawable));
            time += 0.016;
        });

        window.wait_for_vsync();
    }

    Ok(())
}
