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
struct SimData {
    instances: GpuMut<'static, Instance>,
    time: f32,
    count: u32,
}

#[gpu_struct(align(16))]
struct DrawData {
    instance: Gpu<'static, Instance>,
    mvp: f32x4x4,
}

#[gpu_struct]
struct PrepareData {
    encode: GpuMut<'static, Dispatch>,
    ring: GpuMut<'static, Draw>,
    count: u32,
    group: u32,
    segments: u32,
    pad: u32,
}

#[gpu_struct(align(16))]
struct BoundaryData {
    mvp: f32x4x4,
    radius: f32,
    segments: u32,
}

#[gpu_struct(align(16))]
struct EncodeData {
    commands: Commands,
    instances: Gpu<'static, Instance>,
    draws: GpuMut<'static, DrawData>,
    range: GpuMut<'static, ExecRange>,
    mvp: f32x4x4,
    cull_radius: f32,
    count: u32,
}

const SHADERS: &str = include_str!("../shaders/gpu_driven.metal");

const COUNT: usize = 4096;
const GROUP: usize = 64;
const SEGMENTS: u32 = 192;
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window(
        "nga M6 — gpu driven",
        None,
        900,
        700,
        true,
        WindowStyle::Standard,
    );

    let device = Device::required_system_default()?;
    eprintln!("device: {}", device.name());

    let mut ctx = Context::new(&device, 4 * 1024 * 1024, 64 * 1024 * 1024, 1024, 1, 1)?;

    let source = format!(
        "{}\n{SHADERS}",
        msl::header!(
            Instance,
            SimData,
            DrawData,
            PrepareData,
            BoundaryData,
            EncodeData
        )
    );
    msl::dump("gpu_driven", &format!("{MSL_PRELUDE}{source}"));

    let shaders = Shaders::new(&device, &source)?;
    let prepare = shaders.compute("prepare", None)?;
    let simulate = shaders.compute("simulate", None)?;
    let encode = shaders.compute("encode", None)?;
    let draw = shaders.raster(&RasterDesc {
        indirect: true,
        ..RasterDesc::new(
            "vertex_main",
            "fragment_main",
            &[ColorTarget::new(PixelFormat::Bgra8Unorm)],
        )
    })?;
    let boundary = shaders.raster(&RasterDesc::new(
        "boundary_main",
        "boundary_fragment",
        &[ColorTarget {
            blend: BlendMode::Fixed(Blend::ALPHA),
            ..ColorTarget::new(PixelFormat::Bgra8Unorm)
        }],
    ))?;
    let depth_write = depth_state(&device, CompareFunction::Less, true)?;

    let commands = IndirectCommands::draws(&device, COUNT, false)?;
    ctx.resident(&Allocation::from_indirect_command_buffer(&commands.icb))?;

    let instances = ctx.heap.alloc_buffer(COUNT * size_of::<Instance>())?;
    let instances_rw: GpuMut<'static, Instance> = GpuMut::new(instances.gpu);
    let instances_ro: Gpu<'static, Instance> = Gpu::new(instances.gpu);

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

            let cull_radius = 0.75 + 0.7 * (time * 0.7).sin();

            let mut frame = ctx.begin_frame();
            let mut rec = frame.recorder();
            let (Some(sim_root), Some(draws), Some(range)) = (
                rec.arena.alloc(SimData {
                    instances: instances_rw,
                    time,
                    count: COUNT as u32,
                }),
                rec.arena.alloc_scratch::<DrawData>(COUNT),
                rec.arena.alloc(ExecRange {
                    location: 0,
                    length: 0,
                }),
            ) else {
                return;
            };
            let (Some(encode_args), Some(ring_args), Some(boundary_root)) = (
                rec.arena.alloc_scratch::<Dispatch>(1),
                rec.arena.alloc_scratch::<Draw>(1),
                rec.arena.alloc(BoundaryData {
                    mvp,
                    radius: cull_radius,
                    segments: SEGMENTS,
                }),
            ) else {
                return;
            };
            let Some(prepare_root) = rec.arena.alloc(PrepareData {
                encode: GpuMut::new(encode_args.gpu.0.get()),
                ring: GpuMut::new(ring_args.gpu.0.get()),
                count: COUNT as u32,
                group: GROUP as u32,
                segments: SEGMENTS,
                pad: 0,
            }) else {
                return;
            };
            let Some(encode_root) = rec.arena.alloc(EncodeData {
                commands: commands.handle,
                instances: instances_ro,
                draws: GpuMut::new(draws.gpu.0.get()),
                range: GpuMut::new(range.gpu.0.get()),
                mvp,
                cull_radius,
                count: COUNT as u32,
            }) else {
                return;
            };

            {
                let Some(compute) = rec.compute() else {
                    return;
                };
                compute.pipeline(&prepare);
                compute.dispatch(prepare_root.gpu, 1, 1);
                compute.pipeline(&simulate);
                compute.dispatch(sim_root.gpu, COUNT, GROUP);
                compute.barrier(
                    Stages::DISPATCH,
                    Stages::DISPATCH,
                    M4VisibilityOptions::DEVICE,
                );
                compute.pipeline(&encode);
                compute.dispatch_indirect(encode_root.gpu, encode_args.gpu.elements(), GROUP);
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
                raster.wait(
                    Stages::DISPATCH,
                    Stages::VERTEX,
                    M4VisibilityOptions::DEVICE,
                );
                raster.depth(&depth_write);
                raster.pipeline(&draw);
                raster.execute_indirect(&commands, range.gpu);
                raster.pipeline(&boundary);
                raster.draw_indirect(
                    boundary_root.gpu,
                    PrimitiveType::TriangleStrip,
                    ring_args.gpu.elements(),
                );
            }

            drop(rec);

            frame.submit(Some(&drawable));
            time += 0.016;
        });

        window.wait_for_vsync();
    }

    Ok(())
}
