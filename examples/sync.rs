use minmetal::*;
use minwin::*;
use nga::*;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Instance {
    center: [f32; 2],
    angle: f32,
    scale: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
struct SimData {
    instances: GpuMut<'static, Instance>,
    time: f32,
    count: u32,
}

#[repr(C)]
struct BuildData {
    instances: Gpu<'static, Instance>,
    vertices: GpuMut<'static, Vertex>,
    count: u32,
    pad: u32,
}

#[repr(C, align(16))]
struct DrawData {
    vertices: Gpu<'static, Vertex>,
    mvp: f32x4x4,
}

const SHADERS: &str = r#"
struct Instance {
    float2 center;
    float angle;
    float scale;
    float4 color;
};

struct Vertex {
    packed_float3 position;
    packed_float3 color;
};

struct SimData {
    device Instance* instances;
    float time;
    uint count;
};

struct BuildData {
    device const Instance* instances;
    device Vertex* vertices;
    uint count;
    uint pad;
};

struct DrawData {
    device const Vertex* vertices;
    float4x4 mvp;
};

kernel void simulate(device const SimData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    float f = float(i) / float(data->count);
    float orbit = 0.25 + 0.65 * fract(f * 7.0);
    float phase = f * 6.2831853 + data->time * (0.2 + 0.9 * fract(f * 5.0));

    Instance inst;
    inst.center = float2(cos(phase), sin(phase)) * orbit;
    inst.angle = phase * 2.0 + data->time;
    inst.scale = 0.015 + 0.045 * fract(f * 3.0);
    inst.color = float4(0.5 + 0.5 * cos(6.2831853 * f + float3(0.0, 2.0, 4.0)), 1.0);
    data->instances[i] = inst;
}

kernel void build_vertices(device const BuildData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    Instance inst = data->instances[i];
    device Vertex* out = data->vertices + i * 3;
    for (uint k = 0; k < 3; ++k) {
        float a = inst.angle + float(k) * 2.0943951;
        float2 p = inst.center + float2(cos(a), sin(a)) * inst.scale;
        out[k].position = float3(p, 0.5);
        out[k].color = inst.color.rgb;
    }
}

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut vertex_main(device const DrawData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 1.0);
    out.color = v.color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
"#;

const COUNT: usize = 1024;
const GROUP: usize = 64;
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window("nga M4 — sync", None, 900, 700, true, WindowStyle::Standard);

    let device = Device::required_system_default()?;
    eprintln!("device: {}", device.name());

    let mut ctx = Context::new(&device, 64 * 1024, 64 * 1024 * 1024)?;

    let shaders = Shaders::new(&device, SHADERS)?;
    let simulate = shaders.compute("simulate")?;
    let build = shaders.compute("build_vertices")?;
    let draw = shaders.raster(&RasterDesc::new(
        "vertex_main",
        "fragment_main",
        &[ColorTarget::new(PixelFormat::Bgra8Unorm)],
    ))?;
    let depth_write = depth_state(&device, CompareFunction::Less, true)?;

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
            let Some(sim_root) = frame.arena.alloc(SimData {
                instances: instances_rw,
                time,
                count: COUNT as u32,
            }) else {
                return;
            };
            let Some(build_root) = frame.arena.alloc(BuildData {
                instances: instances_ro,
                vertices: vertices_rw,
                count: COUNT as u32,
                pad: 0,
            }) else {
                return;
            };
            let Some(draw_root) = frame.arena.alloc(DrawData {
                vertices: vertices_ro,
                mvp,
            }) else {
                return;
            };

            {
                let Some(compute) = frame.compute() else {
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
            }

            let pass = Pass::new()
                .color(0, &target, Some(ClearColor::new(0.03, 0.03, 0.05, 1.0)))
                .depth(&depth.as_ref().unwrap().0.texture, Some(1.0));
            let viewport = Viewport::new(
                0.0,
                0.0,
                drawable_width as f64,
                drawable_height as f64,
                0.0,
                1.0,
            );

            {
                let Some(raster) = frame.raster(&pass, viewport) else {
                    return;
                };
                raster.wait(
                    Stages::DISPATCH,
                    Stages::VERTEX,
                    M4VisibilityOptions::DEVICE,
                );
                raster.pipeline(&draw);
                raster.depth(&depth_write);
                raster.draw(draw_root.gpu, PrimitiveType::Triangle, 0, COUNT * 3);
            }

            frame.submit(Some(&drawable));
            time += 0.016;
        });

        window.wait_for_vsync();
    }

    Ok(())
}
