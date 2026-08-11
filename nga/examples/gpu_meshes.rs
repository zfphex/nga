use minmetal::*;
use minwin::*;
use nga::*;

#[gpu_struct]
struct Vertex {
    position: [f32; 3],
}

#[gpu_struct]
struct Mesh {
    tri_start: u32,
    tri_count: u32,
    line_start: u32,
    line_count: u32,
    base_vertex: u32,
}

#[gpu_struct(align(16))]
struct Object {
    color: f32x4,
    cell: f32x3,
    scale: f32,
    orbit: f32,
    phase: f32,
    mesh: u32,
    style: u32,
}

#[gpu_struct(align(16))]
struct DrawData {
    vertices: Gpu<'static, Vertex>,
    mvp: f32x4x4,
    model: f32x4x4,
    color: f32x4,
}

#[gpu_struct(align(16))]
struct EncodeData {
    commands: Commands,
    pipelines: Gpu<'static, Pipeline>,
    objects: Gpu<'static, Object>,
    meshes: Gpu<'static, Mesh>,
    vertices: Gpu<'static, Vertex>,
    indices: Gpu<'static, u32>,
    draws: GpuMut<'static, DrawData>,
    range: GpuMut<'static, ExecRange>,
    viewproj: f32x4x4,
    time: f32,
    cull_radius: f32,
    count: u32,
    pad: u32,
}

const SHADERS: &str = include_str!("../shaders/gpu_meshes.metal");

const SIDE: usize = 12;
const OBJECTS: usize = SIDE * SIDE * SIDE;
const GROUP: usize = 64;
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window(
        "nga — gpu driven heterogeneous meshes",
        None,
        1000,
        800,
        true,
        WindowStyle::Standard,
    );

    let device = Device::required_system_default()?;
    eprintln!("device: {}", device.name());

    let mut ctx = Context::new(&device, 4 * 1024 * 1024, 64 * 1024 * 1024, 1024, 1)?;

    let source = format!(
        "{}\n{SHADERS}",
        msl::header!(Vertex, Mesh, Object, DrawData, EncodeData)
    );
    msl::dump("gpu_meshes", &format!("{MSL_PRELUDE}{source}"));

    let shaders = Shaders::new(&device, &source)?;
    let encode = shaders.compute("encode", None)?;
    let solid = shaders.raster(&RasterDesc {
        indirect: true,
        ..RasterDesc::new(
            "solid_vertex",
            "solid_fragment",
            &[ColorTarget::new(PixelFormat::Bgra8Unorm)],
        )
    })?;
    let wire = shaders.raster(&RasterDesc {
        indirect: true,
        ..RasterDesc::new(
            "wire_vertex",
            "wire_fragment",
            &[ColorTarget {
                blend: BlendMode::Fixed(Blend::ALPHA),
                ..ColorTarget::new(PixelFormat::Bgra8Unorm)
            }],
        )
    })?;
    let depth_write = depth_state(&device, CompareFunction::Less, true)?;

    let phi = (1.0 + 5.0f32.sqrt()) * 0.5;
    let tetrahedron: (&[[f32; 3]], &[u32]) = (
        &[[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
        &[0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2],
    );
    let cube_positions: Vec<[f32; 3]> = (0..8)
        .map(|i: u32| {
            [
                (i & 1) as f32 * 2.0 - 1.0,
                ((i >> 1) & 1) as f32 * 2.0 - 1.0,
                ((i >> 2) & 1) as f32 * 2.0 - 1.0,
            ]
        })
        .collect();
    let cube: (&[[f32; 3]], &[u32]) = (
        &cube_positions,
        &[
            0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6,
            4, 1, 5, 7, 1, 7, 3,
        ],
    );
    let octahedron: (&[[f32; 3]], &[u32]) = (
        &[
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        &[0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5],
    );
    let icosahedron: (&[[f32; 3]], &[u32]) = (
        &[
            [-1.0, phi, 0.0],
            [1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [1.0, -phi, 0.0],
            [0.0, -1.0, phi],
            [0.0, 1.0, phi],
            [0.0, -1.0, -phi],
            [0.0, 1.0, -phi],
            [phi, 0.0, -1.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
        ],
        &[
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7,
            6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10,
            8, 6, 7, 9, 8, 1,
        ],
    );

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut meshes: Vec<Mesh> = Vec::new();
    for (positions, tris) in [tetrahedron, cube, octahedron, icosahedron] {
        let base_vertex = vertices.len() as u32;
        for p in positions {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            vertices.push(Vertex {
                position: [p[0] / len, p[1] / len, p[2] / len],
            });
        }
        let tri_start = indices.len() as u32;
        indices.extend_from_slice(tris);
        let mut edges: Vec<(u32, u32)> = tris
            .chunks(3)
            .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        edges.sort_unstable();
        edges.dedup();
        let line_start = indices.len() as u32;
        indices.extend(edges.iter().flat_map(|&(a, b)| [a, b]));
        meshes.push(Mesh {
            tri_start,
            tri_count: tris.len() as u32,
            line_start,
            line_count: edges.len() as u32 * 2,
            base_vertex,
        });
        eprintln!("mesh {}: {} tris, {} edges", meshes.len() - 1, tris.len() / 3, edges.len());
    }

    let mut objects: Vec<Object> = Vec::new();
    for i in 0..OBJECTS {
        let n = ((i as f32 * 12.9898).sin() * 43758.5453).fract();
        let axis = |v: usize| (v as f32 / (SIDE - 1) as f32 * 2.0 - 1.0) * 1.15;
        objects.push(Object {
            color: f32x4([
                0.5 + 0.5 * (6.2831853 * n).cos(),
                0.5 + 0.5 * (6.2831853 * n + 2.0).cos(),
                0.5 + 0.5 * (6.2831853 * n + 4.0).cos(),
                1.0,
            ]),
            cell: f32x3([axis(i % SIDE), axis((i / SIDE) % SIDE), axis(i / (SIDE * SIDE))]),
            scale: 0.035 + 0.025 * n,
            orbit: 0.3 + n,
            phase: n * 6.2831853,
            mesh: (i % 4) as u32,
            style: (i % 5 == 0) as u32,
        });
    }

    let mut vertex_buffer = ctx.heap.alloc_buffer(size_of_val(vertices.as_slice()))?;
    vertex_buffer.write(0, &vertices);
    let mut index_buffer = ctx.heap.alloc_buffer(size_of_val(indices.as_slice()))?;
    index_buffer.write(0, &indices);
    let mut mesh_buffer = ctx.heap.alloc_buffer(size_of_val(meshes.as_slice()))?;
    mesh_buffer.write(0, &meshes);
    let mut object_buffer = ctx.heap.alloc_buffer(size_of_val(objects.as_slice()))?;
    object_buffer.write(0, &objects);
    let mut pipeline_buffer = ctx.heap.alloc_buffer(2 * size_of::<Pipeline>())?;
    pipeline_buffer.write(
        0,
        &[
            Pipeline(solid.gpu_resource_id()?),
            Pipeline(wire.gpu_resource_id()?),
        ],
    );
    ctx.resident(&Allocation { raw: solid.raw })?;
    ctx.resident(&Allocation { raw: wire.raw })?;

    let desc = IndirectCommandBufferDescriptor::new();
    desc.set_command_types(IndirectCommandType::DRAW_INDEXED);
    desc.set_inherit_pipeline_state(false);
    desc.set_inherit_buffers(false);
    desc.set_max_vertex_buffer_bind_count(1);
    let commands = IndirectCommands::new(&device, &desc, OBJECTS)?;
    ctx.resident(&Allocation::from_indirect_command_buffer(&commands.icb))?;

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

            let eye = [
                4.0 * (time * 0.15).cos(),
                1.6 + 0.8 * (time * 0.11).sin(),
                4.0 * (time * 0.15).sin(),
            ];
            let len = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
            let fwd = [-eye[0] / len, -eye[1] / len, -eye[2] / len];
            let rl = (fwd[2] * fwd[2] + fwd[0] * fwd[0]).sqrt();
            let right = [-fwd[2] / rl, 0.0, fwd[0] / rl];
            let up = [
                right[1] * fwd[2] - right[2] * fwd[1],
                right[2] * fwd[0] - right[0] * fwd[2],
                right[0] * fwd[1] - right[1] * fwd[0],
            ];
            let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let view = [
                [right[0], up[0], -fwd[0], 0.0],
                [right[1], up[1], -fwd[1], 0.0],
                [right[2], up[2], -fwd[2], 0.0],
                [-dot(right, eye), -dot(up, eye), dot(fwd, eye), 1.0],
            ];

            let aspect = drawable_width as f32 / drawable_height as f32;
            let focal = 1.0 / (0.5f32).tan();
            let (near, far) = (0.1f32, 20.0f32);
            let mut viewproj = [[0.0f32; 4]; 4];
            for (out, v) in viewproj.iter_mut().zip(view) {
                *out = [
                    v[0] * focal / aspect,
                    v[1] * focal,
                    (v[2] * far + v[3] * near * far) / (near - far),
                    -v[2],
                ];
            }

            let cull_radius = 1.1 + 0.9 * (time * 0.6).sin();

            let mut frame = ctx.begin_frame();
            let mut rec = frame.recorder();
            let (Some(draws), Some(range)) = (
                rec.arena.alloc_scratch::<DrawData>(OBJECTS),
                rec.arena.alloc(ExecRange {
                    location: 0,
                    length: 0,
                }),
            ) else {
                return;
            };
            let Some(encode_root) = rec.arena.alloc(EncodeData {
                commands: commands.handle,
                pipelines: Gpu::new(pipeline_buffer.gpu),
                objects: Gpu::new(object_buffer.gpu),
                meshes: Gpu::new(mesh_buffer.gpu),
                vertices: Gpu::new(vertex_buffer.gpu),
                indices: Gpu::new(index_buffer.gpu),
                draws: GpuMut::new(draws.gpu.0.get()),
                range: GpuMut::new(range.gpu.0.get()),
                viewproj: f32x4x4(viewproj),
                time,
                cull_radius,
                count: OBJECTS as u32,
                pad: 0,
            }) else {
                return;
            };

            {
                let Some(compute) = rec.compute() else {
                    return;
                };
                compute.pipeline(&encode);
                compute.dispatch(encode_root.gpu, OBJECTS, GROUP);
            }

            let pass = Pass::new()
                .color(0, &target, Some(ClearColor::new(0.02, 0.02, 0.04, 1.0)))
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
                let Some(raster) = rec.raster(&pass, viewport) else {
                    return;
                };
                raster.wait(
                    Stages::DISPATCH,
                    Stages::VERTEX,
                    M4VisibilityOptions::DEVICE,
                );
                raster.depth(&depth_write);
                raster.pipeline(&solid);
                raster.execute_indirect(&commands, range.gpu);
            }

            drop(rec);

            frame.submit(Some(&drawable));
            time += 0.016;
        });

        window.wait_for_vsync();
    }

    Ok(())
}
