use minmetal::*;
use minwin::*;
use nga::*;
use std::collections::HashMap;

#[gpu_struct]
struct Vertex {
    position: [f32; 3],
}

#[gpu_struct(align(16))]
struct Meshlet {
    bounds: f32x4,
    cone: f32x4,
    vertex_offset: u32,
    triangle_offset: u32,
    vertex_count: u32,
    triangle_count: u32,
}

#[gpu_struct]
struct Model {
    meshlet_start: u32,
    meshlet_count: u32,
}

#[gpu_struct(align(16))]
struct Object {
    color: f32x4,
    cell: f32x3,
    scale: f32,
    orbit: f32,
    phase: f32,
    model: u32,
}

#[gpu_struct(align(16))]
struct Payload {
    mvp: f32x4x4,
    model: f32x4x4,
    color: f32x4,
    meshlets: [u32; 32],
}

#[gpu_struct(align(16))]
struct Scene {
    objects: Gpu<'static, Object>,
    models: Gpu<'static, Model>,
    meshlets: Gpu<'static, Meshlet>,
    meshlet_vertices: Gpu<'static, u32>,
    meshlet_triangles: Gpu<'static, u32>,
    vertices: Gpu<'static, Vertex>,
    viewproj: f32x4x4,
    planes: [f32x4; 6],
    camera: f32x3,
    time: f32,
    debug: f32,
    count: u32,
    pad: u32,
}

const SHADERS: &str = include_str!("../shaders/meshlets.metal");

const MAX_VERTICES: usize = 64;
const MAX_TRIANGLES: usize = 126;
const MAX_MESHLETS: usize = 32;
const MESH_GROUP: usize = 128;
const SIDE: usize = 8;
const OBJECTS: usize = SIDE * SIDE * SIDE;
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window(
        "nga — mesh shader meshlets",
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
        msl::header!(Vertex, Meshlet, Model, Object, Payload, Scene)
    );
    msl::assert_matches(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/generated/meshlets.metal"
        ),
        &format!("{MSL_PRELUDE}{source}"),
    );

    let shaders = Shaders::new(&device, &source)?;
    let pipeline = shaders.mesh(&MeshDesc::new(
        "meshlet_object",
        "meshlet_mesh",
        "meshlet_fragment",
        &[ColorTarget::new(PixelFormat::Bgra8Unorm)],
        size_of::<Payload>(),
    ))?;
    let depth_write = depth_state(&device, CompareFunction::Less, true)?;

    let phi = (1.0 + 5.0f32.sqrt()) * 0.5;
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut meshlets: Vec<Meshlet> = Vec::new();
    let mut meshlet_vertices: Vec<u32> = Vec::new();
    let mut meshlet_triangles: Vec<u32> = Vec::new();
    let mut models: Vec<Model> = Vec::new();

    for subdivisions in 0..4 {
        let base_vertex = vertices.len() as u32;
        let mut positions: Vec<[f32; 3]> = [
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
        ]
        .into_iter()
        .map(|p| {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            [p[0] / len, p[1] / len, p[2] / len]
        })
        .collect();
        let mut triangles: Vec<[u32; 3]> = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
        .into();

        for _ in 0..subdivisions {
            let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
            let mut split = Vec::with_capacity(triangles.len() * 4);
            for t in &triangles {
                let mut edge = |a: u32, b: u32, positions: &mut Vec<[f32; 3]>| {
                    let key = (a.min(b), a.max(b));
                    *midpoints.entry(key).or_insert_with(|| {
                        let (p, q) = (positions[a as usize], positions[b as usize]);
                        let m = [
                            (p[0] + q[0]) * 0.5,
                            (p[1] + q[1]) * 0.5,
                            (p[2] + q[2]) * 0.5,
                        ];
                        let len = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
                        positions.push([m[0] / len, m[1] / len, m[2] / len]);
                        positions.len() as u32 - 1
                    })
                };
                let ab = edge(t[0], t[1], &mut positions);
                let bc = edge(t[1], t[2], &mut positions);
                let ca = edge(t[2], t[0], &mut positions);
                split.extend([[t[0], ab, ca], [t[1], bc, ab], [t[2], ca, bc], [ab, bc, ca]]);
            }
            triangles = split;
        }

        let normal = |t: &[u32; 3]| {
            let (a, b, c) = (
                positions[t[0] as usize],
                positions[t[1] as usize],
                positions[t[2] as usize],
            );
            let (u, v) = (
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
            );
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            [n[0] / len, n[1] / len, n[2] / len]
        };
        let first = normal(&triangles[0]);
        let centroid = positions[triangles[0][0] as usize];
        if first[0] * centroid[0] + first[1] * centroid[1] + first[2] * centroid[2] < 0.0 {
            for t in &mut triangles {
                t.swap(1, 2);
            }
        }

        let morton = |t: &[u32; 3]| {
            let mut key = 0u32;
            for axis in 0..3 {
                let c = (0..3).map(|i| positions[t[i] as usize][axis]).sum::<f32>() / 3.0;
                let mut bits = ((c * 0.5 + 0.5).clamp(0.0, 1.0) * 1023.0) as u32;
                for bit in 0..10 {
                    key |= (bits & 1) << (bit * 3 + axis);
                    bits >>= 1;
                }
            }
            key
        };
        triangles.sort_by_key(morton);

        let mut groups: Vec<(Vec<u32>, Vec<u32>)> = vec![(Vec::new(), Vec::new())];
        for t in &triangles {
            let group = groups.last_mut().unwrap();
            let missing = t.iter().filter(|g| !group.0.contains(g)).count();
            if group.0.len() + missing > MAX_VERTICES || group.1.len() >= MAX_TRIANGLES {
                groups.push((Vec::new(), Vec::new()));
            }
            let group = groups.last_mut().unwrap();
            let mut local = [0u32; 3];
            for (slot, &global) in local.iter_mut().zip(t) {
                *slot = match group.0.iter().position(|&v| v == global) {
                    Some(index) => index as u32,
                    None => {
                        group.0.push(global);
                        group.0.len() as u32 - 1
                    }
                };
            }
            group.1.push(local[0] | local[1] << 8 | local[2] << 16);
        }

        let meshlet_start = meshlets.len() as u32;
        for (group_vertices, group_triangles) in &groups {
            let mut center = [0.0f32; 3];
            for &v in group_vertices {
                let p = positions[v as usize];
                center = [center[0] + p[0], center[1] + p[1], center[2] + p[2]];
            }
            let n = group_vertices.len() as f32;
            center = [center[0] / n, center[1] / n, center[2] / n];
            let mut radius = 0.0f32;
            for &v in group_vertices {
                let p = positions[v as usize];
                let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                radius = radius.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            }

            let normals: Vec<[f32; 3]> = group_triangles
                .iter()
                .map(|packed| {
                    normal(&[
                        group_vertices[(packed & 0xff) as usize],
                        group_vertices[(packed >> 8 & 0xff) as usize],
                        group_vertices[(packed >> 16 & 0xff) as usize],
                    ])
                })
                .collect();
            let mut axis = [0.0f32; 3];
            for n in &normals {
                axis = [axis[0] + n[0], axis[1] + n[1], axis[2] + n[2]];
            }
            let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            axis = [axis[0] / len, axis[1] / len, axis[2] / len];
            let spread = normals
                .iter()
                .map(|n| axis[0] * n[0] + axis[1] * n[1] + axis[2] * n[2])
                .fold(1.0f32, f32::min);
            let cutoff = match spread > 0.0 {
                true => (1.0 - spread * spread).sqrt(),
                false => 1.0,
            };

            meshlets.push(Meshlet {
                bounds: f32x4([center[0], center[1], center[2], radius]),
                cone: f32x4([axis[0], axis[1], axis[2], cutoff]),
                vertex_offset: meshlet_vertices.len() as u32,
                triangle_offset: meshlet_triangles.len() as u32,
                vertex_count: group_vertices.len() as u32,
                triangle_count: group_triangles.len() as u32,
            });
            meshlet_vertices.extend(group_vertices.iter().map(|v| v + base_vertex));
            meshlet_triangles.extend_from_slice(group_triangles);
        }

        vertices.extend(positions.iter().map(|&p| Vertex { position: p }));
        models.push(Model {
            meshlet_start,
            meshlet_count: meshlets.len() as u32 - meshlet_start,
        });
        eprintln!(
            "model {}: {} tris, {} verts, {} meshlets",
            models.len() - 1,
            triangles.len(),
            positions.len(),
            groups.len()
        );
    }

    assert!(
        models
            .iter()
            .all(|m| m.meshlet_count as usize <= MAX_MESHLETS)
    );

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
            cell: f32x3([
                axis(i % SIDE),
                axis((i / SIDE) % SIDE),
                axis(i / (SIDE * SIDE)),
            ]),
            scale: 0.05 + 0.035 * n,
            orbit: 0.3 + n,
            phase: n * 6.2831853,
            model: (i % 4) as u32,
        });
    }

    let mut vertex_buffer = ctx.heap.alloc_buffer(size_of_val(vertices.as_slice()))?;
    vertex_buffer.write(0, &vertices);
    let mut meshlet_buffer = ctx.heap.alloc_buffer(size_of_val(meshlets.as_slice()))?;
    meshlet_buffer.write(0, &meshlets);
    let mut meshlet_vertex_buffer = ctx
        .heap
        .alloc_buffer(size_of_val(meshlet_vertices.as_slice()))?;
    meshlet_vertex_buffer.write(0, &meshlet_vertices);
    let mut meshlet_triangle_buffer = ctx
        .heap
        .alloc_buffer(size_of_val(meshlet_triangles.as_slice()))?;
    meshlet_triangle_buffer.write(0, &meshlet_triangles);
    let mut model_buffer = ctx.heap.alloc_buffer(size_of_val(models.as_slice()))?;
    model_buffer.write(0, &models);
    let mut object_buffer = ctx.heap.alloc_buffer(size_of_val(objects.as_slice()))?;
    object_buffer.write(0, &objects);

    eprintln!(
        "{OBJECTS} objects, {} meshlets, {} triangles resident",
        meshlets.len(),
        meshlet_triangles.len()
    );

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
    let mut debug = false;
    let mut space_was_down = false;

    while window.open() {
        let _pool = AutoreleasePool::new();

        if window.pressed(Key::Escape) {
            window.close();
        }
        let space_is_down = window.pressed(Key::Space);
        if space_is_down && !space_was_down {
            debug = !debug;
        }
        space_was_down = space_is_down;

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

            let row = |r: usize| {
                [
                    viewproj[0][r],
                    viewproj[1][r],
                    viewproj[2][r],
                    viewproj[3][r],
                ]
            };
            let add = |a: [f32; 4], b: [f32; 4], s: f32| {
                [
                    a[0] + s * b[0],
                    a[1] + s * b[1],
                    a[2] + s * b[2],
                    a[3] + s * b[3],
                ]
            };
            let (x, y, z, w) = (row(0), row(1), row(2), row(3));
            let mut planes = [
                add(w, x, 1.0),
                add(w, x, -1.0),
                add(w, y, 1.0),
                add(w, y, -1.0),
                add(z, z, 0.0),
                add(w, z, -1.0),
            ];
            for p in &mut planes {
                let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                *p = [p[0] / len, p[1] / len, p[2] / len, p[3] / len];
            }

            let mut frame = ctx.begin_frame();
            let mut rec = frame.recorder();
            let Some(scene) = rec.arena.alloc(Scene {
                objects: Gpu::new(object_buffer.gpu),
                models: Gpu::new(model_buffer.gpu),
                meshlets: Gpu::new(meshlet_buffer.gpu),
                meshlet_vertices: Gpu::new(meshlet_vertex_buffer.gpu),
                meshlet_triangles: Gpu::new(meshlet_triangle_buffer.gpu),
                vertices: Gpu::new(vertex_buffer.gpu),
                viewproj: f32x4x4(viewproj),
                planes: planes.map(f32x4),
                camera: f32x3(eye),
                time,
                debug: debug as u32 as f32,
                count: OBJECTS as u32,
                pad: 0,
            }) else {
                return;
            };

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
                raster.depth(&depth_write);
                raster.pipeline(&pipeline);
                raster.draw_mesh(scene.gpu, OBJECTS, MAX_MESHLETS, MESH_GROUP);
            }

            drop(rec);

            frame.submit(Some(&drawable));
            time += 0.016;
        });

        window.wait_for_vsync();
    }

    Ok(())
}
