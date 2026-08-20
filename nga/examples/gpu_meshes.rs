//! Indirect command buffers: a compute shader encodes the entire render pass, picking the
//! *pipeline* per object as well as the draw arguments.
//!
//! `gpu_driven` is the portable version of the same idea and shows where the line is:
//! `draw_indirect_count` lets the GPU decide how many draws there are and what they draw, but
//! all of them share the one pipeline and the one root pointer the host bound. Here every
//! command carries its own, which is why some cubes come out solid and others wireframe with
//! nothing on the CPU deciding which.
//!
//! Metal-only. Vulkan's nearest equivalent is `VK_NV_device_generated_commands`, which is not
//! in the shared API, so `Feature::IndirectCommands` answers `false` there.
//! `tests/icb.rs` is the headless half of this example.

fn main() {
    #[cfg(target_vendor = "apple")]
    if let Err(error) = commands::run() {
        eprintln!("{error}");
    }
    #[cfg(not(target_vendor = "apple"))]
    eprintln!("gpu_meshes needs indirect command buffers, which only the Metal backend has");
}

#[cfg(target_vendor = "apple")]
mod commands {
    use minwin::*;
    use nga::*;

    #[gpu_struct]
    struct Vertex {
        position: [f32; 3],
        color: [f32; 3],
    }

    #[gpu_struct(align(16))]
    struct Object {
        cell: [f32; 3],
        scale: f32,
        orbit: f32,
        phase: f32,
        style: u32,
        color: f32x4,
    }

    #[gpu_struct(align(16))]
    struct EncodeData {
        objects: Gpu<'static, Object>,
        vertices: Gpu<'static, Vertex>,
        draws: GpuMut<'static, u8>,
        commands: Commands,
        pipelines: [PipelineHandle; 2],
        viewproj: f32x4x4,
        time: f32,
        cull_radius: f32,
        count: u32,
        tri_start: u32,
        tri_count: u32,
        line_start: u32,
        line_count: u32,
    }

    const SHADERS: &[u8] = nga::shader!("gpu_meshes");

    const SIDE: usize = 6;
    const OBJECTS: u32 = (SIDE * SIDE * SIDE) as u32;
    const GROUP: usize = 64;
    /// `DrawData` in the shader: a pointer, two `float4x4` and a `float4`, all 16-aligned.
    const DRAW_DATA: usize = 16 + 64 + 64 + 16;

    /// The eight corners of a unit cube, as triangles then as lines. One vertex array serves
    /// both pipelines; the encoded command picks the range and the primitive.
    fn cube() -> Vec<Vertex> {
        const CORNERS: [[f32; 3]; 8] = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        const FACES: [[usize; 4]; 6] = [
            [0, 1, 2, 3],
            [5, 4, 7, 6],
            [4, 0, 3, 7],
            [1, 5, 6, 2],
            [3, 2, 6, 7],
            [4, 5, 1, 0],
        ];
        const EDGES: [[usize; 2]; 12] = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ];

        let mut out = Vec::with_capacity(36 + 24);
        for face in FACES {
            for &corner in &[face[0], face[1], face[2], face[0], face[2], face[3]] {
                out.push(Vertex {
                    position: CORNERS[corner],
                    color: [1.0, 1.0, 1.0],
                });
            }
        }
        for edge in EDGES {
            for corner in edge {
                out.push(Vertex {
                    position: CORNERS[corner],
                    color: [1.0, 1.0, 1.0],
                });
            }
        }
        out
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut window = create_window(
            "nga M6 — gpu meshes",
            None,
            900,
            900,
            true,
            WindowStyle::Standard,
        );

        let device = Device::new(cfg!(debug_assertions))?;
        eprintln!("device: {}", device.name());
        if !device.supports(Feature::IndirectCommands) {
            return Err(Error::new("this device has no indirect command buffers").into());
        }

        let (mut width, mut height) = window.scaled_size();
        let handle = window.ns_view as isize;
        let mut surface = Surface::new(&device, handle, width as u32, height as u32)?;

        let mut shaders = Shaders::new(&device, SHADERS)?;
        // `indirect` is what gives each pipeline a resource ID for the shader to name. Without
        // it the handle is zero and the encoded command replays nothing.
        let solid = shaders.raster(
            &RasterDesc::new(
                "solid_vertex",
                "solid_fragment",
                &[ColorTarget::new(surface.format())],
            )
            .depth(Format::D32Float)
            .indirect(),
        )?;
        let wire = shaders.raster(&RasterDesc {
            topology: Topology::LineList,
            ..RasterDesc::new(
                "wire_vertex",
                "wire_fragment",
                &[ColorTarget::new(surface.format())],
            )
            .depth(Format::D32Float)
            .indirect()
        })?;
        let encode = shaders.compute("encode", None)?;

        let commands = IndirectCommands::draws(&device, OBJECTS, false)?;

        let mesh = cube();
        let tri_count = 36u32;
        let line_count = 24u32;

        // A cube per grid cell, alternating solid and wireframe by a cheap hash so the GPU has
        // a genuine per-object pipeline choice to make.
        let mut objects = Vec::with_capacity(OBJECTS as usize);
        for index in 0..OBJECTS as usize {
            let (x, y, z) = (index % SIDE, index / SIDE % SIDE, index / (SIDE * SIDE));
            let cell = [x, y, z].map(|v| (v as f32 + 0.5) / SIDE as f32 * 1.6 - 0.8);
            let noise = ((index as f32 * 12.9898).sin() * 43758.5453).fract();
            let hue = noise * 6.2831853;
            objects.push(Object {
                cell,
                scale: 0.035 + 0.02 * noise,
                orbit: 0.4 + noise,
                phase: noise * 6.2831853,
                style: (index % 3 == 0) as u32,
                color: f32x4([
                    0.5 + 0.5 * hue.cos(),
                    0.5 + 0.5 * (hue + 2.0).cos(),
                    0.5 + 0.5 * (hue + 4.0).cos(),
                    1.0,
                ]),
            });
        }

        let storage = Arena::new(&device, 1 << 20, Memory::Device)?;
        let vertices = storage.alloc_slice(&mesh).unwrap();
        let vertices: Gpu<'static, Vertex> = Gpu::new(vertices.gpu.0.get());
        let object_list = storage.alloc_slice(&objects).unwrap();
        let object_list: Gpu<'static, Object> = Gpu::new(object_list.gpu.0.get());

        let mut heap = GpuHeap::new(&device, 32 * 1024 * 1024)?;
        let draws = heap.alloc_buffer(DRAW_DATA * OBJECTS as usize)?;
        let mut depth = heap.alloc_texture(&TextureDesc::depth(
            Format::D32Float,
            surface.width(),
            surface.height(),
        ))?;

        let mut ctx = Context::new(&device, &shaders, 256 * 1024, 1)?;

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
                    if surface.width() > 0 && surface.height() > 0 {
                        // The old depth target is still in flight, so it goes on the deferred
                        // list rather than being freed under the GPU.
                        let stale = std::mem::replace(
                            &mut depth,
                            heap.alloc_texture(&TextureDesc::depth(
                                Format::D32Float,
                                surface.width(),
                                surface.height(),
                            ))
                            .unwrap(),
                        );
                        heap.free_later(ctx.frame(), Retired::Texture(stale));
                    }
                }
                if surface.width() == 0 || surface.height() == 0 {
                    return;
                }

                let Some(drawable) = surface.acquire(&device) else {
                    resized = true;
                    return;
                };

                let aspect = surface.width() as f32 / surface.height() as f32;
                let spin = time * 0.25;
                let viewproj = f32x4x4([
                    [spin.cos() / aspect, 0.0, -spin.sin(), 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [spin.sin() / aspect, 0.0, spin.cos(), 0.0],
                    [0.0, 0.0, 0.5, 1.0],
                ]);

                heap.retire(ctx.completed());
                let mut frame = ctx.begin_frame().unwrap();
                let mut rec = frame.recorder();

                let root = rec
                    .arena()
                    .alloc(EncodeData {
                        objects: object_list,
                        vertices,
                        draws: GpuMut::new(draws.address()),
                        commands: commands.handle(),
                        pipelines: [solid.handle(), wire.handle()],
                        viewproj,
                        time,
                        cull_radius: 0.9 + 0.5 * (time * 0.4).sin(),
                        count: OBJECTS,
                        tri_start: 0,
                        tri_count,
                        line_start: tri_count,
                        line_count,
                    })
                    .unwrap()
                    .gpu;

                {
                    let compute = rec.compute();
                    // A slot the encode pass does not claim keeps whatever it last held, so the
                    // buffer is cleared first. A reset slot replays as nothing, which is what
                    // makes "execute everything" safe without a GPU-side count.
                    compute.reset(&commands, commands.all());
                    compute.pipeline(encode);
                    compute.dispatch(root, OBJECTS as usize, GROUP);
                    compute.barrier(Stage::COMPUTE, Stage::INDIRECT, Hazard::DRAW_ARGUMENTS);
                }
                {
                    let pass = Pass::new()
                        .color(0, &drawable, Some([0.02, 0.02, 0.04, 1.0]))
                        .depth(&depth, Some(1.0), false);
                    let raster = rec.raster(&pass, surface.width(), surface.height());
                    raster.depth(&DepthState::new(CompareOp::LessEqual, true));
                    // Every command the shader wrote, pipeline choice included.
                    raster.execute(&commands, commands.all());
                }

                drop(rec);
                frame.submit(Some(&drawable)).unwrap();
                surface.present(&device, &drawable);
                time += 1.0 / 60.0;
            });
        }

        device.wait_idle()?;
        heap.retire(u64::MAX);
        heap.free_buffer(draws);
        heap.free_texture(depth);
        surface.destroy(&device);
        Ok(())
    }
}
