//! Tile shading: a field of triangles, then a compute function that vignettes and posterises
//! the imageblock before it is ever written out to memory.
//!
//! Apple-only, and permanently so. An imageblock is tile-based deferred hardware showing
//! through, not an API Vulkan is missing. `Feature::TileShaders` is the runtime spelling of
//! that, and `tests/tile.rs` is the headless half of this example.

fn main() {
    #[cfg(target_vendor = "apple")]
    if let Err(error) = tiles::run() {
        eprintln!("{error}");
    }
    #[cfg(not(target_vendor = "apple"))]
    eprintln!("tbdr needs tile shaders, which only the Metal backend has and only Apple hardware");
}

#[cfg(target_vendor = "apple")]
mod tiles {
    use minwin::*;
    use nga::*;

    #[gpu_mod]
    mod shader {
        use nga::*;

        pub struct Vertex {
            pub position: [f32; 3],
            pub color: [f32; 3],
        }

        #[repr(C, align(16))]
        pub struct Data {
            pub vertices: Gpu<[Vertex]>,
            pub mvp: f32x4x4,
        }

        pub struct Vignette {
            pub size: f32x2,
            pub tile: f32x2,
            pub strength: f32,
            pub levels: f32,
        }

        pub struct VertexOut {
            pub position: f32x4,
            pub color: f32x3,
        }

        #[derive(Default)]
        pub struct TileColor {
            pub color: f32x4,
        }

        #[vertex]
        fn tbdr_vertex(root: Gpu<Data>, #[vertex_id] vid: u32) -> VertexOut {
            let source = root.vertices[vid];
            VertexOut {
                position: root.mvp.mul(f32x4::new(
                    source.position[0],
                    source.position[1],
                    source.position[2],
                    1.0,
                )),
                color: f32x3::new(source.color[0], source.color[1], source.color[2]),
            }
        }

        #[fragment]
        fn tbdr_fragment(input: VertexOut) -> f32x4 {
            f32x4::new(input.color.x, input.color.y, input.color.z, 1.0)
        }

        #[tile(TileColor)]
        fn tbdr_vignette(
            root: Gpu<Vignette>,
            #[thread_id_in_group] tid: u32x3,
            #[group_id] group: u32x3,
        ) {
            let at = u32x2::new(tid.x, tid.y);
            let pixel = f32x2::new(group.x as f32, group.y as f32) * root.tile
                + f32x2::new(tid.x as f32, tid.y as f32);
            let offset = pixel / root.size - f32x2::splat(0.5);
            let mut texel: TileColor = tile_read(at);
            let mut color = texel.color.xyz();
            color = (color * root.levels).floor() / root.levels;
            color = color * (1.0 - root.strength * offset.dot(offset));
            texel.color = f32x4::new(color.x, color.y, color.z, texel.color.w);
            tile_write(texel, at);
        }
    }

    use shader::*;

    /// One `TileColor` in the shader: a `float4`.
    const IMAGEBLOCK_BYTES: usize = 16;
    /// The tile the imageblock covers. 32x32 is the largest Apple silicon accepts for a
    /// four-byte-per-sample colour target.
    const TILE: u32 = 16;

    const ROWS: usize = 12;
    const COLUMNS: usize = 12;

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut window =
            create_window("nga M6 — tbdr", None, 900, 900, true, WindowStyle::Standard);

        let device = Device::new(cfg!(debug_assertions))?;
        eprintln!("device: {}", device.name());
        if !device.supports(Feature::TileShaders) {
            return Err(Error::new("this device has no tile pipeline").into());
        }

        let (mut width, mut height) = window.scaled_size();
        let mut surface = device.surface(window.handle(), width as u32, height as u32)?;

        let mut shaders = device.shaders(shader::MODULE)?;
        let draw = shaders
            .raster(shader::TBDR_VERTEX, shader::TBDR_FRAGMENT)
            .color(surface.format())
            .build()?;
        let vignette = shaders.tile(&TileDesc::new("tbdr_vignette", &[surface.format()]))?;

        // A triangle per grid cell, each a different hue. Built once: only the transform moves.
        let mut mesh = Vec::with_capacity(ROWS * COLUMNS * 3);
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let x = (column as f32 + 0.5) / COLUMNS as f32 * 2.0 - 1.0;
                let y = (row as f32 + 0.5) / ROWS as f32 * 2.0 - 1.0;
                let hue = (row * COLUMNS + column) as f32 / (ROWS * COLUMNS) as f32;
                let color = [
                    0.5 + 0.5 * (6.2831853 * hue).cos(),
                    0.5 + 0.5 * (6.2831853 * hue + 2.0).cos(),
                    0.5 + 0.5 * (6.2831853 * hue + 4.0).cos(),
                ];
                let radius = 0.7 / COLUMNS as f32;
                for corner in 0..3 {
                    let angle = corner as f32 * 2.0943951;
                    mesh.push(Vertex {
                        position: [x + angle.cos() * radius, y + angle.sin() * radius, 0.5],
                        color,
                    });
                }
            }
        }

        let geometry = Arena::new(&device, 1 << 20, Memory::Device)?;
        let vertices = geometry.alloc_slice(&mesh).unwrap();
        let vertices: Gpu<[Vertex]> = vertices.gpu;
        let count = mesh.len() as u32;

        let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

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

                let aspect = surface.width() as f32 / surface.height() as f32;
                let spin = time * 0.4;
                let mvp = f32x4x4([
                    [spin.cos() / aspect, spin.sin(), 0.0, 0.0],
                    [-spin.sin() / aspect, spin.cos(), 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]);

                let mut frame = ctx.begin_frame().unwrap();
                let rec = frame.recorder();

                // The tile size and the imageblock layout belong to the *pass*: they decide
                // how the hardware splits the render target, so they cannot change per draw.
                let pass = Pass::new()
                    .color(0, &drawable, Some([0.02, 0.02, 0.04, 1.0]))
                    .tile(TILE, TILE, IMAGEBLOCK_BYTES);
                let raster = rec.raster(&pass);
                raster.draw(draw, Data { vertices, mvp }, 0..count).unwrap();

                // Same encoder, same tile memory. On any other backend this would be a
                // second pass over a stored texture.
                raster
                    .dispatch_tile(
                        vignette,
                        Vignette {
                            size: f32x2([surface.width() as f32, surface.height() as f32]),
                            tile: f32x2([TILE as f32, TILE as f32]),
                            // Swings so the effect is visibly the tile pass and not the geometry.
                            strength: 0.9 + 0.6 * (time * 0.7).sin(),
                            levels: 6.0,
                        },
                        TILE,
                        TILE,
                    )
                    .unwrap();

                frame.submit(Some(&drawable)).unwrap();
                surface.present(&device, &drawable);
                time += 1.0 / 60.0;
            });
        }

        device.wait_idle()?;
        Ok(())
    }
}
