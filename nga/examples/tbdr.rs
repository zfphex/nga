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

    #[gpu_struct]
    struct Vertex {
        position: [f32; 3],
        color: [f32; 3],
    }

    #[gpu_struct(align(16))]
    struct Data {
        vertices: Gpu<'static, Vertex>,
        mvp: f32x4x4,
    }

    #[gpu_struct]
    struct Vignette {
        size: f32x2,
        tile: f32x2,
        strength: f32,
        levels: f32,
    }

    const SHADERS: &[u8] = nga::shader!("tbdr");

    /// One `TileColor` in the shader: a `half4`.
    const IMAGEBLOCK_BYTES: usize = 8;
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
        // The platform window handle: an `HWND` on Windows, the `NSView` on macOS.
        let handle = window.ns_view as isize;
        let mut surface = Surface::new(&device, handle, width as u32, height as u32)?;

        let mut shaders = Shaders::new(&device, SHADERS)?;
        let draw = shaders.raster(&RasterDesc::new(
            "tbdr_vertex",
            "tbdr_fragment",
            &[ColorTarget::new(surface.format())],
        ))?;
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
        let vertices: Gpu<'static, Vertex> = Gpu::new(vertices.gpu.0.get());
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
                let mut rec = frame.recorder();

                let draw_root = rec.arena().alloc(Data { vertices, mvp }).unwrap().gpu;
                let vignette_root = rec
                    .arena()
                    .alloc(Vignette {
                        size: f32x2([surface.width() as f32, surface.height() as f32]),
                        tile: f32x2([TILE as f32, TILE as f32]),
                        // Swings so the effect is visibly the tile pass and not the geometry.
                        strength: 0.9 + 0.6 * (time * 0.7).sin(),
                        levels: 6.0,
                    })
                    .unwrap()
                    .gpu;

                {
                    // The tile size and the imageblock layout belong to the *pass*: they decide
                    // how the hardware splits the render target, so they cannot change per draw.
                    let pass = Pass::new()
                        .color(0, &drawable, Some([0.02, 0.02, 0.04, 1.0]))
                        .tile(TILE, TILE, IMAGEBLOCK_BYTES);
                    let raster = rec.raster(&pass, surface.width(), surface.height());
                    raster.pipeline(draw);
                    raster.draw(draw_root, 0, count);

                    // Same encoder, same tile memory. On any other backend this would be a
                    // second pass over a stored texture.
                    raster.pipeline(vignette);
                    raster.dispatch_tile(vignette_root, TILE, TILE);
                }

                drop(rec);
                frame.submit(Some(&drawable)).unwrap();
                surface.present(&device, &drawable);
                time += 1.0 / 60.0;
            });
        }

        device.wait_idle()?;
        surface.destroy(&device);
        Ok(())
    }
}
