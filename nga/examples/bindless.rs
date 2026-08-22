use minwin::*;
use nga::*;
use bindless::{Data, Material, Paint, Vertex};

/// The shader, in Rust. See `triangle.rs` for what `#[gpu_mod]` does with it.
#[gpu_mod]
mod bindless {
    use nga::*;

    pub struct Vertex {
        pub position: f32x2,
        pub uv: f32x2,
    }

    pub struct Material {
        pub layers: [Texture2d; 3],
        pub samp: Sampler,
        pub tint: f32x3,
    }

    #[repr(C, align(16))]
    pub struct Data {
        pub vertices: Gpu<'static, [Vertex]>,
        pub materials: Gpu<'static, [Material]>,
        pub mvp: f32x4x4,
    }

    pub struct Paint {
        pub target: Texture2dRW,
        pub size: u32,
        pub time: f32,
    }

    #[compute(8, 8, 1)]
    fn paint(root: Gpu<'static, Paint>, #[thread_id] tid: u32x3) {
        if tid.x >= root.size || tid.y >= root.size {
            return;
        }
        let p = (f32x2::new(tid.x as f32, tid.y as f32) + 0.5) / root.size as f32 * 2.0 - 1.0;
        let r = p.length();
        let wave = (r * 18.0 - root.time * 6.0).sin() * 0.5 + 0.5;
        let prev = root.target.load(tid.xy());
        root.target.store(
            tid.xy(),
            f32x4::new(wave, wave * (1.0 - r), prev.z * 0.9 + 0.1 * r, 1.0),
        );
    }

    /// `material` is an integer varying, so it is flat on both targets — that is not a choice
    /// either language leaves open.
    pub struct VertexOut {
        pub position: f32x4,
        pub uv: f32x2,
        pub material: u32,
    }

    #[vertex]
    fn vertex(root: Gpu<'static, Data>, #[vertex_id] vid: u32) -> VertexOut {
        let v = root.vertices[vid];
        VertexOut {
            position: root
                .mvp
                .mul(f32x4::new(v.position.x, v.position.y, 0.0, 1.0)),
            uv: v.uv,
            material: vid / 6,
        }
    }

    #[fragment]
    fn fragment(root: Gpu<'static, Data>, input: VertexOut) -> f32x4 {
        let m = root.materials[input.material];
        let layer = ((input.uv.x * 3.0) as u32).min(2);
        let c = m.layers[layer as usize].sample(m.samp, input.uv).xyz() * m.tint;
        f32x4::new(c.x, c.y, c.z, 1.0)
    }
}

const QUADS: u32 = 4;
const LAYERS: u32 = 3;
const TEX: u32 = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window(
        "nga-vk M2 — bindless",
        None,
        900,
        900,
        true,
        WindowStyle::Standard,
    );

    let device = Device::new(cfg!(debug_assertions))?;
    eprintln!("device: {}", device.name());
    eprintln!("this device holds up to {} descriptors", device.max_resources());

    let (mut width, mut height) = window.scaled_size();
    // The platform window handle: an `HWND` on Windows, the `NSView` on macOS.
    #[cfg(not(target_vendor = "apple"))]
    let handle = window.hwnd;
    #[cfg(target_vendor = "apple")]
    let handle = window.ns_view as isize;
    let mut surface = Surface::new(&device, handle, width as u32, height as u32)?;

    let mut shaders = Shaders::new(&device, &nga::compile(bindless::ENTRIES))?;
    let pipeline = shaders.raster(&RasterDesc::new(
        bindless::VERTEX.name,
        bindless::FRAGMENT.name,
        &[ColorTarget::new(surface.format())],
    ))?;
    let paint_pipeline = shaders.compute(bindless::PAINT.name, None)?;

    let mut heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;
    // One sampler plus one range of layers per quad. A heap costs descriptor bytes for the
    // slots it is asked for, so this asks for what it uses rather than for the ceiling.
    let mut textures = ResourceHeap::new(&device, 1 + QUADS * LAYERS)?;

    let mut quads = Vec::new();
    for quad in 0..QUADS {
        let (x, y) = ((quad % 2) as f32, (quad / 2) as f32);
        let (x0, y0) = (x * 0.9 - 0.95, 0.95 - y * 0.9 - 0.9);
        let (x1, y1) = (x0 + 0.9, y0 + 0.9);
        for &(px, py, u, v) in &[
            (x0, y1, 0.0, 0.0),
            (x0, y0, 0.0, 1.0),
            (x1, y0, 1.0, 1.0),
            (x0, y1, 0.0, 0.0),
            (x1, y0, 1.0, 1.0),
            (x1, y1, 1.0, 0.0),
        ] {
            quads.push(Vertex {
                position: f32x2([px, py]),
                uv: f32x2([u, v]),
            });
        }
    }
    let vertex_buffer = heap.alloc_buffer(size_of_val(quads.as_slice()))?;
    heap.write(&vertex_buffer, 0, &quads);

    let sampler = device.sampler(&SamplerDesc::nearest())?;
    let sampler_range = textures.alloc_range(1)?;
    let samp = textures.write_sampler(sampler_range, 0, &sampler);

    let mut live = Vec::new();
    let mut materials = Vec::new();
    let mut paint_target = Texture2dRW::default();
    let mut pixels = vec![0u8; (TEX * TEX * 4) as usize];
    for quad in 0..QUADS {
        let range = textures.alloc_range(LAYERS)?;
        let mut layer_handles = [Texture2d::default(); LAYERS as usize];
        for layer in 0..LAYERS {
            for y in 0..TEX {
                for x in 0..TEX {
                    let (fx, fy) = (x as f32 / TEX as f32, y as f32 / TEX as f32);
                    let value = match layer {
                        0 => ((x / 8 + y / 8) % 2) as f32,
                        1 => {
                            (((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt() * 12.0).sin() * 0.5
                                + 0.5
                        }
                        _ => fx * fy,
                    };
                    let rgb = [
                        value * (quad as f32 / QUADS as f32).max(0.25),
                        value * (1.0 - fy),
                        value * fx,
                    ];
                    let texel = ((y * TEX + x) * 4) as usize;
                    pixels[texel] = (rgb[0] * 255.0) as u8;
                    pixels[texel + 1] = (rgb[1] * 255.0) as u8;
                    pixels[texel + 2] = (rgb[2] * 255.0) as u8;
                    pixels[texel + 3] = 255;
                }
            }
            let painted = quad == 0 && layer == 0;
            let mut desc = TextureDesc::texture_2d(Format::Rgba8Unorm, TEX, TEX);
            if painted {
                desc = desc.usage(Usage::STORAGE);
            }
            let texture = heap.alloc_texture(&desc)?;
            heap.upload(&texture, 0, &pixels)?;

            // A painted texture is read and written in the same frame, so it stays in GENERAL.
            // Metal needs no such distinction, because there is no layout to choose.
            layer_handles[layer as usize] = match painted {
                true => {
                    heap.transition(&texture, Usage::STORAGE);
                    paint_target = textures.write_storage(range, layer, &texture);
                    textures.write_texture_in(range, layer, &texture, Usage::STORAGE)
                }
                false => {
                    heap.transition(&texture, Usage::SAMPLED);
                    textures.write_texture(range, layer, &texture)
                }
            };
            live.push(texture);
        }
        materials.push(Material {
            layers: layer_handles,
            samp,
            tint: f32x3([
                1.0 - quad as f32 * 0.15,
                0.6 + quad as f32 * 0.1,
                0.5 + quad as f32 * 0.12,
            ]),
        });
        eprintln!("material {quad}: layers {layer_handles:?}");
    }
    let material_buffer = heap.alloc_buffer(size_of_val(materials.as_slice()))?;
    heap.write(&material_buffer, 0, &materials);

    // Every copy and transition above was recorded, not submitted. One flush lands the lot,
    // where the old path drained the device once per texture.
    heap.flush()?;

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

            let aspect = surface.height() as f32 / surface.width() as f32;
            let mut frame = ctx.begin_frame().unwrap();
            let mut rec = frame.recorder();

            let paint_root = rec
                .arena()
                .alloc(Paint {
                    target: paint_target,
                    size: TEX,
                    time,
                })
                .unwrap()
                .gpu;
            let data_root = rec
                .arena()
                .alloc(Data {
                    vertices: Gpu::new(vertex_buffer.address()),
                    materials: Gpu::new(material_buffer.address()),
                    mvp: f32x4x4([
                        [aspect.min(1.0), 0.0, 0.0, 0.0],
                        [0.0, (1.0 / aspect).min(1.0), 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]),
                })
                .unwrap()
                .gpu;

            {
                let compute = rec.compute();
                compute.heap(&textures);
                compute.pipeline(paint_pipeline);
                compute.dispatch_2d(paint_root, TEX, TEX, 8);
                compute.barrier(Stage::COMPUTE, Stage::PIXEL, Hazard::ALL);
            }

            {
                let pass = Pass::new().color(0, &drawable, Some([0.05, 0.05, 0.07, 1.0]));
                let raster = rec.raster(&pass, surface.width(), surface.height());
                raster.heap(&textures);
                raster.pipeline(pipeline);
                raster.draw(data_root, 0, QUADS * 6);
            }

            drop(rec);
            frame.submit(Some(&drawable)).unwrap();

            resized = surface.present(&device, &drawable);

            time += 0.016;
        });

        window.wait_for_vsync();
    }

    device.wait_idle()?;
    for texture in live {
        heap.free_texture(texture);
    }
    heap.free_buffer(material_buffer);
    heap.free_buffer(vertex_buffer);
    // The sampler destroys itself; nothing here has to name a Vulkan call to tear down.
    drop(sampler);
    Ok(())
}
