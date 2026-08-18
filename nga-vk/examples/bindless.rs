use ash::vk;
use minwin::*;
use nga_vk::*;

#[gpu_struct]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
}

#[gpu_struct]
struct Material {
    texture_base: u32,
    sampler: u32,
    tint: [f32; 3],
}

#[gpu_struct]
struct Data {
    vertices: Gpu<'static, Vertex>,
    materials: Gpu<'static, Material>,
    mvp: f32x4x4,
}

#[gpu_struct]
struct Paint {
    target: u32,
    size: u32,
    time: f32,
}

const SHADERS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bindless.spv"));
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
    eprintln!("device: {}", device.name);
    eprintln!(
        "descriptor sizes: sampled {} storage {} sampler {} (Metal: 8 for all three)",
        device.descriptor_sizes.sampled_image,
        device.descriptor_sizes.storage_image,
        device.descriptor_sizes.sampler
    );

    let (mut width, mut height) = window.scaled_size();
    let mut swapchain = Swapchain::new(&device, window.hwnd, width as u32, height as u32)?;

    slang::dump(
        "bindless",
        &format!(
            "{SLANG_PRELUDE}{}",
            slang::header!(Vertex, Material, Data, Paint)
        ),
    );

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let pipeline = shaders.raster(&RasterDesc::new(
        "vertex_main",
        "fragment_main",
        &[ColorTarget::new(swapchain.format)],
    ))?;
    let paint_pipeline = shaders.compute("paint", None)?;

    let mut heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;
    let mut textures = ResourceHeap::new(&device, device.bindless_cap)?;

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
                position: [px, py],
                uv: [u, v],
            });
        }
    }
    let vertex_buffer = heap.alloc_buffer(size_of_val(quads.as_slice()))?;
    heap.write(&vertex_buffer, 0, &quads);

    let sampler = unsafe {
        device.device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .min_filter(vk::Filter::NEAREST)
                .mag_filter(vk::Filter::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        )?
    };
    let sampler_range = textures.alloc_range(1)?;
    textures.write_sampler(sampler_range, 0, &sampler);

    let mut live = Vec::new();
    let mut materials = Vec::new();
    let mut paint_target = 0;
    let mut pixels = vec![0u8; (TEX * TEX * 4) as usize];
    for quad in 0..QUADS {
        let range = textures.alloc_range(LAYERS)?;
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
            let mut desc = TextureDesc::texture_2d(vk::Format::R8G8B8A8_UNORM, TEX, TEX);
            if painted {
                desc.usage |= vk::ImageUsageFlags::STORAGE;
            }
            let texture = heap.alloc_texture(&desc)?;
            heap.upload(&texture, &pixels)?;

            // A painted texture is read and written in the same frame, so it stays in GENERAL.
            // Metal needs no such distinction — there is no layout to choose.
            match painted {
                true => {
                    heap.transition(&texture, vk::ImageLayout::GENERAL);
                    textures.write_storage(range, layer, &texture);
                    textures.write_texture_in(range, layer, &texture, vk::ImageLayout::GENERAL);
                    paint_target = range.base + layer;
                }
                false => {
                    heap.transition(&texture, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                    textures.write_texture(range, layer, &texture);
                }
            }
            live.push(texture);
        }
        materials.push(Material {
            texture_base: range.base,
            sampler: sampler_range.base,
            tint: [
                1.0 - quad as f32 * 0.15,
                0.6 + quad as f32 * 0.1,
                0.5 + quad as f32 * 0.12,
            ],
        });
        eprintln!("material {quad}: texture_base {}", range.base);
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
                swapchain
                    .rebuild(&device, width as u32, height as u32)
                    .unwrap();
                resized = false;
            }
            if swapchain.width == 0 || swapchain.height == 0 {
                return;
            }

            let Some(drawable) = swapchain.acquire(&device) else {
                resized = true;
                return;
            };

            let aspect = swapchain.height as f32 / swapchain.width as f32;
            let mut frame = ctx.begin_frame().unwrap();
            let mut rec = frame.recorder();

            let paint_root = rec
                .arena
                .alloc(Paint {
                    target: paint_target,
                    size: TEX,
                    time,
                })
                .unwrap()
                .gpu;
            let data_root = rec
                .arena
                .alloc(Data {
                    vertices: Gpu::new(vertex_buffer.gpu),
                    materials: Gpu::new(material_buffer.gpu),
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
                compute.barrier(Stages::DISPATCH, Stages::FRAGMENT, Visibility::Device);
            }

            {
                let pass = Pass::new().color(0, &drawable, Some([0.05, 0.05, 0.07, 1.0]));
                let raster = rec.raster(&pass, swapchain.width, swapchain.height);
                raster.heap(&textures);
                raster.pipeline(pipeline);
                raster.draw(data_root, 0, QUADS * 6);
            }

            drop(rec);
            frame.submit(Some(&drawable)).unwrap();

            let indices = [drawable.index];
            let swapchains = [swapchain.swapchain];
            let waits = [drawable.released];
            let present = unsafe {
                device.swapchain_ext.queue_present(
                    device.queue(),
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&waits)
                        .swapchains(&swapchains)
                        .image_indices(&indices),
                )
            };
            if matches!(present, Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Ok(true)) {
                resized = true;
            }

            time += 0.016;
        });

        window.wait_for_vsync();
    }

    unsafe {
        device.device.device_wait_idle()?;
        device.device.destroy_sampler(sampler, None);
    }
    Ok(())
}
