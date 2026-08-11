use minmetal::*;
use minwin::*;
use nga::*;

#[gpu_struct]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
}

#[gpu_struct]
struct Material {
    texture_base: u32,
    tint: [f32; 3],
}

#[gpu_struct]
struct Data {
    vertices: Gpu<'static, Vertex>,
    materials: Gpu<'static, Material>,
    mvp: f32x4x4,
}

const SHADERS: &str = include_str!("../shaders/bindless.metal");

const FRAMES_IN_FLIGHT: u64 = 2;
const QUADS: u32 = 4;
const LAYERS: u32 = 3;
const TEX: usize = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window("nga M2 — bindless", None, 900, 900, true, WindowStyle::Standard);

    let device = Device::required_system_default()?;
    eprintln!("device: {}", device.name());

    let queue = device.new_m4_command_queue().ok_or(MetalError::new(
        "MTL4CommandQueue is unavailable — Metal 4 requires macOS 26",
    ))?;
    let command_buffer = device
        .new_m4_command_buffer()
        .ok_or(MetalError::new("newCommandBuffer returned nil"))?;
    let allocators = [
        device
            .new_m4_command_allocator()
            .ok_or(MetalError::new("newCommandAllocator returned nil"))?,
        device
            .new_m4_command_allocator()
            .ok_or(MetalError::new("newCommandAllocator returned nil"))?,
    ];
    let event = device.new_shared_event()?;

    let source = format!(
        "{}\n{SHADERS}",
        msl::header!(Vertex, Material, Data)
    );
    msl::assert_matches(
        concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/generated/bindless.metal"),
        &format!("{MSL_PRELUDE}{source}"),
    );

    let compiler = device.new_m4_compiler_with_descriptor(&M4CompilerDescriptor::new())?;
    let library_desc = M4LibraryDescriptor::new();
    library_desc.set_source(&format!("{MSL_PRELUDE}{source}"));
    let library = compiler.new_library_with_descriptor(&library_desc)?;

    let vertex_function = M4LibraryFunctionDescriptor::new();
    vertex_function.set_name("vertex_main");
    vertex_function.set_library(Some(&library));
    let fragment_function = M4LibraryFunctionDescriptor::new();
    fragment_function.set_name("fragment_main");
    fragment_function.set_library(Some(&library));

    let pipeline_desc = M4RenderPipelineDescriptor::new();
    pipeline_desc.set_vertex_function_descriptor(Some(&vertex_function.as_function_descriptor()));
    pipeline_desc
        .set_fragment_function_descriptor(Some(&fragment_function.as_function_descriptor()));
    pipeline_desc
        .color_attachments()
        .object_at(0)
        .set_pixel_format(PixelFormat::Bgra8Unorm);
    let pipeline =
        compiler.new_render_pipeline_state(&pipeline_desc.as_pipeline_descriptor(), None)?;

    let arena = Arena::new(&device, 64 * 1024)?;
    let mut heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;
    let mut textures = ResourceHeap::new(&mut heap, 1024)?;

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
    let mut vertex_buffer = heap.alloc_buffer(size_of_val(quads.as_slice()))?;
    vertex_buffer.write(0, &quads);

    let mut live = Vec::new();
    let mut materials = Vec::new();
    let mut pixels = vec![0u8; TEX * TEX * 4];
    for quad in 0..QUADS {
        let range = textures.alloc_range(LAYERS)?;
        for layer in 0..LAYERS {
            for y in 0..TEX {
                for x in 0..TEX {
                    let (fx, fy) = (x as f32 / TEX as f32, y as f32 / TEX as f32);
                    let value = match layer {
                        0 => ((x / 8 + y / 8) % 2) as f32,
                        1 => (((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt() * 12.0).sin() * 0.5
                            + 0.5,
                        _ => fx * fy,
                    };
                    let rgb = [
                        value * (quad as f32 / QUADS as f32).max(0.25),
                        value * (1.0 - fy),
                        value * fx,
                    ];
                    let texel = (y * TEX + x) * 4;
                    pixels[texel] = (rgb[0] * 255.0) as u8;
                    pixels[texel + 1] = (rgb[1] * 255.0) as u8;
                    pixels[texel + 2] = (rgb[2] * 255.0) as u8;
                    pixels[texel + 3] = 255;
                }
            }
            let desc = TextureDescriptor::texture_2d(PixelFormat::Rgba8Unorm, TEX, TEX, false);
            desc.set_usage(TextureUsage::SHADER_READ);
            let texture = heap.alloc_texture(&desc)?;
            texture.texture.replace_region(
                Region::new_2d(0, 0, TEX, TEX),
                0,
                &pixels,
                TEX * 4,
            );
            textures.write(range, layer, texture.texture.gpu_resource_id()?);
            live.push(texture);
        }
        materials.push(Material {
            texture_base: range.base,
            tint: [
                1.0 - quad as f32 * 0.15,
                0.6 + quad as f32 * 0.1,
                0.5 + quad as f32 * 0.12,
            ],
        });
        eprintln!("material {quad}: texture_base {}", range.base);
    }
    let mut material_buffer = heap.alloc_buffer(size_of_val(materials.as_slice()))?;
    material_buffer.write(0, &materials);

    let mut data: Pool<Data> = Pool::new(&arena, FRAMES_IN_FLIGHT as u32)
        .ok_or(MetalError::new("frame arena is too small"))?;
    for _ in 0..FRAMES_IN_FLIGHT {
        data.alloc(Data {
            vertices: Gpu::new(vertex_buffer.gpu),
            materials: Gpu::new(material_buffer.gpu),
            mvp: f32x4x4([[0.0; 4]; 4]),
        });
    }

    let residency = device.new_residency_set(&ResidencySetDescriptor::new()?)?;
    residency.add_allocation(&Allocation::from_buffer(&arena.buffer))?;
    residency.add_allocation(&Allocation::from_heap(&heap.heap))?;
    residency.commit()?;
    queue.add_residency_set(&residency);

    let table_desc = M4ArgumentTableDescriptor::new();
    table_desc.set_max_buffer_bind_count(2);
    let table = device.new_m4_argument_table_with_descriptor(&table_desc)?;
    table.set_address(textures.storage.gpu, 1);

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

    let mut frame = 0u64;
    let mut angle = 0.0f32;

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
            let Ok(texture) = drawable.texture() else {
                return;
            };

            if frame >= FRAMES_IN_FLIGHT {
                event.wait_until_signaled_value(frame - FRAMES_IN_FLIGHT + 1, 1000);
            }
            let slot = (frame % FRAMES_IN_FLIGHT) as u32;

            let (sin, cos) = angle.sin_cos();
            let aspect = drawable_height as f32 / drawable_width as f32;
            data[slot].mvp = f32x4x4([
                [cos * aspect, sin, 0.0, 0.0],
                [-sin * aspect, cos, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            table.set_address(data.gpu(slot).0.get(), 0);

            allocators[slot as usize].reset();
            command_buffer.begin_command_buffer_with_allocator(&allocators[slot as usize]);

            let pass = M4RenderPassDescriptor::new();
            let color = pass.color_attachments().object_at_indexed_subscript(0);
            color.base().set_texture(Some(&texture));
            color.base().set_load_action(LoadAction::Clear);
            color.base().set_store_action(StoreAction::Store);
            color.set_clear_color(ClearColor::new(0.05, 0.05, 0.07, 1.0));

            let Some(encoder) = command_buffer.render_command_encoder_with_descriptor(&pass) else {
                return;
            };
            encoder.set_render_pipeline_state(&pipeline);
            encoder.set_argument_table(&table, RenderStages::VERTEX | RenderStages::FRAGMENT);
            encoder.set_viewport(Viewport::new(
                0.0,
                0.0,
                drawable_width as f64,
                drawable_height as f64,
                0.0,
                1.0,
            ));
            encoder.draw_primitives(PrimitiveType::Triangle, 0, quads.len());
            encoder.end_encoding();

            command_buffer.end_command_buffer();

            queue.wait_for_drawable(&drawable);
            queue.commit(std::slice::from_ref(&command_buffer));
            queue.signal_event(&event.as_event(), frame + 1);
            queue.signal_drawable(&drawable);
            drawable.present();

            frame += 1;
            angle += 0.005;
        });

        window.wait_for_vsync();
    }

    Ok(())
}
