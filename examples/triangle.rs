use minmetal::*;
use minwin::*;
use nga::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
struct Data {
    vertices: Gpu<'static, Vertex>,
    mvp: f32x4x4,
}

const SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    packed_float3 position;
    packed_float3 color;
};

struct Data {
    device const Vertex* vertices;
    float4x4 mvp;
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut vertex_main(device const Data* data [[buffer(0)]], uint vid [[vertex_id]]) {
    VertexOut out;
    out.position = data->mvp * float4(data->vertices[vid].position, 1.0);
    out.color = data->vertices[vid].color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
"#;

const FRAMES_IN_FLIGHT: u64 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window("nga M0", None, 800, 600, true, WindowStyle::Standard);

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

    let compiler = device.new_m4_compiler_with_descriptor(&M4CompilerDescriptor::new())?;
    let library_desc = M4LibraryDescriptor::new();
    library_desc.set_source(SHADERS);
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
    let mut heap = GpuHeap::new(&device, 16 * 1024 * 1024)?;

    let triangle = [
        Vertex {
            position: [0.0, 0.6, 0.0],
            color: [1.0, 0.2, 0.2],
        },
        Vertex {
            position: [-0.6, -0.5, 0.0],
            color: [0.2, 1.0, 0.3],
        },
        Vertex {
            position: [0.6, -0.5, 0.0],
            color: [0.3, 0.4, 1.0],
        },
    ];
    let scratch = heap.alloc_texture(&TextureDescriptor::texture_2d(
        PixelFormat::Rgba8Unorm,
        256,
        256,
        false,
    ))?;
    let mut mesh = heap.alloc_buffer(size_of_val(&triangle))?;
    mesh.write(0, &triangle);
    let vertices: Gpu<'static, Vertex> = Gpu::new(mesh.gpu);
    eprintln!(
        "heap: texture {:?} at {:#x}, mesh at {:#x}",
        scratch.texture.raw, scratch.alloc.offset, mesh.alloc.offset
    );
    drop(scratch.texture);
    heap.free(scratch.alloc);

    let mut data: Pool<Data> = Pool::new(&arena, FRAMES_IN_FLIGHT as u32)
        .ok_or(MetalError::new("frame arena is too small"))?;
    for _ in 0..FRAMES_IN_FLIGHT {
        data.alloc(Data {
            vertices,
            mvp: f32x4x4([[0.0; 4]; 4]),
        });
    }

    let residency = device.new_residency_set(&ResidencySetDescriptor::new()?)?;
    residency.add_allocation(&Allocation::from_buffer(&arena.buffer))?;
    residency.add_allocation(&Allocation::from_heap(&heap.heap))?;
    residency.commit()?;
    queue.add_residency_set(&residency);

    let table_desc = M4ArgumentTableDescriptor::new();
    table_desc.set_max_buffer_bind_count(1);
    let table = device.new_m4_argument_table_with_descriptor(&table_desc)?;

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
            let slot = (frame % FRAMES_IN_FLIGHT) as usize;

            let (sin, cos) = angle.sin_cos();
            let aspect = drawable_height as f32 / drawable_width as f32;
            data[slot as u32].mvp = f32x4x4([
                [cos * aspect, sin, 0.0, 0.0],
                [-sin * aspect, cos, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            table.set_address(data.gpu(slot as u32).0.get(), 0);

            allocators[slot].reset();
            command_buffer.begin_command_buffer_with_allocator(&allocators[slot]);

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
            encoder.draw_primitives(PrimitiveType::Triangle, 0, 3);
            encoder.end_encoding();

            command_buffer.end_command_buffer();

            queue.wait_for_drawable(&drawable);
            queue.commit(std::slice::from_ref(&command_buffer));
            queue.signal_event(&event.as_event(), frame + 1);
            queue.signal_drawable(&drawable);
            drawable.present();

            frame += 1;
            angle += 0.02;
        });

        window.wait_for_vsync();
    }

    Ok(())
}
