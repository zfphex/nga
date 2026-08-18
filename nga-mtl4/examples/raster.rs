use minmetal::*;
use minwin::*;
use nga_mtl4::*;

#[gpu_struct]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[gpu_struct]
struct Data {
    vertices: Gpu<'static, Vertex>,
    mvp: f32x4x4,
    tint: f32x4,
}

#[derive(Constants)]
struct Options {
    boost: bool,
    gamma: f32,
}

const SHADERS: &str = include_str!("../shaders/raster.metal");

const FRAMES_IN_FLIGHT: u64 = 2;
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            out[column][row] = (0..4).map(|k| a[k][row] * b[column][k]).sum();
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window("nga M3 — raster", None, 900, 700, true, WindowStyle::Standard);

    let device = Device::required_system_default()?;
    eprintln!("device: {}", device.name());

    let queue = device.new_m4_command_queue().ok_or(MetalError::new(
        "MTL4CommandQueue is unavailable — Metal 4 requires macOS 26",
    ))?;
    let mut command_buffer = device
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
        "{}{}\n{SHADERS}",
        msl::header!(Vertex, Data),
        Options::msl()
    );
    msl::dump("raster", &format!("{MSL_PRELUDE}{source}"));
    let shaders = Shaders::new(&device, &source)?;

    let opaque = shaders.raster(&RasterDesc {
        constants: Some(&Options {
            boost: false,
            gamma: 1.0,
        }),
        ..RasterDesc::new(
            "vertex_main",
            "fragment_main",
            &[ColorTarget::new(PixelFormat::Bgra8Unorm)],
        )
    })?;
    let translucent = shaders.raster(&RasterDesc {
        constants: Some(&Options {
            boost: true,
            gamma: 0.6,
        }),
        ..RasterDesc::new(
            "vertex_main",
            "fragment_main",
            &[ColorTarget {
                blend: BlendMode::Fixed(Blend::ALPHA),
                ..ColorTarget::new(PixelFormat::Bgra8Unorm)
            }],
        )
    })?;
    let screen = shaders.raster(&RasterDesc::new(
        "vertex_main",
        "fragment_screen",
        &[ColorTarget {
            blend: BlendMode::Programmable,
            ..ColorTarget::new(PixelFormat::Bgra8Unorm)
        }],
    ))?;

    let depth_write = depth_state(&device, CompareFunction::Less, true)?;
    let depth_test = depth_state(&device, CompareFunction::Less, false)?;
    let depth_off = depth_state(&device, CompareFunction::Always, false)?;

    let arena = Arena::new(&device, 64 * 1024)?;
    let mut heap = GpuHeap::new(&device, 64 * 1024 * 1024, StorageMode::Shared)?;

    let mut corners = Vec::new();
    for i in 0..8u32 {
        let (x, y, z) = (
            (i & 1) as f32 - 0.5,
            ((i >> 1) & 1) as f32 - 0.5,
            ((i >> 2) & 1) as f32 - 0.5,
        );
        corners.push(Vertex {
            position: [x, y, z],
            color: [x + 0.5, y + 0.5, z + 0.5],
        });
    }
    let indices: [u16; 36] = [
        0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4,
        2, 6, 7, 2, 7, 3,
    ];

    let mut vertex_buffer = heap.alloc_buffer(size_of_val(corners.as_slice()))?;
    vertex_buffer.write(0, &corners);
    let mut index_buffer = heap.alloc_buffer(size_of_val(&indices))?;
    index_buffer.write(0, &indices);
    let vertices: Gpu<'static, Vertex> = Gpu::new(vertex_buffer.gpu);
    let index_ptr: Gpu<'static, u16> = Gpu::new(index_buffer.gpu);

    let mut data: Pool<Data> = Pool::new(&arena, FRAMES_IN_FLIGHT as u32 * 3)
        .ok_or(MetalError::new("frame arena is too small"))?;
    for _ in 0..FRAMES_IN_FLIGHT * 3 {
        data.alloc(Data {
            vertices,
            mvp: f32x4x4([[0.0; 4]; 4]),
            tint: f32x4([1.0; 4]),
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

    let mut depth: Option<(HeapTexture, usize, usize)> = None;
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

            if depth
                .as_ref()
                .is_none_or(|&(_, w, h)| w != drawable_width || h != drawable_height)
            {
                if let Some((old, _, _)) = depth.take() {
                    heap.free(old.alloc);
                }
                let desc =
                    TextureDescriptor::texture_2d(DEPTH_FORMAT, drawable_width, drawable_height, false);
                desc.set_usage(TextureUsage::RENDER_TARGET);
                let Ok(texture) = heap.alloc_texture(&desc) else {
                    return;
                };
                depth = Some((texture, drawable_width, drawable_height));
            }

            let aspect = drawable_width as f32 / drawable_height as f32;
            let f = 1.0 / (0.9f32 / 2.0).tan();
            let projection = [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, 10.0 / 9.9, 1.0],
                [0.0, 0.0, -1.0 / 9.9, 0.0],
            ];
            let (sin, cos) = angle.sin_cos();
            let view_rotation = [
                [cos, sin * 0.35, -sin * 0.94, 0.0],
                [0.0, 0.94, 0.35, 0.0],
                [sin, -cos * 0.35, cos * 0.94, 0.0],
                [0.0, 0.0, 4.6, 1.0],
            ];
            let view_projection = mul(projection, view_rotation);
            let scaled = |s: f32| {
                mul(
                    view_projection,
                    [
                        [s, 0.0, 0.0, 0.0],
                        [0.0, s, 0.0, 0.0],
                        [0.0, 0.0, s, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                )
            };

            let slot = (frame % FRAMES_IN_FLIGHT) as u32 * 3;
            data[slot].mvp = f32x4x4(scaled(1.0));
            data[slot].tint = f32x4([1.0, 1.0, 1.0, 1.0]);
            data[slot + 1].mvp = f32x4x4(scaled(1.6));
            data[slot + 1].tint = f32x4([0.4, 0.8, 1.0, 0.35]);
            data[slot + 2].mvp = f32x4x4(scaled(2.1));
            data[slot + 2].tint = f32x4([1.0, 0.5, 0.2, 0.6]);

            allocators[(frame % FRAMES_IN_FLIGHT) as usize].reset();
            command_buffer
                .begin_command_buffer_with_allocator(&allocators[(frame % FRAMES_IN_FLIGHT) as usize]);

            let pass = Pass::new()
                .color(0, &texture, Some(ClearColor::new(0.04, 0.04, 0.06, 1.0)))
                .depth(&depth.as_ref().unwrap().0.texture, Some(1.0), false);
            let viewport = Viewport::new(
                0.0,
                0.0,
                drawable_width as f64,
                drawable_height as f64,
                0.0,
                1.0,
            );

            {
                let Some(raster) = Raster::new(&mut command_buffer, &pass, &table, viewport) else {
                    return;
                };
                raster.cull(CullMode::Back, Winding::Clockwise);

                raster.pipeline(&opaque);
                raster.depth(&depth_write);
                raster.draw_indexed(data.gpu(slot), PrimitiveType::Triangle, index_ptr, indices.len());

                raster.pipeline(&translucent);
                raster.depth(&depth_test);
                raster.draw_indexed(
                    data.gpu(slot + 1),
                    PrimitiveType::Triangle,
                    index_ptr,
                    indices.len(),
                );

                raster.pipeline(&screen);
                raster.depth(&depth_off);
                raster.draw_indexed(
                    data.gpu(slot + 2),
                    PrimitiveType::Triangle,
                    index_ptr,
                    indices.len(),
                );
            }

            command_buffer.end_command_buffer();

            queue.wait_for_drawable(&drawable);
            queue.commit(std::slice::from_ref(&command_buffer));
            queue.signal_event(&event.as_event(), frame + 1);
            queue.signal_drawable(&drawable);
            drawable.present();

            frame += 1;
            angle += 0.01;
        });

        window.wait_for_vsync();
    }

    Ok(())
}
