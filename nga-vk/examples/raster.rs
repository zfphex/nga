use ash::vk;
use minwin::*;
use nga_vk::*;

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

const SHADERS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/raster.spv"));
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

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
    let mut window = create_window("nga-vk M3 — raster", None, 900, 700, true, WindowStyle::Standard);

    let device = Device::new(cfg!(debug_assertions))?;
    eprintln!("device: {}", device.name);

    let (mut width, mut height) = window.scaled_size();
    let mut swapchain = Swapchain::new(&device, window.hwnd, width as u32, height as u32)?;

    slang::dump(
        "raster",
        &format!(
            "{SLANG_PRELUDE}{}{}",
            slang::header!(Vertex, Data),
            Options::slang()
        ),
    );

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let opaque = shaders.raster(&RasterDesc {
        depth: DEPTH_FORMAT,
        constants: Some(&Options {
            boost: false,
            gamma: 1.0,
        }),
        ..RasterDesc::new(
            "vertex_main",
            "fragment_main",
            &[ColorTarget::new(swapchain.format)],
        )
    })?;
    let translucent = shaders.raster(&RasterDesc {
        depth: DEPTH_FORMAT,
        constants: Some(&Options {
            boost: true,
            gamma: 0.6,
        }),
        ..RasterDesc::new(
            "vertex_main",
            "fragment_main",
            &[ColorTarget {
                blend: BlendMode::Alpha,
                ..ColorTarget::new(swapchain.format)
            }],
        )
    })?;
    let screen = shaders.raster(&RasterDesc {
        depth: DEPTH_FORMAT,
        ..RasterDesc::new(
            "vertex_main",
            "fragment_screen",
            &[ColorTarget {
                blend: BlendMode::Screen,
                ..ColorTarget::new(swapchain.format)
            }],
        )
    })?;

    let depth_write = depth_state(vk::CompareOp::LESS, true);
    let depth_test = depth_state(vk::CompareOp::LESS, false);
    let depth_off = depth_state(vk::CompareOp::ALWAYS, false);

    let mut heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;

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

    let vertex_buffer = heap.alloc_buffer(size_of_val(corners.as_slice()))?;
    heap.write(&vertex_buffer, 0, &corners);
    let index_buffer = heap.alloc_buffer(size_of_val(&indices))?;
    heap.write(&index_buffer, 0, &indices);
    heap.flush()?;
    let vertices: Gpu<'static, Vertex> = Gpu::new(vertex_buffer.gpu);

    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut depth: Option<(HeapTexture, u32, u32)> = None;
    let mut angle = 0.0f32;
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

            if depth
                .as_ref()
                .is_none_or(|&(_, w, h)| w != swapchain.width || h != swapchain.height)
            {
                unsafe { device.device.device_wait_idle().unwrap() };
                if let Some((old, _, _)) = depth.take() {
                    unsafe {
                        device.device.destroy_image_view(old.view, None);
                        device.device.destroy_image(old.image, None);
                    }
                    heap.free(old.alloc);
                }
                let texture = heap
                    .alloc_texture(&TextureDesc::depth(
                        DEPTH_FORMAT,
                        swapchain.width,
                        swapchain.height,
                    ))
                    .unwrap();
                depth = Some((texture, swapchain.width, swapchain.height));
            }

            let aspect = swapchain.width as f32 / swapchain.height as f32;
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

            let mut frame = ctx.begin_frame().unwrap();
            let arena = &frame.arenas[0];
            let solid = arena
                .alloc(Data {
                    vertices,
                    mvp: f32x4x4(scaled(1.0)),
                    tint: f32x4([1.0, 1.0, 1.0, 1.0]),
                })
                .unwrap()
                .gpu;
            let glass = arena
                .alloc(Data {
                    vertices,
                    mvp: f32x4x4(scaled(1.6)),
                    tint: f32x4([0.4, 0.8, 1.0, 0.35]),
                })
                .unwrap()
                .gpu;
            let halo = arena
                .alloc(Data {
                    vertices,
                    mvp: f32x4x4(scaled(2.1)),
                    tint: f32x4([1.0, 0.5, 0.2, 0.6]),
                })
                .unwrap()
                .gpu;

            let mut recorder = frame.recorder();
            {
                let pass = Pass::new()
                    .color(0, &drawable, Some([0.04, 0.04, 0.06, 1.0]))
                    .depth(&depth.as_ref().unwrap().0, Some(1.0), false);
                let raster = recorder.raster(&pass, swapchain.width, swapchain.height);
                raster.cull(vk::CullModeFlags::BACK, vk::FrontFace::CLOCKWISE);

                raster.pipeline(opaque);
                raster.depth(&depth_write);
                raster.draw_indexed::<_, u16>(solid, &index_buffer, indices.len() as u32);

                raster.pipeline(translucent);
                raster.depth(&depth_test);
                raster.draw_indexed::<_, u16>(glass, &index_buffer, indices.len() as u32);

                raster.pipeline(screen);
                raster.depth(&depth_off);
                raster.draw_indexed::<_, u16>(halo, &index_buffer, indices.len() as u32);
            }
            drop(recorder);
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

            angle += 0.01;
        });

        window.wait_for_vsync();
    }

    unsafe { device.device.device_wait_idle()? };
    Ok(())
}
