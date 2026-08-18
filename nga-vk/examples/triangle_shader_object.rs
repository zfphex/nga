//! `triangle`, rebuilt on `VK_EXT_shader_object` with no `VkPipeline` anywhere.
//!
//! Everything else is deliberately identical to `examples/triangle.rs`, so the diff between the
//! two is exactly what the extension costs and saves. See NOTES.md.

use ash::vk;
use minwin::*;
use nga_vk::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Data {
    vertices: Gpu<'static, Vertex>,
    mvp: f32x4x4,
}

const SHADERS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.spv"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window(
        "nga-vk M0 — shader object",
        None,
        800,
        600,
        true,
        WindowStyle::Standard,
    );

    let device = Device::new(cfg!(debug_assertions))?;
    eprintln!("device: {}", device.name);

    let (mut width, mut height) = window.scaled_size();
    let mut swapchain = Swapchain::new(&device, window.hwnd, width as u32, height as u32)?;

    // The whole of pipeline creation. No render pass, no PSO, no colour-target formats, no
    // blend or depth description — just the two entry points out of one SPIR-V module.
    let shaders = ShaderObjects::raster(&device, SHADERS, "vertex_main", "fragment_main")?;

    // `Context` still needs a layout for push constants, so it borrows the shader objects'.
    let mut ctx = Context::with_layout(&device, shaders.layout, 64 * 1024, 1)?;

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

    let mesh = Arena::new(&device, 64 * 1024, Placement::Bar)?;
    let vertices = mesh.alloc_slice(&triangle).unwrap();
    let vertices: Gpu<'static, Vertex> = Gpu::new(vertices.gpu.0.get());

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

            let mut frame = ctx.begin_frame().unwrap();
            let (sin, cos) = angle.sin_cos();
            let aspect = swapchain.height as f32 / swapchain.width as f32;
            let root = frame.arenas[0]
                .alloc(Data {
                    vertices,
                    mvp: f32x4x4([
                        [cos * aspect, sin, 0.0, 0.0],
                        [-sin * aspect, cos, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]),
                })
                .unwrap()
                .gpu;

            let mut recorder = frame.recorder();
            {
                let pass = Pass::new().color(0, &drawable, Some([0.05, 0.05, 0.07, 1.0]));
                let raster = recorder.raster(&pass, swapchain.width, swapchain.height);
                // Replaces `raster.pipeline(..)`: binds both stages and restates the ~18 pieces
                // of state a PSO would have carried.
                raster.shaders(&shaders, swapchain.width, swapchain.height);
                raster.draw(root, 0, 3);
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

            angle += 0.02;
        });

        window.wait_for_vsync();
    }

    unsafe { device.device.device_wait_idle()? };
    drop(mesh);
    drop(ctx);
    swapchain.destroy(&device);
    Ok(())
}
