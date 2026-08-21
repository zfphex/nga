//! The same source drives both backends. Nothing here names Vulkan or Metal.

use minwin::*;
use nga::*;
use nga_shaders::triangle::{self, Vertex, Data};




fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window("nga triangle", None, 800, 600, true, WindowStyle::Standard);

    let device = Device::new(cfg!(debug_assertions))?;
    eprintln!("device: {}", device.name());

    let (mut width, mut height) = window.scaled_size();
    // The platform window handle: an `HWND` on Windows, the `NSView` on macOS.
    #[cfg(not(target_vendor = "apple"))]
    let handle = window.hwnd;
    #[cfg(target_vendor = "apple")]
    let handle = window.ns_view as isize;
    let mut surface = Surface::new(&device, handle, width as u32, height as u32)?;

    let mut shaders = Shaders::new(&device, &nga::compile(triangle::ENTRIES))?;
    let pipeline = shaders.raster(&RasterDesc::new(
        "vertex",
        "fragment",
        &[ColorTarget::new(surface.format())],
    ))?;

    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let triangle = [
        Vertex {
            position: f32x3([0.0, 0.6, 0.0]),
            color: f32x3([1.0, 0.2, 0.2]),
        },
        Vertex {
            position: f32x3([-0.6, -0.5, 0.0]),
            color: f32x3([0.2, 1.0, 0.3]),
        },
        Vertex {
            position: f32x3([0.6, -0.5, 0.0]),
            color: f32x3([0.3, 0.4, 1.0]),
        },
    ];

    let mesh = Arena::new(&device, 64 * 1024, Memory::Device)?;
    let vertices = mesh.alloc_slice(&triangle).unwrap();
    let vertices: Gpu<'static, [Vertex]> = Gpu::new(vertices.gpu.0.get());

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

            let mut frame = ctx.begin_frame().unwrap();
            let (sin, cos) = angle.sin_cos();
            let aspect = surface.height() as f32 / surface.width() as f32;
            let data = frame.arena(0)
                .alloc(Data {
                    vertices,
                    mvp: f32x4x4([
                        [cos * aspect, sin, 0.0, 0.0],
                        [-sin * aspect, cos, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]),
                })
                .unwrap();
            let root = data.gpu;

            let mut recorder = frame.recorder();
            {
                let pass = Pass::new().color(0, &drawable, Some([0.05, 0.05, 0.07, 1.0]));
                let raster = recorder.raster(&pass, surface.width(), surface.height());
                raster.pipeline(pipeline);
                raster.draw(root, 0, 3);
            }
            drop(recorder);
            frame.submit(Some(&drawable)).unwrap();

            resized = surface.present(&device, &drawable);
            angle += 0.02;
        });

        window.wait_for_vsync();
    }

    device.wait_idle()?;
    drop(mesh);
    drop(ctx);
    drop(shaders);
    surface.destroy(&device);
    Ok(())
}
