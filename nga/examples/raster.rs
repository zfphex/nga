use minwin::*;
use nga::*;
use raster::{Data, Options, Vertex};

/// The shader, in Rust. See `triangle.rs` for what `#[gpu_mod]` does with it.
#[gpu_mod]
mod raster {
    use nga::*;

    pub struct Vertex {
        pub position: f32x3,
        pub color: f32x3,
    }

    #[repr(C, align(16))]
    pub struct Data {
        pub vertices: Gpu<[Vertex]>,
        pub mvp: f32x4x4,
        pub tint: f32x4,
    }

    /// One specialization constant per field, numbered by field index — the same number
    /// `Constants::values` pushes, so the host and the module cannot disagree about which is
    /// which.
    #[derive(Constants)]
    pub struct Options {
        pub boost: bool,
        pub gamma: f32,
    }

    pub struct VertexOut {
        pub position: f32x4,
        pub color: f32x3,
        pub alpha: f32,
    }

    #[vertex]
    fn vertex(root: Gpu<Data>, #[vertex_id] vid: u32) -> VertexOut {
        let v = root.vertices[vid];
        VertexOut {
            position: root
                .mvp
                .mul(f32x4::new(v.position.x, v.position.y, v.position.z, 1.0)),
            color: v.color * root.tint.xyz(),
            alpha: root.tint.w,
        }
    }

    #[fragment]
    fn fragment(input: VertexOut, #[constants] options: Options) -> f32x4 {
        let mut c = input.color.pow(f32x3::splat(options.gamma));
        if options.boost {
            c = (c * 1.6).min(f32x3::splat(1.0));
        }
        f32x4::new(c.x, c.y, c.z, input.alpha)
    }

    /// The Metal original read `float4 dst [[color(0)]]` — a framebuffer fetch — and computed
    /// screen blending itself. Desktop Vulkan has no framebuffer fetch, so this emits `src`
    /// only and the screen formula is a fixed blend equation instead: `ONE`,
    /// `ONE_MINUS_SRC_COLOR` is `src + dst * (1 - src)`, which is exactly screen.
    #[fragment]
    fn fragment_screen(input: VertexOut) -> f32x4 {
        let c = input.color * input.alpha;
        f32x4::new(c.x, c.y, c.z, 1.0)
    }
}

const DEPTH_FORMAT: Format = Format::D32Float;

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
    let mut window = create_window(
        "nga-vk M3 — raster",
        None,
        900,
        700,
        true,
        WindowStyle::Standard,
    );

    let device = Device::new(cfg!(debug_assertions))?;
    eprintln!("device: {}", device.name());

    let (mut width, mut height) = window.scaled_size();
    let mut surface = device.surface(window.handle(), width as u32, height as u32)?;
    let mut shaders = device.shaders(raster::MODULE)?;
    let opaque = shaders
        .raster(raster::VERTEX, raster::FRAGMENT)
        .color(surface.format())
        .depth(DEPTH_FORMAT)
        .constants(&Options {
            boost: false,
            gamma: 1.0,
        })
        .build()?;
    let translucent = shaders
        .raster(raster::VERTEX, raster::FRAGMENT)
        .color(surface.format())
        .blend(BlendMode::Alpha)
        .depth(DEPTH_FORMAT)
        .constants(&Options {
            boost: true,
            gamma: 0.6,
        })
        .build()?;
    let screen = shaders
        .raster(raster::VERTEX, raster::FRAGMENT_SCREEN)
        .color(surface.format())
        .blend(BlendMode::Screen)
        .depth(DEPTH_FORMAT)
        .build()?;

    let depth_write = DepthState::new(CompareOp::Less, true);
    let depth_test = DepthState::new(CompareOp::Less, false);
    let depth_off = DepthState::new(CompareOp::Always, false);

    let heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;

    let mut corners = Vec::new();
    for i in 0..8u32 {
        let (x, y, z) = (
            (i & 1) as f32 - 0.5,
            ((i >> 1) & 1) as f32 - 0.5,
            ((i >> 2) & 1) as f32 - 0.5,
        );
        corners.push(Vertex {
            position: f32x3([x, y, z]),
            color: f32x3([x + 0.5, y + 0.5, z + 0.5]),
        });
    }
    let indices: [u16; 36] = [
        0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4,
        2, 6, 7, 2, 7, 3,
    ];

    let vertex_buffer = heap.alloc_slice(&corners)?;
    let index_buffer = heap.alloc_slice(&indices)?;
    heap.flush()?;
    let vertices: Gpu<[Vertex]> = vertex_buffer.gpu();

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

            if depth
                .as_ref()
                .is_none_or(|&(_, w, h)| w != surface.width() || h != surface.height())
            {
                device.wait_idle().unwrap();
                depth = None;
                let texture = heap
                    .alloc_texture(&TextureDesc::depth(
                        DEPTH_FORMAT,
                        surface.width(),
                        surface.height(),
                    ))
                    .unwrap();
                depth = Some((texture, surface.width(), surface.height()));
            }

            let aspect = surface.width() as f32 / surface.height() as f32;
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
            let recorder = frame.recorder();
            let pass = Pass::new()
                .color(0, &drawable, Some([0.04, 0.04, 0.06, 1.0]))
                .depth(&depth.as_ref().unwrap().0, Some(1.0), false);
            let raster = recorder.raster(&pass);
            raster.cull(Cull::Back, Winding::Clockwise);

            raster.depth(&depth_write);
            raster
                .draw_indexed::<_, u16>(
                    opaque,
                    Data {
                        vertices,
                        mvp: f32x4x4(scaled(1.0)),
                        tint: f32x4([1.0, 1.0, 1.0, 1.0]),
                    },
                    &index_buffer,
                    indices.len() as u32,
                )
                .unwrap();

            raster.depth(&depth_test);
            raster
                .draw_indexed::<_, u16>(
                    translucent,
                    Data {
                        vertices,
                        mvp: f32x4x4(scaled(1.6)),
                        tint: f32x4([0.4, 0.8, 1.0, 0.35]),
                    },
                    &index_buffer,
                    indices.len() as u32,
                )
                .unwrap();

            raster.depth(&depth_off);
            raster
                .draw_indexed::<_, u16>(
                    screen,
                    Data {
                        vertices,
                        mvp: f32x4x4(scaled(2.1)),
                        tint: f32x4([1.0, 0.5, 0.2, 0.6]),
                    },
                    &index_buffer,
                    indices.len() as u32,
                )
                .unwrap();

            frame.submit(Some(&drawable)).unwrap();

            resized = surface.present(&device, &drawable);

            angle += 0.01;
        });

        window.wait_for_vsync();
    }

    device.wait_idle()?;
    Ok(())
}
