use minmetal::*;
use minwin::*;
use nga::*;

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
    layer: u32,
}

#[gpu_struct]
struct Composite {
    color: u32,
    depth: u32,
    layers: u32,
    tile: u32x2,
    size: u32x2,
}

const SHADERS: &str = include_str!("../shaders/tbdr.metal");

const FRAMES_IN_FLIGHT: u64 = 2;
const SAMPLES: usize = 4;
const LAYERS: usize = 2;
const LAYER_SIZE: usize = 256;
const COLOR_FORMAT: PixelFormat = PixelFormat::Bgra8Unorm;
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;
const STENCIL_FORMAT: PixelFormat = PixelFormat::Stencil8;
const CUBE_STENCIL: u32 = 1;
const QUERIES: usize = 2;

fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            out[column][row] = (0..4).map(|k| a[k][row] * b[column][k]).sum();
        }
    }
    out
}

struct Targets {
    msaa_color: HeapTexture,
    msaa_depth: HeapTexture,
    msaa_stencil: HeapTexture,
    resolved_color: HeapTexture,
    resolved_depth: HeapTexture,
    width: usize,
    height: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window("nga — TBDR", None, 1000, 800, true, WindowStyle::Standard);

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

    let source = format!("{}\n{SHADERS}", msl::header!(Vertex, Data, Composite));
    msl::dump("tbdr", &format!("{MSL_PRELUDE}{source}"));
    let shaders = Shaders::new(&device, &source)?;

    let layered = shaders.raster(&RasterDesc {
        topology: PrimitiveTopologyClass::Triangle,
        ..RasterDesc::new(
            "vertex_layered",
            "fragment_layered",
            &[ColorTarget::new(COLOR_FORMAT)],
        )
    })?;
    let prepass = shaders.raster(&RasterDesc {
        samples: SAMPLES,
        ..RasterDesc::new("vertex_main", "", &[])
    })?;
    let opaque = shaders.raster(&RasterDesc {
        samples: SAMPLES,
        ..RasterDesc::new(
            "vertex_main",
            "fragment_main",
            &[ColorTarget::new(COLOR_FORMAT)],
        )
    })?;
    let backdrop = shaders.raster(&RasterDesc {
        samples: SAMPLES,
        ..RasterDesc::new(
            "vertex_fullscreen",
            "fragment_backdrop",
            &[ColorTarget::new(COLOR_FORMAT)],
        )
    })?;
    let composite = shaders.raster(&RasterDesc::new(
        "vertex_fullscreen",
        "fragment_composite",
        &[ColorTarget::new(COLOR_FORMAT)],
    ))?;
    let vignette = shaders.tile(&TileDesc::new("tile_vignette", &[COLOR_FORMAT]))?;

    let depth_write = depth_state(&device, CompareFunction::Less, true)?;
    let depth_off = depth_state(&device, CompareFunction::Always, false)?;
    let stamp = stencil_state(
        &device,
        CompareFunction::Less,
        true,
        StencilFace {
            pass: StencilOperation::Replace,
            ..StencilFace::KEEP
        },
        StencilFace {
            pass: StencilOperation::Replace,
            ..StencilFace::KEEP
        },
    )?;
    let inside = stencil_state(
        &device,
        CompareFunction::Equal,
        false,
        StencilFace {
            compare: CompareFunction::Equal,
            write: 0,
            ..StencilFace::KEEP
        },
        StencilFace {
            compare: CompareFunction::Equal,
            write: 0,
            ..StencilFace::KEEP
        },
    )?;
    let outside = stencil_state(
        &device,
        CompareFunction::Always,
        false,
        StencilFace {
            compare: CompareFunction::NotEqual,
            write: 0,
            ..StencilFace::KEEP
        },
        StencilFace {
            compare: CompareFunction::NotEqual,
            write: 0,
            ..StencilFace::KEEP
        },
    )?;

    let arena = Arena::new(&device, 64 * 1024)?;
    let mut heap = GpuHeap::new(&device, 16 * 1024 * 1024, StorageMode::Shared)?;
    let mut render_targets = GpuHeap::new(&device, 256 * 1024 * 1024, StorageMode::Private)?;
    let mut resources = ResourceHeap::new(&mut heap, 64)?;

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

    let queries = heap.alloc_buffer(8 * QUERIES * FRAMES_IN_FLIGHT as usize)?;
    unsafe { std::ptr::write_bytes(queries.cpu, 0, 8 * QUERIES * FRAMES_IN_FLIGHT as usize) };

    let layer_desc = TextureDescriptor::texture_2d_array(
        COLOR_FORMAT,
        LAYER_SIZE,
        LAYER_SIZE,
        LAYERS,
        false,
    );
    layer_desc.set_usage(TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ);
    let layer_color = render_targets.alloc_texture(&layer_desc)?;
    let layer_depth_desc = TextureDescriptor::texture_2d_array(
        DEPTH_FORMAT,
        LAYER_SIZE,
        LAYER_SIZE,
        LAYERS,
        false,
    );
    layer_depth_desc.set_usage(TextureUsage::RENDER_TARGET);
    let layer_depth = render_targets.alloc_texture(&layer_depth_desc)?;

    let slots = resources.alloc_range(3)?;
    resources.write(slots, 2, layer_color.texture.gpu_resource_id()?);

    let mut data: Pool<Data> = Pool::new(&arena, FRAMES_IN_FLIGHT as u32 * 4)
        .ok_or(MetalError::new("frame arena is too small"))?;
    let mut composites: Pool<Composite> = Pool::new(&arena, FRAMES_IN_FLIGHT as u32)
        .ok_or(MetalError::new("frame arena is too small"))?;
    for _ in 0..FRAMES_IN_FLIGHT * 4 {
        data.alloc(Data {
            vertices,
            mvp: f32x4x4([[0.0; 4]; 4]),
            tint: f32x4([1.0; 4]),
            layer: 0,
        });
    }
    for _ in 0..FRAMES_IN_FLIGHT {
        composites.alloc(Composite {
            color: slots.base,
            depth: slots.base + 1,
            layers: slots.base + 2,
            tile: u32x2([32, 32]),
            size: u32x2([1, 1]),
        });
    }

    let residency = device.new_residency_set(&ResidencySetDescriptor::new()?)?;
    residency.add_allocation(&Allocation::from_buffer(&arena.buffer))?;
    residency.add_allocation(&Allocation::from_heap(&heap.heap))?;
    residency.add_allocation(&Allocation::from_heap(&render_targets.heap))?;
    residency.commit()?;
    queue.add_residency_set(&residency);

    let table_desc = M4ArgumentTableDescriptor::new();
    table_desc.set_max_buffer_bind_count(2);
    let table = device.new_m4_argument_table_with_descriptor(&table_desc)?;
    table.set_address(resources.storage.gpu, 1);

    let scale = window.scale_factor();
    let (mut drawable_width, mut drawable_height) = window.scaled_size();
    let layer = unsafe {
        MetalLayer::attach_to_view(
            id(window.ns_view),
            &device,
            COLOR_FORMAT,
            drawable_width,
            drawable_height,
            scale,
        )?
    };

    let mut targets: Option<Targets> = None;
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
                let done = (frame - FRAMES_IN_FLIGHT) % FRAMES_IN_FLIGHT;
                let counts = unsafe {
                    std::slice::from_raw_parts(
                        queries.cpu.cast::<u64>().add(done as usize * QUERIES),
                        QUERIES,
                    )
                };
                if frame % 120 == 0 {
                    eprintln!("occlusion: cube {} samples, backdrop {}", counts[0], counts[1]);
                }
            }

            if targets
                .as_ref()
                .is_none_or(|t| t.width != drawable_width || t.height != drawable_height)
            {
                if frame > 0 {
                    event.wait_until_signaled_value(frame, 1000);
                }
                if let Some(old) = targets.take() {
                    for texture in [
                        old.msaa_color,
                        old.msaa_depth,
                        old.msaa_stencil,
                        old.resolved_color,
                        old.resolved_depth,
                    ] {
                        render_targets.free(texture.alloc);
                    }
                }
                let multisampled = |format: PixelFormat| {
                    let desc = TextureDescriptor::texture_2d(
                        format,
                        drawable_width,
                        drawable_height,
                        false,
                    );
                    desc.set_texture_type(TextureType::D2Multisample);
                    desc.set_sample_count(SAMPLES);
                    desc.set_usage(TextureUsage::RENDER_TARGET);
                    desc
                };
                let resolved = |format: PixelFormat| {
                    let desc = TextureDescriptor::texture_2d(
                        format,
                        drawable_width,
                        drawable_height,
                        false,
                    );
                    desc.set_usage(TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ);
                    desc
                };
                let Ok(built) = (|| -> Result<Targets, MetalError> {
                    Ok(Targets {
                        msaa_color: render_targets.alloc_texture(&multisampled(COLOR_FORMAT))?,
                        msaa_depth: render_targets.alloc_texture(&multisampled(DEPTH_FORMAT))?,
                        msaa_stencil: render_targets
                            .alloc_texture(&multisampled(STENCIL_FORMAT))?,
                        resolved_color: render_targets.alloc_texture(&resolved(COLOR_FORMAT))?,
                        resolved_depth: render_targets.alloc_texture(&resolved(DEPTH_FORMAT))?,
                        width: drawable_width,
                        height: drawable_height,
                    })
                })() else {
                    return;
                };
                let Ok(color_id) = built.resolved_color.texture.gpu_resource_id() else {
                    return;
                };
                let Ok(depth_id) = built.resolved_depth.texture.gpu_resource_id() else {
                    return;
                };
                resources.write(slots, 0, color_id);
                resources.write(slots, 1, depth_id);
                targets = Some(built);
            }
            let targets = targets.as_ref().unwrap();

            let aspect = drawable_width as f32 / drawable_height as f32;
            let f = 1.0 / (0.9f32 / 2.0).tan();
            let projection = |aspect: f32| {
                [
                    [f / aspect, 0.0, 0.0, 0.0],
                    [0.0, f, 0.0, 0.0],
                    [0.0, 0.0, 10.0 / 9.9, 1.0],
                    [0.0, 0.0, -1.0 / 9.9, 0.0],
                ]
            };
            let view = |spin: f32| {
                let (sin, cos) = spin.sin_cos();
                [
                    [cos, sin * 0.35, -sin * 0.94, 0.0],
                    [0.0, 0.94, 0.35, 0.0],
                    [sin, -cos * 0.35, cos * 0.94, 0.0],
                    [0.0, 0.0, 2.6, 1.0],
                ]
            };

            let slot = (frame % FRAMES_IN_FLIGHT) as u32 * 4;
            let query_base = (frame % FRAMES_IN_FLIGHT) as usize * QUERIES * 8;
            data[slot].mvp = f32x4x4(mul(projection(aspect), view(angle)));
            data[slot].tint = f32x4([1.0, 1.0, 1.0, 1.0]);
            data[slot + 1].tint = f32x4([0.30, 0.55, 0.95, angle.fract()]);
            data[slot + 2].mvp = f32x4x4(mul(projection(1.0), view(angle * 1.7)));
            data[slot + 2].tint = f32x4([1.0, 0.55, 0.25, 1.0]);
            data[slot + 2].layer = 0;
            data[slot + 3].mvp = f32x4x4(mul(projection(1.0), view(-angle * 0.9)));
            data[slot + 3].tint = f32x4([0.35, 1.0, 0.65, 1.0]);
            data[slot + 3].layer = 1;

            let composite_slot = (frame % FRAMES_IN_FLIGHT) as u32;
            composites[composite_slot].size = u32x2([drawable_width as u32, drawable_height as u32]);

            allocators[(frame % FRAMES_IN_FLIGHT) as usize].reset();
            command_buffer
                .begin_command_buffer_with_allocator(&allocators[(frame % FRAMES_IN_FLIGHT) as usize]);

            let layer_viewport =
                Viewport::new(0.0, 0.0, LAYER_SIZE as f64, LAYER_SIZE as f64, 0.0, 1.0);
            let viewport = Viewport::new(
                0.0,
                0.0,
                drawable_width as f64,
                drawable_height as f64,
                0.0,
                1.0,
            );

            let layers = Pass::new()
                .color(0, &layer_color.texture, Some(ClearColor::new(0.02, 0.02, 0.03, 1.0)))
                .depth(&layer_depth.texture, Some(1.0), false)
                .layers(LAYERS);
            {
                let Some(raster) = Raster::new(&command_buffer, &layers, &table, layer_viewport)
                else {
                    return;
                };
                raster.cull(CullMode::Back, Winding::Clockwise);
                raster.pipeline(&layered);
                raster.depth(&depth_write);
                for slice in 0..LAYERS as u32 {
                    raster.draw_indexed(
                        data.gpu(slot + 2 + slice),
                        PrimitiveType::Triangle,
                        index_ptr,
                        indices.len(),
                    );
                }
            }

            let prepass_desc = Pass::new()
                .depth(&targets.msaa_depth.texture, Some(1.0), true)
                .stencil(&targets.msaa_stencil.texture, Some(0), true);
            {
                let Some(raster) = Raster::new(&command_buffer, &prepass_desc, &table, viewport)
                else {
                    return;
                };
                raster.wait(
                    Stages::FRAGMENT,
                    Stages::VERTEX | Stages::FRAGMENT,
                    M4VisibilityOptions::DEVICE,
                );
                raster.cull(CullMode::Back, Winding::Clockwise);
                raster.pipeline(&prepass);
                raster.depth(&stamp);
                raster.stencil_reference(CUBE_STENCIL);
                raster.draw_indexed(
                    data.gpu(slot),
                    PrimitiveType::Triangle,
                    index_ptr,
                    indices.len(),
                );
            }

            let Ok(main) = Pass::new()
                .msaa(
                    0,
                    &targets.msaa_color.texture,
                    &targets.resolved_color.texture,
                    Some(ClearColor::new(0.04, 0.04, 0.06, 1.0)),
                )
                .stencil(&targets.msaa_stencil.texture, None, false)
                .occlusion(&queries, false)
                .msaa_depth(
                    &targets.msaa_depth.texture,
                    &targets.resolved_depth.texture,
                    None,
                    MultisampleDepthResolveFilter::Min,
                )
            else {
                return;
            };
            {
                let Some(raster) = Raster::new(&command_buffer, &main, &table, viewport) else {
                    return;
                };
                raster.wait(
                    Stages::FRAGMENT,
                    Stages::VERTEX | Stages::FRAGMENT,
                    M4VisibilityOptions::DEVICE,
                );
                raster.stencil_reference(CUBE_STENCIL);

                raster.cull(CullMode::Back, Winding::Clockwise);
                raster.pipeline(&opaque);
                raster.depth(&inside);
                raster.occlusion(VisibilityResultMode::Counting, query_base);
                raster.draw_indexed(
                    data.gpu(slot),
                    PrimitiveType::Triangle,
                    index_ptr,
                    indices.len(),
                );

                raster.cull(CullMode::None, Winding::Clockwise);
                raster.pipeline(&backdrop);
                raster.depth(&outside);
                raster.occlusion(VisibilityResultMode::Counting, query_base + 8);
                raster.draw(data.gpu(slot + 1), PrimitiveType::Triangle, 0, 3);
                raster.occlusion(VisibilityResultMode::Disabled, 0);
            }

            let screen = Pass::new()
                .color(0, &texture, Some(ClearColor::new(0.0, 0.0, 0.0, 1.0)));
            {
                let Some(raster) = Raster::new(&command_buffer, &screen, &table, viewport) else {
                    return;
                };
                raster.wait(
                    Stages::FRAGMENT,
                    Stages::FRAGMENT,
                    M4VisibilityOptions::DEVICE,
                );
                let (tile_width, tile_height) = (raster.encoder.tile_width(), raster.encoder.tile_height());
                composites[composite_slot].tile = u32x2([tile_width as u32, tile_height as u32]);

                raster.cull(CullMode::None, Winding::Clockwise);
                raster.pipeline(&composite);
                raster.depth(&depth_off);
                raster.draw(composites.gpu(composite_slot), PrimitiveType::Triangle, 0, 3);

                raster.pipeline(&vignette);
                raster.dispatch_tile(composites.gpu(composite_slot), tile_width, tile_height);
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
