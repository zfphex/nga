use nga::*;

#[gpu_mod]
mod shader {
    use nga::*;

    pub struct Root {
        pub positions: Gpu<[f32x4]>,
        pub batch: u32,
    }

    pub struct VertexOut {
        pub position: f32x4,
    }

    #[vertex]
    fn vertex(root: Gpu<Root>, #[vertex_id] index: u32) -> VertexOut {
        VertexOut {
            position: root.positions[index],
        }
    }

    #[fragment]
    fn fragment(root: Gpu<Root>, _input: VertexOut, #[primitive_id] primitive: u32) -> f32x4 {
        if root.batch != 0 {
            if primitive == 0 {
                return f32x4::new(0.0, 0.0, 1.0, 1.0);
            }
            return f32x4::new(1.0, 1.0, 0.0, 1.0);
        }
        if primitive == 0 {
            return f32x4::new(1.0, 0.0, 0.0, 1.0);
        }
        if primitive == 1 {
            return f32x4::new(0.0, 1.0, 0.0, 1.0);
        }
        f32x4::new(1.0, 0.0, 1.0, 1.0)
    }
}

#[test]
fn primitive_id_codegen() {
    let msl = nga::msl::module(shader::MODULE);
    assert!(msl.contains("[[primitive_id]]"));
    let words = nga::spirv::words(shader::MODULE);
    let mut geometry = false;
    let mut primitive = false;
    let mut cursor = 5;
    while cursor < words.len() {
        let count = (words[cursor] >> 16) as usize;
        assert!(count > 0 && cursor + count <= words.len());
        let instruction = &words[cursor..cursor + count];
        if instruction[0] & 0xffff == 17 && count == 2 && instruction[1] == 2 {
            geometry = true;
        }
        if instruction[0] & 0xffff == 71
            && count == 4
            && instruction[2] == 11
            && instruction[3] == 7
        {
            primitive = true;
        }
        cursor += count;
    }
    assert!(geometry && primitive);
}

#[cfg(target_vendor = "apple")]
#[test]
#[ignore = "requires Metal 4"]
fn indexed_primitives_ignore_index_offset_and_reset_per_draw()
-> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(true)?;
    let mut shaders = device.shaders(shader::MODULE)?;
    let pipeline = shaders
        .raster(shader::VERTEX, shader::FRAGMENT)
        .indirect()
        .color(Format::Rgba8Unorm)
        .build()?;
    let heap = GpuHeap::new(&device, 1048576)?;
    let positions = heap.alloc_slice(&[
        f32x4::new(-1.0, -1.0, 0.0, 1.0),
        f32x4::new(1.0, -1.0, 0.0, 1.0),
        f32x4::new(1.0, 1.0, 0.0, 1.0),
        f32x4::new(-1.0, 1.0, 0.0, 1.0),
        f32x4::new(-1.0, -1.0, 0.0, 1.0),
        f32x4::new(3.0, -1.0, 0.0, 1.0),
        f32x4::new(-1.0, 3.0, 0.0, 1.0),
    ])?;
    let indices = heap.alloc_slice(&[999u32, 0, 1, 2, 0, 2, 3, 4, 5, 6])?;
    let targets = [
        heap.alloc_texture(
            &TextureDesc::texture_2d(Format::Rgba8Unorm, 4, 4)
                .usage(Usage::COLOR | Usage::COPY_SRC),
        )?,
        heap.alloc_texture(
            &TextureDesc::texture_2d(Format::Rgba8Unorm, 4, 4)
                .usage(Usage::COLOR | Usage::COPY_SRC),
        )?,
    ];
    let readback = Arena::new(&device, 4096, Memory::Readback)?;
    let first = readback.alloc_slice(&[0u8; 64]).unwrap();
    let second = readback.alloc_slice(&[0u8; 64]).unwrap();
    let persistent = IndirectCommands::indexed::<u32>(
        &device,
        Topology::TriangleList,
        &indices,
        &[
            DrawIndexed {
                index_count: 6,
                instance_count: 1,
                index_start: 1,
                base_vertex: 0,
                base_instance: 0,
            },
            DrawIndexed {
                index_count: 3,
                instance_count: 1,
                index_start: 7,
                base_vertex: 0,
                base_instance: 0,
            },
        ],
    )?;
    assert!(persistent.allocated_size() > 0);
    assert!(
        IndirectCommands::indexed::<u32>(
            &device,
            Topology::TriangleList,
            &indices,
            &[DrawIndexed {
                index_count: 3,
                instance_count: 1,
                index_start: 9,
                base_vertex: 0,
                base_instance: 0
            },]
        )
        .is_err()
    );
    let mut context = Context::new(&device, &shaders, 16384, 1)?;
    for native in [false, true, true] {
        let mut frame = context.begin_frame()?;
        let arena = frame.arena(0);
        let draws = arena
            .alloc_slice(&[
                DrawIndexed {
                    index_count: 6,
                    instance_count: 1,
                    index_start: 1,
                    base_vertex: 0,
                    base_instance: 0,
                },
                DrawIndexed {
                    index_count: 3,
                    instance_count: 1,
                    index_start: 7,
                    base_vertex: 0,
                    base_instance: 0,
                },
            ])
            .unwrap();
        let recorder = frame.recorder();
        for batch in 0..2 {
            let pass = Pass::new().color_texture(0, &targets[batch], Some([0.0; 4]));
            let commands = unsafe {
                Gpu::new(draws.gpu.get() + batch as u64 * size_of::<DrawIndexed>() as u64)
            };
            let root = shader::Root {
                positions: positions.gpu(),
                batch: batch as u32,
            };
            if native {
                recorder.raster(&pass).draw_indexed_commands(
                    pipeline,
                    root,
                    &persistent,
                    &[batch as u32..batch as u32 + 1],
                )?;
            } else {
                recorder.raster(&pass).draw_indexed_indirect::<u32, _>(
                    pipeline, root, &indices, arena, commands, 1,
                )?;
            }
        }
        let compute = recorder.compute();
        compute.barrier(Stage::COLOR_OUT, Stage::TRANSFER, Hazard::TEXTURES);
        compute.read_texture(&targets[0], &readback, first.gpu.as_rw());
        compute.read_texture(&targets[1], &readback, second.gpu.as_rw());
        frame.submit(None)?;
        device.wait_idle()?;
        assert_eq!(&first[..4], &[0, 255, 0, 255]);
        assert_eq!(&first[60..64], &[255, 0, 0, 255]);
        for pixel in first.chunks_exact(4) {
            assert!(pixel == [255, 0, 0, 255] || pixel == [0, 255, 0, 255]);
        }
        for pixel in second.chunks_exact(4) {
            assert_eq!(pixel, [0, 0, 255, 255]);
        }
    }
    Ok(())
}
