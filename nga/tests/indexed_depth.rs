#![cfg(target_vendor = "apple")]

use nga::*;

#[gpu_mod]
mod shader {
    use nga::*;
    pub struct Root {
        pub positions: Gpu<[f32x4]>,
        pub matrix: f32x4x4,
    }
    pub struct VertexOut {
        pub position: f32x4,
    }
    #[vertex]
    fn vertex(root: Gpu<Root>, #[vertex_id] id: u32) -> VertexOut {
        VertexOut {
            position: root.matrix.mul(root.positions[id]),
        }
    }
    #[fragment]
    fn fragment() {}
}

#[test]
#[ignore = "requires Metal 4"]
fn indexed_reversed_depth_atlas_matches_direct() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(true)?;
    let mut shaders = device.shaders(shader::MODULE)?;
    for (fixture, topology) in [
        (0, Topology::TriangleList),
        (1, Topology::TriangleList),
        (0, Topology::TriangleStrip),
        (1, Topology::LineList),
        (1, Topology::LineStrip),
    ] {
        let pipeline = shaders
            .raster(shader::VERTEX, shader::FRAGMENT)
            .indirect()
            .topology(topology)
            .depth(Format::D32Float)
            .build()?;
        let heap = GpuHeap::new(&device, 32 * 1048576)?;
        let mut positions = Vec::new();
        let mut draws = Vec::new();
        let mut selected = Vec::new();
        let mut ranges = Vec::new();
        let mut individual = Vec::new();
        if fixture == 0 {
            positions.extend_from_slice(&[
                f32x4::new(-0.8, -0.9, 0.001, 0.3),
                f32x4::new(1.2, -0.9, 0.002, 0.6),
                f32x4::new(-0.8, 1.1, 0.003, 0.9),
                f32x4::new(1.2, 1.1, 0.004, 1.2),
            ]);
            draws.push(DrawIndexed {
                index_count: 6,
                instance_count: 1,
                index_start: 1,
                base_vertex: 0,
                base_instance: 2,
            });
            selected.push(draws[0]);
            ranges.push(0..1);
            individual.push(0..1);
        } else {
            for slot in 0..6 {
                let x = (slot % 3) as f32 * 0.6 - 0.9;
                let y = (slot / 3) as f32 * 0.9 - 0.85;
                let z = 0.001 + slot as f32 * 0.000173;
                positions.extend_from_slice(&[
                    f32x4::new(x, y, z, 1.0),
                    f32x4::new(x + 0.5, y, z, 1.0),
                    f32x4::new(x, y + 0.75, z, 1.0),
                    f32x4::new(x + 0.5, y + 0.75, z, 1.0),
                ]);
                draws.push(DrawIndexed {
                    index_count: 6,
                    instance_count: 1,
                    index_start: 1,
                    base_vertex: slot * 4,
                    base_instance: slot as u32 + 7,
                });
            }
            ranges.extend_from_slice(&[1..2, 3..6]);
            for range in &ranges {
                for slot in range.clone() {
                    selected.push(draws[slot as usize]);
                    individual.push(slot..slot + 1);
                }
            }
        }
        let positions = heap.alloc_slice(&positions)?;
        let indices = heap.alloc_slice(&[999u32, 0, 1, 2, 2, 1, 3])?;
        let commands = IndirectCommands::indexed::<u32>(&device, topology, &indices, &draws)?;
        let readback = Arena::new(&device, 24 * 1048576, Memory::Readback)?;
        let mut targets = Vec::new();
        let mut pixels = Vec::new();
        for _ in 0..3 {
            targets.push(
                heap.alloc_texture(
                    &TextureDesc::depth(Format::D32Float, 1536, 1024)
                        .usage(Usage::DEPTH_STENCIL | Usage::COPY_SRC),
                )?,
            );
            pixels.push(unsafe { readback.alloc_scratch::<f32>(1536 * 1024) }.unwrap());
        }
        let mut context = Context::new(&device, &shaders, 1048576, 1)?;
        let mut frame = context.begin_frame()?;
        let arena = frame.arena(0);
        let arguments = arena.alloc_slice(&selected).unwrap();
        let recorder = frame.recorder();
        for mode in 0..3 {
            let pass = Pass::new().depth(&targets[mode], Some(0.0), true);
            let raster = recorder.raster(&pass);
            raster.depth(&DepthState::new(CompareOp::GreaterEqual, true));
            for face in 0..6 {
                let x = face % 3 * 512;
                let y = face / 3 * 512;
                raster.viewport(
                    Viewport {
                        x: x as f32,
                        y: y as f32,
                        width: 512.0,
                        height: 512.0,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    },
                    Scissor {
                        x,
                        y,
                        width: 512,
                        height: 512,
                    },
                );
                let root = shader::Root {
                    positions: positions.gpu(),
                    matrix: f32x4x4([
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.1 + face as f32 * 0.01, 0.0],
                        [0.0, 0.0, 0.0001234, 1.0],
                    ]),
                };
                if mode == 0 {
                    raster.draw_indexed_indirect::<u32, _>(
                        pipeline,
                        root,
                        &indices,
                        arena,
                        arguments.gpu,
                        selected.len() as u32,
                    )?;
                } else {
                    raster.draw_indexed_commands(
                        pipeline,
                        root,
                        &commands,
                        if mode == 1 { &ranges } else { &individual },
                    )?;
                }
            }
        }
        let compute = recorder.compute();
        compute.barrier(Stage::DEPTH_STENCIL, Stage::TRANSFER, Hazard::TEXTURES);
        for index in 0..3 {
            compute.read_texture(&targets[index], &readback, unsafe {
                Gpu::<[u8]>::new(pixels[index].gpu.get()).as_rw()
            });
        }
        frame.submit(None)?;
        device.wait_idle()?;
        let mut changed = [0usize; 2];
        let expected = unsafe { pixels[0].cpu.as_ref() };
        assert!(expected.iter().any(|&depth| depth > 0.0));
        for mode in 1..3 {
            for (&a, &b) in expected.iter().zip(unsafe { pixels[mode].cpu.as_ref() }) {
                assert!(a.is_finite() && b.is_finite());
                changed[mode - 1] += (a.to_bits() != b.to_bits()) as usize;
            }
        }
        eprintln!(
            "indexed reversed-depth atlas: fixture={fixture} topology={topology:?} merged_changed={} individual_changed={}",
            changed[0], changed[1]
        );
        assert_eq!(changed, [0; 2]);
    }
    Ok(())
}
