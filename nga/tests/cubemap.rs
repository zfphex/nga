use nga::*;

#[gpu_mod]
mod shader {
    use nga::*;

    pub struct Root {
        pub cube: TextureCube,
        pub flat: Texture2d,
        pub sampler: Sampler,
    }

    pub struct VertexOut {
        pub position: f32x4,
    }

    #[vertex]
    fn vertex(#[vertex_id] vid: u32) -> VertexOut {
        let uv = f32x2::new((vid % 2) as f32 * 2.0, (vid / 2) as f32 * 2.0);
        VertexOut {
            position: f32x4::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0),
        }
    }

    #[fragment]
    fn fragment(root: Gpu<Root>, input: VertexOut) -> f32x4 {
        let pixel = input.position.x as u32;
        let face = pixel / 2;
        let mut direction = f32x3::new(1.0, 0.0, 0.0);
        if face == 1 {
            direction = f32x3::new(-1.0, 0.0, 0.0);
        }
        if face == 2 {
            direction = f32x3::new(0.0, 1.0, 0.0);
        }
        if face == 3 {
            direction = f32x3::new(0.0, -1.0, 0.0);
        }
        if face == 4 {
            direction = f32x3::new(0.0, 0.0, 1.0);
        }
        if face == 5 {
            direction = f32x3::new(0.0, 0.0, -1.0);
        }
        if root.cube.dimensions().x != 2 {
            return f32x4::splat(0.0);
        }
        root.cube
            .sample_level(root.sampler, direction, (pixel % 2) as f32)
            * root.flat.sample(root.sampler, f32x2::splat(0.5))
            * root.cube.sample(root.sampler, f32x3::new(1.0, 0.0, 0.0)).x
    }
}

#[test]
fn cubemap_shader_types() {
    let msl = nga::msl::module(shader::MODULE);
    assert!(msl.contains("nga_deref_texcube"));
    let words = nga::spirv::words(shader::MODULE);
    let mut dimensions = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        let opcode = words[offset] & 0xffff;
        if opcode == 25 {
            dimensions.push(words[offset + 3]);
        }
        offset += count;
    }
    assert!(dimensions.contains(&1));
    assert!(dimensions.contains(&3));
}

#[test]
#[ignore = "requires a GPU"]
fn cubemap_faces_and_mips() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(true)?;
    let mut shaders = device.shaders(shader::MODULE)?;
    let pipeline = shaders
        .raster(shader::VERTEX, shader::FRAGMENT)
        .color(Format::Rgba8Unorm)
        .build()?;
    let heap = GpuHeap::new(&device, 1024 * 1024)?;
    let cube = heap.alloc_texture(&TextureDesc::texture_cube(Format::Rgba8Unorm, 2).mips(2))?;
    assert_eq!(cube.layers(), 6);
    let colors = [
        [255u8, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
        [0, 255, 255, 255],
        [255, 0, 255, 255],
    ];
    for face in 0..6 {
        let color = colors[face];
        heap.upload_layer(&cube, 0, face as u32, &color.repeat(4))?;
        heap.upload_layer(
            &cube,
            1,
            face as u32,
            &[color[0] / 2, color[1] / 2, color[2] / 2, 255],
        )?;
    }
    assert!(heap.upload_layer(&cube, 0, 6, &[0; 16]).is_err());
    heap.transition(&cube, Usage::SAMPLED);
    let flat = heap.alloc_texture(&TextureDesc::texture_2d(Format::Rgba8Unorm, 1, 1))?;
    heap.upload(&flat, 0, &[255; 4])?;
    heap.transition(&flat, Usage::SAMPLED);
    let output = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, 12, 1).usage(Usage::COLOR | Usage::COPY_SRC),
    )?;
    let filter = device.sampler(&SamplerDesc::nearest())?;
    let mut resources = ResourceHeap::new(&device, 3)?;
    let range = resources.alloc_range(3)?;
    let root = shader::Root {
        cube: resources.write_cube(range, 0, &cube),
        flat: resources.write_texture(range, 1, &flat),
        sampler: resources.write_sampler(range, 2, &filter),
    };
    heap.flush()?;
    let readback = Arena::new(&device, 4096, Memory::Readback)?;
    let pixels = readback.alloc_slice(&[0u8; 48]).unwrap();
    let mut context = Context::new(&device, &shaders, 4096, 1)?;
    let mut frame = context.begin_frame()?;
    let recorder = frame.recorder();
    {
        let pass = Pass::new().color_texture(0, &output, Some([0.0; 4]));
        let raster = recorder.raster(&pass);
        raster.heap(&resources);
        raster.cull(Cull::None, Winding::CounterClockwise);
        raster.draw(pipeline, root, 0..3)?;
    }
    let compute = recorder.compute();
    compute.barrier(Stage::COLOR_OUT, Stage::TRANSFER, Hazard::TEXTURES);
    compute.read_texture(&output, &readback, pixels.gpu.as_rw());
    frame.submit(None)?;
    device.wait_idle()?;
    let pixels = unsafe { pixels.cpu.as_ref() };
    for face in 0..6 {
        for level in 0..2 {
            for channel in 0..4 {
                let expected = if channel == 3 {
                    255
                } else {
                    colors[face][channel] >> level
                };
                assert!(pixels[(face * 2 + level) * 4 + channel].abs_diff(expected) <= 1);
            }
        }
    }
    Ok(())
}
