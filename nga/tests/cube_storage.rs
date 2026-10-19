use nga::*;

#[gpu_mod]
mod shader {
    use nga::*;

    pub struct Root {
        pub cube: TextureCubeRW,
        pub flat: Texture2dRW,
        pub mip: u32,
    }

    #[compute(4, 4)]
    fn store(root: Gpu<Root>, #[thread_id] tid: u32x3) {
        let size = 4u32 >> root.mip;
        if tid.x >= size || tid.y >= size || tid.z >= 6 {
            return;
        }
        let value = f32x4::new(tid.x as f32, tid.y as f32, tid.z as f32, root.mip as f32);
        root.cube.store(tid, value);
        if tid.z == 0 {
            root.flat.store(tid.xy(), value);
        }
    }
}

#[test]
fn storage_cube_shader_types() {
    let msl = nga::msl::module(shader::MODULE);
    assert!(msl.contains("nga_deref_texcube_rw(nga_push_data"));
    let words = nga::spirv::words(shader::MODULE);
    let mut images = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if words[offset] & 0xffff == 25 {
            images.push((words[offset + 3], words[offset + 7]));
        }
        offset += count;
    }
    assert!(images.contains(&(1, 2)));
    assert!(images.contains(&(3, 2)));
}

#[test]
#[ignore = "requires a GPU"]
fn storage_cube_faces_and_mips() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(true)?;
    let mut shaders = device.shaders(shader::MODULE)?;
    let pipeline = shaders.compute(shader::STORE)?;
    let heap = GpuHeap::new(&device, 1048576)?;
    let desc = TextureDesc::texture_cube(Format::Rgba32Float, 4)
        .mips(3)
        .usage(Usage::STORAGE | Usage::COPY_SRC);
    let cube = heap.alloc_texture(&desc)?;
    let flat = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba32Float, 4, 4).usage(Usage::STORAGE | Usage::COPY_SRC),
    )?;
    let mut resources = ResourceHeap::new(&device, 4)?;
    let range = resources.alloc_range(4)?;
    let output = resources.write_storage(range, 3, &flat);
    assert!(
        resources
            .write_storage_cube_mip(range, 0, &cube, 3)
            .is_err()
    );
    assert!(
        resources
            .write_storage_cube_mip(range, 0, &flat, 0)
            .is_err()
    );
    let mut handles = Vec::new();
    for mip in 0..3 {
        handles.push(resources.write_storage_cube_mip(range, mip, &cube, mip)?);
    }
    let readback = Arena::new(&device, 8192, Memory::Readback)?;
    let mut pairs = Vec::new();
    for mip in 0..3 {
        for face in 0..6 {
            pairs.push((
                mip,
                face,
                unsafe { readback.alloc_scratch::<f32x4>(((4u32 >> mip).pow(2)) as usize) }
                    .unwrap(),
            ));
        }
    }
    let flat_pixels = unsafe { readback.alloc_scratch::<f32x4>(16) }.unwrap();
    let mut context = Context::new(&device, &shaders, 8192, 1)?;
    let value = context.frame();
    let mut frame = context.begin_frame()?;
    let compute = frame.recorder().compute();
    compute.transition(&cube, Usage::STORAGE);
    compute.transition(&flat, Usage::STORAGE);
    compute.heap(&resources);
    for mip in 0..3 {
        compute.dispatch(
            pipeline,
            shader::Root {
                cube: handles[mip as usize],
                flat: output,
                mip,
            },
            DispatchThreads {
                threads: [4 >> mip, 4 >> mip, 6],
            },
        )?;
        compute.barrier(Stage::COMPUTE, Stage::COMPUTE, Hazard::TEXTURES);
    }
    compute.barrier(Stage::COMPUTE, Stage::TRANSFER, Hazard::TEXTURES);
    for (mip, face, pixels) in &pairs {
        compute.read_texture_at(&cube, *mip, *face, &readback, unsafe {
            Gpu::<[u8]>::new(pixels.gpu.get()).as_rw()
        });
    }
    compute.read_texture(&flat, &readback, unsafe {
        Gpu::<[u8]>::new(flat_pixels.gpu.get()).as_rw()
    });
    let occupied = heap.usage().allocated;
    heap.free_later(value, Retired::Texture(cube));
    let deferred = heap.usage().retired;
    assert!(deferred > 0);
    heap.retire(value);
    assert_eq!(heap.usage().allocated, occupied);
    let replacement = heap.alloc_texture(&desc)?;
    assert_eq!(heap.usage().allocated, occupied + deferred);
    drop(replacement);
    assert_eq!(heap.usage().allocated, occupied);
    assert_eq!(heap.usage().peak, occupied + deferred);
    frame.submit(None)?;
    device.wait_idle()?;
    heap.retire(context.completed());
    assert_eq!(heap.usage().allocated, occupied - deferred);
    assert_eq!(heap.usage().retired, 0);
    assert_eq!(heap.usage().peak, occupied + deferred);
    for (mip, face, pixels) in pairs {
        let size = 4usize >> mip;
        for (index, value) in unsafe { pixels.cpu.as_ref() }.iter().enumerate() {
            assert_eq!(
                [value.x, value.y, value.z, value.w],
                [
                    (index % size) as f32,
                    (index / size) as f32,
                    face as f32,
                    mip as f32
                ]
            );
        }
    }
    for (index, value) in unsafe { flat_pixels.cpu.as_ref() }.iter().enumerate() {
        let x = index % 4;
        let y = index / 4;
        let mip = if x == 0 && y == 0 {
            2.0
        } else if x < 2 && y < 2 {
            1.0
        } else {
            0.0
        };
        assert_eq!(
            [value.x, value.y, value.z, value.w],
            [x as f32, y as f32, 0.0, mip]
        );
    }
    assert_eq!(device.validation_errors(), 0);
    Ok(())
}
