//! The full M3 chain, headless: upload three textures, create a sampler, write all four into
//! the resource heap, and have a shader reach every one of them through a handle.
//!
//! This is the test that would catch a heap-write bug, a sampler-descriptor size mistake, or a
//! handle whose meaning drifted between the host and the prelude.

use nga::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct Data {
    layers: [Texture2d; 3],
    samp: Sampler,
    target: Texture2dRW,
    size: [u32; 2],
}

const SHADERS: &[u8] = nga::shader!("sample");

const TEX: u32 = 8;
const SIZE: u32 = 96;

/// Three flat colours, one per layer, so which texture was sampled is unambiguous.
const COLORS: [[u8; 4]; 3] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];

#[test]
fn shader_samples_through_bindless_handles() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let pipeline = shaders.compute("sample_main", None)?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut heap = GpuHeap::new(&device, 32 * 1024 * 1024)?;
    let mut resources = ResourceHeap::new(&device, 64)?;
    let range = resources.alloc_range(8)?;

    let mut textures = Vec::new();
    let mut layers = [Texture2d::default(); 3];
    for (index, color) in COLORS.iter().enumerate() {
        let texture = heap.alloc_texture(
            &TextureDesc::texture_2d(Format::Rgba8Unorm, TEX, TEX).usage(Usage::SAMPLED),
        )?;
        let pixels: Vec<u8> = color.repeat((TEX * TEX) as usize);
        heap.upload(&texture, &pixels)?;
        layers[index] = resources.write_texture(range, index as u32, &texture);
        textures.push(texture);
    }
    heap.flush()?;

    let sampler = device.sampler(&SamplerDesc::nearest())?;
    let samp = resources.write_sampler(range, 0, &sampler);

    let target = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE)
            .usage(Usage::STORAGE | Usage::COPY_SRC),
    )?;
    let target_handle = resources.write_storage(range, 3, &target);

    let readback = Arena::new(&device, 1024 * 1024, Memory::Readback)?;
    let pixels = readback
        .alloc_scratch::<u8>((SIZE * SIZE * 4) as usize)
        .unwrap();
    let dest: GpuMut<'_, [u8]> = GpuMut::new(pixels.gpu.0.get());

    let mut frame = ctx.begin_frame()?;
    let data = frame
        .arena(0)
        .alloc(Data {
            layers,
            samp,
            target: target_handle,
            size: [SIZE, SIZE],
        })
        .unwrap();
    let root = data.gpu;

    let mut recorder = frame.recorder();
    {
        let compute = recorder.compute();
        compute.heap(&resources);
        compute.pipeline(pipeline);
        for texture in &textures {
            compute.transition(texture, Usage::SAMPLED);
        }
        compute.transition(&target, Usage::STORAGE);
        compute.dispatch_2d(root, SIZE, SIZE, 8);
        compute.barrier(Stage::COMPUTE, Stage::TRANSFER, Hazard::TEXTURES);
        compute.read_texture(&target, &readback, dest);
    }
    drop(recorder);
    frame.submit(None)?;
    device.wait_idle()?;

    // The shader picks a layer from the horizontal third, so each third of the output must be
    // the flat colour of the corresponding texture. Anything else means a handle went astray.
    let bytes: &[u8] = &pixels;
    let texel = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
    };

    for (third, expected) in COLORS.iter().enumerate() {
        let x = SIZE / 6 + third as u32 * SIZE / 3;
        let found = texel(x, SIZE / 2);
        assert_eq!(
            found, *expected,
            "the {third} third sampled the wrong texture at x={x}"
        );
    }

    heap.free_texture(target);
    for texture in textures {
        heap.free_texture(texture);
    }
    Ok(())
}
