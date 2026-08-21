//! M3's exit criterion, headless: a compute pipeline writes a bindless storage texture through
//! a handle the host put in the resource heap, and the result is read back and checked.
//!
//! Nothing in this file names Vulkan or Metal.

use nga::*;
use nga_shaders::gradient::{self, Data};


const SIZE: u32 = 64;

#[test]
fn compute_writes_a_bindless_storage_texture() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };

    let mut shaders = Shaders::new(&device, &nga::compile(gradient::ENTRIES))?;
    let pipeline = shaders.compute("gradient", None)?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut heap = GpuHeap::new(&device, 16 * 1024 * 1024)?;
    let texture = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE)
            .usage(Usage::STORAGE | Usage::COPY_SRC),
    )?;

    // The handle the shader dereferences is whatever the heap says it is. The host never
    // computes a descriptor index by hand.
    let mut resources = ResourceHeap::new(&device, 64)?;
    let range = resources.alloc_range(1)?;
    let target = resources.write_storage(range, 0, &texture);

    let readback = Arena::new(&device, 1024 * 1024, Memory::Readback)?;
    let pixels = readback
        .alloc_scratch::<u8>((SIZE * SIZE * 4) as usize)
        .unwrap();
    let dest: GpuMut<'_, [u8]> = GpuMut::new(pixels.gpu.0.get());

    let mut frame = ctx.begin_frame()?;
    let data = frame
        .arena(0)
        .alloc(Data {
            target,
            size: u32x2([SIZE, SIZE]),
        })
        .unwrap();
    let root = data.gpu;

    let mut recorder = frame.recorder();
    {
        let compute = recorder.compute();
        compute.heap(&resources);
        compute.pipeline(pipeline);
        compute.transition(&texture, Usage::STORAGE);
        compute.dispatch_2d(root, SIZE, SIZE, 8);
        compute.barrier(Stage::COMPUTE, Stage::TRANSFER, Hazard::TEXTURES);
        compute.read_texture(&texture, &readback, dest);
    }
    drop(recorder);
    frame.submit(None)?;
    device.wait_idle()?;

    // The shader writes uv.x into red and uv.y into green, so the gradient is checkable at
    // the corners without reproducing the filtering rules.
    let bytes: &[u8] = &pixels;
    let texel = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        (bytes[i], bytes[i + 1], bytes[i + 3])
    };

    assert_eq!(texel(0, 0), (0, 0, 255), "origin should be black, opaque");
    let (r, _, _) = texel(SIZE - 1, 0);
    assert!(r > 240, "red should approach full at the right edge, got {r}");
    let (_, g, _) = texel(0, SIZE - 1);
    assert!(g > 240, "green should approach full at the bottom, got {g}");

    let mid = texel(SIZE / 2, SIZE / 2);
    assert!(
        (120..=136).contains(&mid.0) && (120..=136).contains(&mid.1),
        "centre should be mid-grey in red and green, got {mid:?}"
    );

    heap.free_texture(texture);
    Ok(())
}
