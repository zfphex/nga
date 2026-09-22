//! A `ResourceHeap` allocates for the slots it asks for, not for the descriptor ceiling the
//! set layouts declare. On Vulkan those two numbers differ by six orders of magnitude — the
//! layouts are variable-count and sized to the hardware limit — so this renders through a
//! heap of eight slots and checks the sampled texel came back, with validation switched on.

use nga::*;
use table::{Data, Palette};

#[gpu_mod]
mod table {
    use nga::*;

    pub struct Palette {
        pub texture: Texture2d,
        pub samp: Sampler,
    }

    #[repr(C, align(16))]
    pub struct Data {
        pub palette: Gpu<Palette>,
    }

    pub struct VertexOut {
        pub position: f32x4,
        pub uv: f32x2,
        pub render_target_array_index: u32,
    }

    #[vertex]
    fn vertex(#[vertex_id] vid: u32) -> VertexOut {
        let uv = f32x2::new((vid % 2) as f32 * 2.0, (vid / 2) as f32 * 2.0);
        VertexOut {
            position: f32x4::new(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0),
            uv: uv,
            render_target_array_index: 1,
        }
    }

    #[fragment]
    fn fragment(root: Gpu<Data>, input: VertexOut) -> f32x4 {
        let palette = root.palette;
        palette.texture.sample_level(palette.samp, input.uv, 1.0)
    }

    pub struct VertexOutSimple {
        pub position: f32x4,
        pub uv: f32x2,
    }

    #[vertex]
    fn vertex_simple(#[vertex_id] vid: u32) -> VertexOutSimple {
        let uv = f32x2::new((vid % 2) as f32 * 2.0, (vid / 2) as f32 * 2.0);
        VertexOutSimple {
            position: f32x4::new(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0),
            uv: uv,
        }
    }

    #[fragment]
    fn fragment_offset(root: Gpu<Data>, input: VertexOutSimple) -> f32x4 {
        let palette = root.palette;
        palette
            .texture
            .offset(1)
            .sample_level(palette.samp, input.uv, 0.0)
    }
}

const SLOTS: u32 = 8;
const SIZE: u32 = 4;

#[test]
fn a_small_heap_addresses_a_layout_sized_to_the_device() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(true) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };
    assert!(
        device.max_resources() >= SLOTS,
        "a device that cannot hold {SLOTS} descriptors cannot run this"
    );

    let mut shaders = device.shaders(table::MODULE)?;
    let pipeline = shaders
        .raster(table::VERTEX, table::FRAGMENT)
        .color(Format::Rgba8Unorm)
        .build()?;

    let mut heap = GpuHeap::new(&device, 4 * 1024 * 1024)?;
    let mut resources = ResourceHeap::new(&device, SLOTS)?;

    // The last slot in the heap, so a descriptor lands at the far end of a buffer that is a
    // fraction of what the layout declares.
    let range = resources.alloc_range(SLOTS)?;
    let sampler = device.sampler(&SamplerDesc::nearest())?;
    let samp = resources.write_sampler(range, SLOTS - 2, &sampler);

    // Two levels, both filled by the caller, so the mip plumbing is under the validation layer
    // rather than only under a build. The fragment shader explicitly reads level one.
    let source =
        heap.alloc_texture(&TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE).mips(2))?;
    heap.upload(
        &source,
        0,
        &[0x20, 0x40, 0x60, 0xff].repeat((SIZE * SIZE) as usize),
    )?;
    heap.upload(
        &source,
        1,
        &[0x80, 0x80, 0x80, 0xff].repeat((SIZE * SIZE / 4) as usize),
    )?;
    heap.transition(&source, Usage::SAMPLED);
    let texture = resources.write_texture(range, SLOTS - 1, &source);

    let palette = heap.alloc_buffer(size_of::<Palette>())?;
    heap.write(&palette, 0, &[Palette { texture, samp }]);
    heap.flush()?;

    let target = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE)
            .mips(2)
            .layers(2)
            .usage(Usage::COLOR | Usage::COPY_SRC),
    )?;
    let readback = Arena::new(&device, (SIZE * SIZE * 4) as usize, Memory::Readback)?;

    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;
    let mut frame = ctx.begin_frame()?;
    let mut recorder = frame.recorder();
    {
        let pass =
            Pass::new()
                .layers(2)
                .color_texture_at(0, &target, 1, 0, Some([1.0, 0.0, 1.0, 1.0]));
        let raster = recorder.raster(&pass, SIZE / 2, SIZE / 2);
        raster.heap(&resources);
        raster.draw(
            pipeline,
            Data {
                palette: palette.gpu(),
            },
            0,
            3,
        )?;
    }
    let (cpu, gpu) = readback
        .alloc_bytes((SIZE * SIZE * 4) as usize, 16)
        .ok_or(Error::new("readback arena is full"))?;
    let compute = recorder.compute();
    compute.barrier(Stage::COLOR_OUT, Stage::TRANSFER, Hazard::TEXTURES);
    compute.read_texture_at(&target, 1, 1, &readback, unsafe { GpuRw::new(gpu) });
    drop(compute);
    drop(recorder);
    frame.submit(None)?;
    device.wait_idle()?;

    let pixels = unsafe { std::slice::from_raw_parts(cpu, (SIZE * SIZE * 4) as usize) };
    assert_eq!(
        &pixels[..4],
        &[0x80, 0x80, 0x80, 0xff],
        "the descriptor in the last heap slot did not reach the shader"
    );

    assert_eq!(
        device.validation_errors(),
        0,
        "the validation layer reported an error; see the output above"
    );

    heap.free_texture(target);
    heap.free_texture(source);
    heap.free_buffer(palette);
    Ok(())
}

#[test]
fn base_slot_offset_pattern() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(true) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };

    let mut shaders = device.shaders(table::MODULE)?;
    let pipeline = shaders
        .raster(table::VERTEX_SIMPLE, table::FRAGMENT_OFFSET)
        .color(Format::Rgba8Unorm)
        .build()?;

    let mut heap = GpuHeap::new(&device, 4 * 1024 * 1024)?;
    let mut resources = ResourceHeap::new(&device, 8)?;

    let range = resources.alloc_range(4)?;
    let sampler = device.sampler(&SamplerDesc::nearest())?;
    let samp = resources.write_sampler(range, 2, &sampler);

    // Texture 0 (slot 0): Red [0xff, 0x00, 0x00, 0xff]
    let tex0 = heap.alloc_texture(&TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE))?;
    heap.upload(
        &tex0,
        0,
        &[0xff, 0x00, 0x00, 0xff].repeat((SIZE * SIZE) as usize),
    )?;
    heap.transition(&tex0, Usage::SAMPLED);
    let base_texture = resources.write_texture(range, 0, &tex0);

    // Texture 1 (slot 1): Blue [0x00, 0x00, 0xff, 0xff]
    let tex1 = heap.alloc_texture(&TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE))?;
    heap.upload(
        &tex1,
        0,
        &[0x00, 0x00, 0xff, 0xff].repeat((SIZE * SIZE) as usize),
    )?;
    heap.transition(&tex1, Usage::SAMPLED);
    resources.write_texture(range, 1, &tex1);

    // Palette only stores base_texture (slot 0)
    let palette = heap.alloc_buffer(size_of::<Palette>())?;
    heap.write(
        &palette,
        0,
        &[Palette {
            texture: base_texture,
            samp,
        }],
    );
    heap.flush()?;

    let target = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE)
            .usage(Usage::COLOR | Usage::COPY_SRC),
    )?;
    let readback = Arena::new(&device, (SIZE * SIZE * 4) as usize, Memory::Readback)?;

    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;
    let mut frame = ctx.begin_frame()?;
    let mut recorder = frame.recorder();
    {
        let pass = Pass::new().color_texture_at(0, &target, 0, 0, Some([1.0, 0.0, 0.0, 1.0]));
        let raster = recorder.raster(&pass, SIZE, SIZE);
        raster.heap(&resources);
        raster.draw(
            pipeline,
            Data {
                palette: palette.gpu(),
            },
            0,
            3,
        )?;
    }
    let (cpu, gpu) = readback
        .alloc_bytes((SIZE * SIZE * 4) as usize, 16)
        .ok_or(Error::new("readback arena is full"))?;
    let compute = recorder.compute();
    compute.barrier(Stage::COLOR_OUT, Stage::TRANSFER, Hazard::TEXTURES);
    compute.read_texture_at(&target, 0, 0, &readback, unsafe { GpuRw::new(gpu) });
    drop(compute);
    drop(recorder);
    frame.submit(None)?;
    device.wait_idle()?;

    let pixels = unsafe { std::slice::from_raw_parts(cpu, (SIZE * SIZE * 4) as usize) };
    // The shader sampled `base_texture.offset(1)`, so it must read tex1 (Blue: 0, 0, 0xff, 0xff)
    assert_eq!(
        &pixels[..4],
        &[0x00, 0x00, 0xff, 0xff],
        "offset(1) did not sample texture at slot 1"
    );

    assert_eq!(device.validation_errors(), 0);

    heap.free_texture(target);
    heap.free_texture(tex0);
    heap.free_texture(tex1);
    heap.free_buffer(palette);
    Ok(())
}
