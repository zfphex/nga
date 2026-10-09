use nga::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[path = "../examples/shaders/sync.rs"]
mod shaders;

#[test]
#[ignore = "requires a GPU and validation layers"]
fn recording_and_heap_lifetimes() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(true)?;
    let errors = device.validation_errors();
    {
        let heap = GpuHeap::with_ring(&device, 64 * 1024, 4096)?;
        for _ in 0..16 {
            let buffer = heap.alloc_buffer(64 * 1024)?;
            assert!(heap.alloc_buffer(1024).is_err());
            drop(buffer);
        }
        for _ in 0..16 {
            let texture =
                heap.alloc_texture(&TextureDesc::texture_2d(Format::Rgba8Unorm, 16, 16))?;
            drop(texture);
            let buffer = heap.alloc_buffer(64 * 1024)?;
            drop(buffer);
        }
        let buffer = heap.alloc_buffer(64 * 1024)?;
        heap.free_later(0, Retired::Buffer(buffer));
        heap.retire(0);
        assert!(heap.alloc_buffer(1024).is_err());
        heap.retire(1);
        let buffer = heap.alloc_buffer(64 * 1024)?;
        heap.free_buffer(buffer);
        let buffer = heap.alloc_buffer(64 * 1024)?;
        drop(buffer);
    }
    {
        let heap = GpuHeap::with_ring(&device, 64 * 1024, 4096)?;
        let texture = heap.alloc_texture(&TextureDesc::texture_2d(Format::Rgba8Unorm, 16, 16))?;
        heap.free_later(0, Retired::Texture(texture));
        let heap = heap;
        heap.retire(1);
        let buffer = heap.alloc_buffer(64 * 1024)?;
        heap.free_later(u64::MAX, Retired::Buffer(buffer));
    }
    {
        let shaders = device.shaders(shaders::MODULE)?;
        let heap = GpuHeap::with_ring(&device, 1024 * 1024, 4096)?;
        let target = heap.alloc_texture(
            &TextureDesc::texture_2d(Format::Rgba8Unorm, 16, 16)
                .usage(Usage::COLOR | Usage::COPY_SRC),
        )?;
        let buffer = heap.alloc_slice(&[0u32; 256])?;
        heap.write(&buffer, 0, &[1u32; 256])?;
        heap.flush()?;
        let readback = Arena::new(&device, 4096, Memory::Readback)?;
        let pixels = readback.alloc_slice(&[0u8; 16 * 16 * 4]).unwrap();
        let mut ctx = Context::new(&device, &shaders, 4096, 2)?;
        for _ in 0..4 {
            let before = ALLOCATIONS.load(Ordering::Relaxed);
            let mut frame = ctx.begin_frame()?;
            for rec in frame.recorders(&[2]) {
                rec.compute();
            }
            frame.submit(None)?;
            assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), before);
        }
        for _ in 0..4 {
            let submitted = ctx.frame();
            let mut frame = ctx.begin_frame()?;
            frame.recorder().compute();
            drop(frame);
            assert_eq!(ctx.frame(), submitted);
        }
        heap.upload(&target, 0, &[255u8; 16 * 16 * 4])?;
        let upload = heap.flush_async()?;
        let mut frame = ctx.begin_frame()?;
        frame.after_uploads(upload, &heap);
        frame
            .recorder()
            .compute()
            .read_texture(&target, &readback, pixels.gpu.as_rw());
        frame.submit(None)?;
        device.wait_idle()?;
        for &byte in pixels.iter() {
            assert_eq!(byte, 255);
        }
        for _ in 0..4 {
            let mut frame = ctx.begin_frame()?;
            let rec = frame.recorder();
            let compute = rec.compute();
            compute.fill_buffer(&buffer, 0, 1024, 0);
            let pass = Pass::new().color_texture(0, &target, Some([1.0, 0.0, 0.0, 1.0]));
            rec.raster(&pass);
            let pass = Pass::new().color_texture(0, &target, Some([0.0, 1.0, 0.0, 1.0]));
            rec.raster(&pass);
            let compute = rec.compute();
            compute.read_texture(&target, &readback, pixels.gpu.as_rw());
            let compute = rec.compute();
            compute.barrier(Stage::TRANSFER, Stage::TRANSFER, Hazard::BUFFERS);
            frame.submit(None)?;
            device.wait_idle()?;
            for pixel in pixels.chunks_exact(4) {
                assert_eq!(pixel, [0, 255, 0, 255]);
            }
        }
        let mut frame = ctx.begin_frame()?;
        let rec = frame.recorder();
        let pass = Pass::new().color_texture(0, &target, Some([0.0, 0.0, 1.0, 1.0]));
        rec.raster(&pass);
        frame.submit(None)?;
        let mut frame = ctx.begin_frame()?;
        let rec = frame.recorder();
        rec.compute()
            .read_texture(&target, &readback, pixels.gpu.as_rw());
        frame.submit(None)?;
        device.wait_idle()?;
        for pixel in pixels.chunks_exact(4) {
            assert_eq!(pixel, [0, 0, 255, 255]);
        }
        let mut frame = ctx.begin_frame()?;
        let recorders = frame.recorders(&[2]);
        std::thread::scope(|scope| {
            for rec in recorders {
                scope.spawn(move || {
                    rec.compute();
                });
            }
        });
        frame.submit(None)?;
        device.wait_idle()?;
    }
    assert_eq!(device.validation_errors(), errors);
    Ok(())
}
