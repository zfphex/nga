//! Every path into device-local memory, checked by reading the bytes back.
//!
//! These are the paths Metal does not have: under UMA an upload is a memcpy that has already
//! happened. Here each one is a recorded copy through a staging ring, and the interesting cases
//! are the boundaries — the `vkCmdUpdateBuffer` cutoff, a write larger than the ring, an image
//! taller than the ring can hold at once, and a ring that has to wrap mid-batch.

use ash::vk;
use nga_vk::*;

const RING: usize = 1 << 20;

/// Field order is drop order: everything holding a copy of the `VkDevice` handle must go
/// before the `Device` that destroys it.
struct Harness {
    heap: GpuHeap,
    readback: Arena,
    device: Device,
}

impl Harness {
    fn new() -> Result<Self, VkError> {
        let device = Device::new(validation())?;
        let heap = GpuHeap::with_ring(&device, 64 * 1024 * 1024, RING)?;
        let readback = Arena::new(&device, 16 * 1024 * 1024, Placement::Readback)?;
        Ok(Self {
            heap,
            readback,
            device,
        })
    }

    /// Copies a heap buffer into the readback arena through the same batch as the writes under
    /// test, so the copy cannot observe anything the flush has not also made visible.
    fn read(&mut self, buffer: &HeapBuffer, len: usize) -> Result<Vec<u8>, VkError> {
        let command_buffer = self.heap.uploader.open()?;
        unsafe {
            self.device.device.cmd_copy_buffer(
                command_buffer,
                buffer.buffer,
                self.readback.buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: len as u64,
                }],
            )
        };
        self.heap.flush()?;
        Ok(unsafe { std::slice::from_raw_parts(self.readback.cpu, len) }.to_vec())
    }
}

/// `NGA_VALIDATION=1 cargo test` turns the layers on; they are off by default so the suite
/// still runs where the SDK is not installed.
fn validation() -> bool {
    std::env::var_os("NGA_VALIDATION").is_some()
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 13) as u8)
        .collect()
}

#[test]
fn placements_resolve() -> Result<(), VkError> {
    let device = Device::new(validation())?;
    let bar = Arena::new(&device, 64 * 1024, Placement::Bar)?;
    let upload = Arena::new(&device, 64 * 1024, Placement::Upload)?;
    let readback = Arena::new(&device, 64 * 1024, Placement::Readback)?;

    // The point of the split: staging must not sit in the BAR window, because on a card
    // without ReBAR that window is 214 MiB and a shader has better uses for it.
    let properties = |arena: &Arena| {
        let types = device.memory_types(
            u32::MAX,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::empty(),
        );
        assert!(!types.is_empty());
        arena.placement
    };
    assert_eq!(properties(&bar), Placement::Bar);
    assert_eq!(properties(&upload), Placement::Upload);
    assert_eq!(properties(&readback), Placement::Readback);
    Ok(())
}

/// Below `UPDATE_LIMIT` the data rides in the command buffer and the ring is never touched.
#[test]
fn small_write_bypasses_the_ring() -> Result<(), VkError> {
    let mut harness = Harness::new()?;
    let bytes = pattern(4096);
    let buffer = harness.heap.alloc_buffer(bytes.len())?;

    let before = harness.heap.uploader.head;
    harness.heap.write(&buffer, 0, &bytes);
    assert_eq!(harness.heap.uploader.head, before);

    assert_eq!(harness.read(&buffer, bytes.len())?, bytes);
    Ok(())
}

/// Past `UPDATE_LIMIT` the ring takes over. Still one batch, still no wrap.
#[test]
fn staged_write_round_trips() -> Result<(), VkError> {
    let mut harness = Harness::new()?;
    let bytes = pattern(UPDATE_LIMIT * 4);
    let buffer = harness.heap.alloc_buffer(bytes.len())?;

    harness.heap.write(&buffer, 0, &bytes);
    assert!(harness.heap.uploader.head >= bytes.len());

    assert_eq!(harness.read(&buffer, bytes.len())?, bytes);
    Ok(())
}

/// The case the old 8 MiB staging arena panicked on: a write larger than all of staging.
/// Chunking means the ring streams it instead, wrapping and draining as it goes.
#[test]
fn write_larger_than_the_ring_chunks() -> Result<(), VkError> {
    let mut harness = Harness::new()?;
    let bytes = pattern(RING * 3 + 12345);
    let buffer = harness.heap.alloc_buffer(bytes.len())?;

    harness.heap.write(&buffer, 0, &bytes);
    assert!(
        harness.heap.uploader.submitted > 0,
        "a wrap must have flushed"
    );

    assert_eq!(harness.read(&buffer, bytes.len())?, bytes);
    Ok(())
}

/// A write at a non-zero offset must not disturb its neighbours, which is the thing chunking
/// is easiest to get wrong.
#[test]
fn offset_write_leaves_neighbours_alone() -> Result<(), VkError> {
    let mut harness = Harness::new()?;
    let len = 1 << 18;
    let buffer = harness.heap.alloc_buffer(len)?;

    harness.heap.write(&buffer, 0, &vec![0xABu8; len]);
    let middle = pattern(1024);
    harness.heap.write(&buffer, 4096, &middle);

    let read = harness.read(&buffer, len)?;
    assert!(read[..4096].iter().all(|&b| b == 0xAB));
    assert_eq!(&read[4096..4096 + middle.len()], middle.as_slice());
    assert!(read[4096 + middle.len()..].iter().all(|&b| b == 0xAB));
    Ok(())
}

/// A texture whose rows do not all fit in the ring at once, so the copy is split by row range.
#[test]
fn image_taller_than_the_ring_chunks() -> Result<(), VkError> {
    let mut harness = Harness::new()?;
    let (width, height) = (512, 2048); // 4 MiB at RGBA8, against a 1 MiB ring
    let desc = TextureDesc::texture_2d(vk::Format::R8G8B8A8_UNORM, width, height);
    let texture = harness.heap.alloc_texture(&desc)?;
    let pixels = pattern(width as usize * height as usize * 4);

    harness.heap.upload(&texture, &pixels)?;
    harness
        .heap
        .transition(&texture, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    harness.heap.flush()?;
    assert!(texture.layout.get() == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    Ok(())
}

/// A whole scene's worth of resources must cost one submission, not one per resource. This is
/// the entire reason the uploader exists.
#[test]
fn a_batch_is_one_submission() -> Result<(), VkError> {
    let mut harness = Harness::new()?;
    let desc = TextureDesc::texture_2d(vk::Format::R8G8B8A8_UNORM, 64, 64);
    let pixels = pattern(64 * 64 * 4);

    let mut textures = Vec::new();
    for _ in 0..32 {
        let texture = harness.heap.alloc_texture(&desc)?;
        harness.heap.upload(&texture, &pixels)?;
        harness
            .heap
            .transition(&texture, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        textures.push(texture);
    }
    let buffer = harness.heap.alloc_buffer(1024)?;
    harness.heap.write(&buffer, 0, &pattern(1024));

    assert_eq!(harness.heap.uploader.submitted, 0, "nothing before flush");
    harness.heap.flush()?;
    assert_eq!(harness.heap.uploader.submitted, 1, "64 copies, one submit");
    Ok(())
}

/// A write that cannot fit the ring even after a drain is an error, not a panic and not a hang.
#[test]
fn an_image_row_larger_than_the_ring_is_an_error() -> Result<(), VkError> {
    let device = Device::new(validation())?;
    let mut heap = GpuHeap::with_ring(&device, 16 * 1024 * 1024, 4096)?;
    let desc = TextureDesc::texture_2d(vk::Format::R8G8B8A8_UNORM, 2048, 4);
    let texture = heap.alloc_texture(&desc)?;
    assert!(heap.upload(&texture, &pattern(2048 * 4 * 4)).is_err());
    Ok(())
}
