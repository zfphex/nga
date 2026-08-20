//! The two pieces of lifetime plumbing that are not the frame loop: a caller-owned counter,
//! and releasing a heap allocation only once the GPU has stopped reading it.

use nga::*;

#[test]
fn a_fence_counts_up_and_can_be_waited_on() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };

    let fence = Fence::new(&device)?;
    assert_eq!(fence.value(), 0);
    // Already reached, so this must not block.
    assert!(fence.wait(0));

    fence.signal(7);
    assert_eq!(fence.value(), 7);
    assert!(fence.wait(7));
    assert!(fence.wait(3), "waiting for a value already passed");
    Ok(())
}

#[test]
fn deferred_frees_land_when_the_frame_completes() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };

    // Over half the heap, so a second allocation of the same size only fits once the first is back.
    const SIZE: usize = 1 << 20;
    let mut heap = GpuHeap::new(&device, SIZE as u64)?;
    let buffer = heap.alloc_buffer(SIZE * 3 / 4)?;
    assert!(
        heap.alloc_buffer(SIZE * 3 / 4).is_err(),
        "the heap should be full with one buffer outstanding"
    );

    heap.free_later(4, Retired::Buffer(buffer));
    heap.retire(4);
    assert!(
        heap.alloc_buffer(SIZE * 3 / 4).is_err(),
        "frame 4 has not completed, so nothing may be released yet"
    );

    heap.retire(5);
    let second = heap.alloc_buffer(SIZE * 3 / 4)?;
    heap.free_buffer(second);
    Ok(())
}
