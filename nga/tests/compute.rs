//! Headless end-to-end check: allocate, dispatch, read back. No window, no surface, so this
//! is safe to run from a build script or CI and is the exit criterion for the command layer.

use nga::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct Data {
    output: GpuMut<'static, u32>,
    count: u32,
}

const SHADERS: &[u8] = nga::shader!("fill");

const COUNT: usize = 256;

#[test]
fn fill_writes_every_element() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };

    let mut shaders = Shaders::new(&device, SHADERS)?;
    let pipeline = shaders.compute("fill_main", None)?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    // Readback memory is host-cached, which is what makes the CPU-side read below cheap
    // rather than an uncached crawl over the PCIe window.
    let output = Arena::new(&device, 64 * 1024, Memory::Readback)?;
    let slot = output.alloc_scratch::<u32>(COUNT).unwrap();
    let target: GpuMut<'static, u32> = GpuMut::new(slot.gpu.0.get());

    let mut frame = ctx.begin_frame()?;
    let data = frame
        .arena(0)
        .alloc(Data {
            output: target,
            count: COUNT as u32,
        })
        .unwrap();
    let root = data.gpu;

    let mut recorder = frame.recorder();
    {
        let compute = recorder.compute();
        compute.pipeline(pipeline);
        compute.dispatch(root, COUNT, 64);
        compute.barrier(Stage::COMPUTE, Stage::TRANSFER, Hazard::BUFFERS);
    }
    drop(recorder);
    frame.submit(None)?;
    device.wait_idle()?;

    let written: &[u32] = &slot;
    for (index, &value) in written.iter().enumerate() {
        assert_eq!(value, index as u32 * 3 + 1, "element {index} is wrong");
    }
    Ok(())
}
