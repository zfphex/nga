//! The two things `nga-shaders` has no shader for: a specialization constant reaching the
//! module, and an atomic counting every invocation.

use nga::*;

use nga_shaders::fill::Data;

/// A specialization constant per field, numbered by field index — the same number
/// `Constants::values` pushes, so the host and the module cannot disagree.
#[gpu_struct]
#[derive(Constants)]
struct Options {
    stride: u32,
    bias: u32,
}

#[gpu(compute(64))]
fn fill_scaled(root: Gpu<'static, Data>, #[thread_id] id: u32x3, #[constants] options: Options) {
    if id.x < root.count {
        let mut out = root.output;
        out[id.x] = id.x * options.stride + options.bias;
    }
}

const COUNT: usize = 256;

#[test]
fn specialization_reaches_the_module() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };

    let mut shaders = Shaders::new(&device, &nga::compile(&[&FILL_SCALED]))?;
    let options = Options {
        stride: 7,
        bias: 2,
    };
    let pipeline = shaders.compute("fill_scaled", Some(&options))?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let output = Arena::new(&device, 64 * 1024, Memory::Readback)?;
    let slot = output.alloc_scratch::<u32>(COUNT).unwrap();

    let mut frame = ctx.begin_frame()?;
    let data = frame
        .arena(0)
        .alloc(Data {
            output: GpuMut::new(slot.gpu.0.get()),
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
        assert_eq!(value, index as u32 * 7 + 2, "element {index} is wrong");
    }
    Ok(())
}

#[gpu_struct]
struct Counter {
    total: GpuMut<'static, AtomicU32>,
    count: u32,
}

#[gpu(compute(64))]
fn count_up(root: Gpu<'static, Counter>, #[thread_id] id: u32x3) {
    if id.x < root.count {
        let mut total = root.total;
        total.atomic_add(1);
    }
}

#[test]
fn atomics_count_every_invocation() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };

    let mut shaders = Shaders::new(&device, &nga::compile(&[&COUNT_UP]))?;
    let pipeline = shaders.compute("count_up", None)?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let output = Arena::new(&device, 64 * 1024, Memory::Readback)?;
    let slot = output.alloc_scratch::<u32>(1).unwrap();

    let mut frame = ctx.begin_frame()?;
    let data = frame
        .arena(0)
        .alloc(Counter {
            total: GpuMut::new(slot.gpu.0.get()),
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
    assert_eq!(written[0], COUNT as u32, "every invocation should have counted");
    Ok(())
}

/// The Metal form: a function constant per field, at the same index the host pushes.
#[test]
fn metal_declares_function_constants() {
    let source = nga::msl::module(&[&FILL_SCALED, &COUNT_UP]);
    println!("{source}");
    assert!(source.contains("constant uint nga_Options_stride [[function_constant(0)]];"));
    assert!(source.contains("constant uint nga_Options_bias [[function_constant(1)]];"));
    assert!(source.contains("atomic_fetch_add_explicit"));
}
