//! A mesh pipeline drawing the same triangle `raster.rs` draws through the vertex path, so the
//! two can be compared pixel for pixel. Headless, and skipped where the backend has no mesh
//! pipeline, which is what `Feature::MeshShaders` is for.

// Mesh execution remains feature-gated even though the shader now uses the shared Rust IR.
use nga::*;
#[cfg(target_vendor = "apple")]
use shader::{Data, Vertex};

#[gpu_mod]
mod shader {
    use nga::*;

    pub struct Vertex {
        pub position: [f32; 3],
        pub color: [f32; 3],
    }

    pub struct Data {
        pub vertices: Gpu<[Vertex]>,
        pub mvp: f32x4x4,
    }

    pub struct VertexOut {
        pub position: f32x4,
        pub color: f32x3,
    }

    pub struct Payload {
        pub mesh: u32,
    }

    #[object]
    fn object_stage(#[payload] mut payload: Payload, #[thread_index_in_group] lane: u32) {
        let mut count = ThreadgroupAtomicU32::new(0);
        if lane == 0 {
            count.atomic_exchange(0);
        }
        threadgroup_barrier();
        if lane == 0 {
            payload.mesh = 0;
            mesh_set_threadgroups(u32x3::new(count.atomic_add(0) + 1, 1, 1));
        }
    }

    #[mesh(VertexOut, 3, 1)]
    fn mesh_main(root: Gpu<Data>, #[thread_index_in_group] lane: u32) {
        if lane == 0 {
            mesh_set_primitive_count(1);
            mesh_set_index(0, 0);
            mesh_set_index(1, 1);
            mesh_set_index(2, 2);
        }
        let source = root.vertices[lane];
        let vertex = VertexOut {
            position: root.mvp.mul(f32x4::new(
                source.position[0],
                source.position[1],
                source.position[2],
                1.0,
            )),
            color: f32x3::new(source.color[0], source.color[1], source.color[2]),
        };
        mesh_set_vertex(lane, vertex);
    }

    #[fragment]
    fn mesh_fragment(input: VertexOut) -> f32x4 {
        f32x4::new(input.color.x, input.color.y, input.color.z, 1.0)
    }
}

#[cfg(target_vendor = "apple")]
const SIZE: u32 = 128;

#[test]
fn mesh_shader_is_generated_from_rust() {
    let source = msl::module(shader::ENTRIES);
    assert!(source.contains("using NgaMeshOut_mesh_main = metal::mesh<VertexOut, void, 3, 1"));
    assert!(source.contains("[[mesh]] void mesh_main("));
    assert!(source.contains("nga_mesh_out.set_primitive_count("));
    assert!(source.contains("[[object]] void object_stage("));
    assert!(source.contains("object_data Payload& payload [[payload]]"));
    assert!(source.contains("threadgroup atomic_uint count;"));
    assert!(source.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
    assert!(source.contains("atomic_fetch_add_explicit(&(count),"));
    assert!(source.contains("nga_mesh_grid.set_threadgroups_per_grid("));
}

#[cfg(target_vendor = "apple")]
#[test]
fn a_mesh_pipeline_emits_a_triangle() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(device) = Device::new(cfg!(debug_assertions)) else {
        eprintln!("no supported device; skipping");
        return Ok(());
    };
    if !device.supports(Feature::MeshShaders) {
        eprintln!("{} has no mesh pipeline; skipping", device.name());
        return Ok(());
    }

    let mut shaders = device.shaders(shader::MODULE)?;
    let pipeline = shaders.mesh(&MeshDesc::new(
        "",
        "mesh_main",
        "mesh_fragment",
        &[ColorTarget::new(Format::Rgba8Unorm)],
        0,
    ))?;
    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut heap = GpuHeap::new(&device, 16 * 1024 * 1024)?;
    let target = heap.alloc_texture(
        &TextureDesc::texture_2d(Format::Rgba8Unorm, SIZE, SIZE)
            .usage(Usage::COLOR | Usage::COPY_SRC),
    )?;

    let white = [1.0, 1.0, 1.0];
    let mesh = Arena::new(&device, 64 * 1024, Memory::Device)?;
    let vertices = mesh
        .alloc_slice(&[
            Vertex {
                position: [0.0, 0.8, 0.0],
                color: white,
            },
            Vertex {
                position: [-0.8, -0.8, 0.0],
                color: white,
            },
            Vertex {
                position: [0.8, -0.8, 0.0],
                color: white,
            },
        ])
        .unwrap();
    let vertices: Gpu<[Vertex]> = vertices.gpu;

    let readback = Arena::new(&device, 1024 * 1024, Memory::Readback)?;
    // The render and copy below initialise every byte before the host reads it.
    let pixels = unsafe {
        readback
            .alloc_scratch::<u8>((SIZE * SIZE * 4) as usize)
            .unwrap()
    };
    let dest: GpuRw<[u8]> = pixels.gpu.as_rw();

    let mut frame = ctx.begin_frame()?;
    let data = frame
        .arena(0)
        .alloc(Data {
            vertices,
            mvp: f32x4x4([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        })
        .unwrap();
    let root = data.gpu;

    let mut recorder = frame.recorder();
    {
        let pass = Pass::new().color_texture(0, &target, Some([0.0, 0.0, 0.0, 1.0]));
        let raster = recorder.raster(&pass, SIZE, SIZE);
        raster.pipeline(pipeline);
        // One mesh threadgroup of three threads, one thread per vertex. No object stage, so the
        // object threadgroup size is unused.
        raster.draw_mesh(root, [1, 1, 1], [1, 1, 1], [3, 1, 1]);
    }
    {
        let compute = recorder.compute();
        compute.barrier(Stage::COLOR_OUT, Stage::TRANSFER, Hazard::TEXTURES);
        compute.read_texture(&target, &readback, dest);
    }
    drop(recorder);
    frame.submit(None)?;
    device.wait_idle()?;

    let bytes: &[u8] = &pixels;
    let luma = |x: u32, y: u32| bytes[((y * SIZE + x) * 4) as usize];
    assert_eq!(luma(2, 2), 0, "the top-left corner is outside the triangle");
    assert_eq!(
        luma(SIZE / 2, SIZE / 2),
        255,
        "the centre should be covered by the triangle"
    );

    // The same window `raster.rs` asserts, because it is the same triangle.
    let covered = bytes.chunks_exact(4).filter(|p| p[0] > 128).count();
    let total = (SIZE * SIZE) as usize;
    assert!(
        (total / 4..total / 2).contains(&covered),
        "{covered} of {total} pixels covered; that is not the triangle"
    );

    heap.free_texture(target);
    Ok(())
}
