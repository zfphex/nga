use gpu_driven::{BoundaryData, DrawArgs, DrawData, EncodeData, ExecutionRange, Instance, SimData};
use minwin::*;
use nga::*;

/// The shader, in Rust. See `triangle.rs` for what `#[gpu_mod]` does with it.
#[gpu_mod]
mod gpu_driven {
    use nga::*;

    const TAU: f32 = 6.2831853;

    pub struct Instance {
        pub center: f32x2,
        pub angle: f32,
        pub scale: f32,
        pub color: f32x4,
    }

    /// `VkDrawIndirectCommand`, laid out by the spec rather than by us.
    pub struct DrawArgs {
        pub vertex_count: u32,
        pub instance_count: u32,
        pub first_vertex: u32,
        pub first_instance: u32,
    }

    pub struct SimData {
        pub instances: GpuRw<[Instance]>,
        pub time: f32,
        pub count: u32,
    }

    pub struct EncodeData {
        pub instances: Gpu<[Instance]>,
        pub draws: GpuRw<[DrawArgs]>,
        pub range: GpuRw<ExecutionRange>,
        pub commands: Commands,
        pub cull_radius: f32,
        pub total: u32,
    }

    /// Metal's indirect-command execution range. Vulkan reads `length` as its draw count.
    pub struct ExecutionRange {
        pub location: u32,
        pub length: AtomicU32,
    }

    #[repr(C, align(16))]
    pub struct DrawData {
        pub instances: Gpu<[Instance]>,
        pub mvp: f32x4x4,
    }

    #[repr(C, align(16))]
    pub struct BoundaryData {
        pub mvp: f32x4x4,
        pub radius: f32,
        pub segments: u32,
    }

    #[compute(64)]
    fn simulate(root: Gpu<SimData>, #[thread_id] tid: u32x3) {
        let i = tid.x;
        if i >= root.count {
            return;
        }
        let side = 64u32;
        let cell =
            (f32x2::new((i % side) as f32, (i / side) as f32) + 0.5) / side as f32 * 2.0 - 1.0;
        let n = ((i as f32 * 12.9898).sin() * 43758.5453).fract();
        let phase = root.time * (0.4 + n) + n * TAU;
        let wheel = (f32x3::splat(TAU * n) + f32x3::new(0.0, 2.0, 4.0)).cos() * 0.5 + 0.5;

        let mut out = root.instances;
        out[i] = Instance {
            center: cell + f32x2::new(phase.cos(), phase.sin()) * 0.008,
            angle: phase,
            scale: 0.009 + 0.004 * n,
            color: f32x4::new(wheel.x, wheel.y, wheel.z, 1.0),
        };
    }

    /// Metal builds a real `render_command` here and binds a per-draw root pointer to it.
    /// Vulkan has no GPU-side encoder, so this writes a `VkDrawIndirectCommand` instead and
    /// smuggles the instance index through `first_instance`, which the vertex stage reads back
    /// as the base instance.
    #[compute(64)]
    fn encode(root: Gpu<EncodeData>, #[thread_id] tid: u32x3) {
        let i = tid.x;
        if i >= root.total {
            return;
        }
        let inst = root.instances[i];
        if inst.center.length() > root.cull_radius {
            return;
        }
        let mut range = root.range;
        let slot = range.length.atomic_add(1);
        let mut draws = root.draws;
        draws[slot] = DrawArgs {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: i,
        };
        // Metal encodes this compact slot into an inherited-state ICB. Vulkan already has the
        // DrawArgs record above, so its compiler lowers this call to no instructions.
        command_draw(root.commands, slot, false, 0, 3, 1, i);
    }

    pub struct VertexOut {
        pub position: f32x4,
        pub color: f32x4,
    }

    #[vertex]
    fn vertex(
        root: Gpu<DrawData>,
        #[vertex_id] vid: u32,
        #[base_instance] base: u32,
    ) -> VertexOut {
        let inst = root.instances[base];
        let a = inst.angle + vid as f32 * 2.0943951;
        let p = inst.center + f32x2::new(a.cos(), a.sin()) * inst.scale;
        VertexOut {
            position: root.mvp.mul(f32x4::new(p.x, p.y, 0.5, 1.0)),
            color: inst.color,
        }
    }

    #[fragment]
    fn fragment(input: VertexOut) -> f32x4 {
        input.color
    }

    /// A vertex stage that returns a bare `f32x4` is returning clip space and nothing else.
    #[vertex]
    fn boundary(root: Gpu<BoundaryData>, #[vertex_id] vid: u32) -> f32x4 {
        let a = (vid / 2) as f32 / root.segments as f32 * TAU;
        let mut r = root.radius + 0.004;
        if vid % 2 == 0 {
            r = root.radius - 0.004;
        }
        root.mvp
            .mul(f32x4::new(a.cos() * r, a.sin() * r, 0.25, 1.0))
    }

    #[fragment]
    fn boundary_fragment() -> f32x4 {
        f32x4::new(1.0, 1.0, 1.0, 0.6)
    }
}

const COUNT: usize = 4096;
const GROUP: usize = 64;
const SEGMENTS: u32 = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = create_window(
        "nga-vk M5 — gpu driven",
        None,
        900,
        900,
        true,
        WindowStyle::Standard,
    );

    let device = Device::new(cfg!(debug_assertions))?;
    eprintln!("device: {}", device.name());

    let (mut width, mut height) = window.scaled_size();
    let mut surface = device.surface(window.handle(), width as u32, height as u32)?;
    let mut shaders = device.shaders(gpu_driven::MODULE)?;
    let simulate = shaders.compute(gpu_driven::SIMULATE)?;
    let encode = shaders.compute(gpu_driven::ENCODE)?;
    let draw = shaders
        .raster(gpu_driven::VERTEX, gpu_driven::FRAGMENT)
        .color(surface.format())
        .indirect()
        .build()?;
    let boundary = shaders
        .raster(gpu_driven::BOUNDARY, gpu_driven::BOUNDARY_FRAGMENT)
        .color(surface.format())
        .blend(BlendMode::Alpha)
        .topology(Topology::TriangleStrip)
        .build()?;

    let mut heap = GpuHeap::new(&device, 64 * 1024 * 1024)?;
    let instances = heap.alloc_buffer(COUNT * size_of::<Instance>())?;
    let draws = heap.alloc_buffer(COUNT * size_of::<DrawArgs>())?;
    let range = heap.alloc_buffer(size_of::<ExecutionRange>())?;
    let commands = IndirectCommands::counted_draws(&device, COUNT as u32)?;

    let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

    let mut time = 0.0f32;
    let mut resized = false;

    while window.open() {
        if window.pressed(Key::Escape) {
            window.close();
        }

        window.draw(|win| {
            let (new_width, new_height) = win.scaled_size();
            if (new_width, new_height) != (width, height) || resized {
                (width, height) = (new_width, new_height);
                surface
                    .rebuild(&device, width as u32, height as u32)
                    .unwrap();
                resized = false;
            }
            if surface.width() == 0 || surface.height() == 0 {
                return;
            }

            let Some(drawable) = surface.acquire(&device) else {
                resized = true;
                return;
            };

            let aspect = surface.width() as f32 / surface.height() as f32;
            let mvp = f32x4x4([
                [1.0 / aspect, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            let cull_radius = 0.55 + 0.35 * (time * 0.6).sin();

            let mut frame = ctx.begin_frame().unwrap();
            let mut rec = frame.recorder();

            let sim_root = rec
                .arena()
                .alloc(SimData {
                    instances: instances.gpu_rw(),
                    time,
                    count: COUNT as u32,
                })
                .unwrap()
                .gpu;
            let encode_root = rec
                .arena()
                .alloc(EncodeData {
                    instances: instances.gpu(),
                    draws: draws.gpu_rw(),
                    range: range.gpu_rw(),
                    commands: commands.handle(),
                    cull_radius,
                    total: COUNT as u32,
                })
                .unwrap()
                .gpu;
            let draw_root = rec
                .arena()
                .alloc(DrawData {
                    instances: instances.gpu(),
                    mvp,
                })
                .unwrap()
                .gpu;
            let boundary_root = rec
                .arena()
                .alloc(BoundaryData {
                    mvp,
                    radius: cull_radius,
                    segments: SEGMENTS,
                })
                .unwrap()
                .gpu;

            {
                let compute = rec.compute();
                // Clear `{ location, length }`. Only compact slots below the new length execute,
                // so neither backend needs the entire draw array cleared.
                compute.fill_buffer(&range, 0, size_of::<ExecutionRange>(), 0);
                compute.barrier(Stage::TRANSFER, Stage::COMPUTE, Hazard::BUFFERS);

                compute.pipeline(simulate);
                compute.dispatch(sim_root, COUNT, GROUP);
                compute.barrier(Stage::COMPUTE, Stage::COMPUTE, Hazard::BUFFERS);

                compute.pipeline(encode);
                compute.dispatch(encode_root, COUNT, GROUP);
                compute.barrier(
                    Stage::COMPUTE,
                    Stage::INDIRECT | Stage::VERTEX,
                    Hazard::BUFFERS | Hazard::DRAW_ARGUMENTS,
                );
            }

            {
                let pass = Pass::new().color(0, &drawable, Some([0.02, 0.02, 0.04, 1.0]));
                let raster = rec.raster(&pass, surface.width(), surface.height());
                raster.pipeline(draw);
                raster.draw_indirect_count(draw_root, &commands, &draws, &range);

                raster.pipeline(boundary);
                raster.draw(boundary_root, 0, SEGMENTS * 2 + 2);
            }

            drop(rec);
            frame.submit(Some(&drawable)).unwrap();

            resized = surface.present(&device, &drawable);

            time += 0.016;
        });

        window.wait_for_vsync();
    }

    device.wait_idle()?;
    heap.free_buffer(range);
    heap.free_buffer(draws);
    heap.free_buffer(instances);
    Ok(())
}
