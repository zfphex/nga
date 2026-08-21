//! Every shader in the tree, in Rust.
//!
//! One module per former `.slang` file. The structs live here too, not in the example that
//! draws with them, because they are the ABI: the host writes them and the shader reads them,
//! and there is now exactly one definition of each rather than two that have to be kept in
//! step. That is the whole reason the shaders are their own crate.
//!
//! Nothing in here is called on the host. The bodies compile as ordinary Rust so that `rustc`
//! type-checks them, and `#[gpu]` records the same body as IR for `nga::compile`.

#![allow(clippy::needless_range_loop)]

use nga::*;

const TAU: f32 = 6.2831853;

pub mod fill {
    use super::*;

    /// Every entry point in this module, which is what one library holds.
    pub const ENTRIES: &[&nga::ir::GpuFn] = &[&FILL];

    #[gpu_struct]
    pub struct Data {
        pub output: GpuMut<'static, [u32]>,
        pub count: u32,
    }

    #[gpu(compute(64))]
    fn fill(root: Gpu<'static, Data>, #[thread_id] id: u32x3) {
        if id.x < root.count {
            let mut out = root.output;
            out[id.x] = id.x * 3 + 1;
        }
    }
}

pub mod gradient {
    use super::*;

    /// Every entry point in this module, which is what one library holds.
    pub const ENTRIES: &[&nga::ir::GpuFn] = &[&GRADIENT];

    /// A bindless storage texture, written through a handle the host put in the resource heap.
    #[gpu_struct]
    pub struct Data {
        pub target: Texture2dRW,
        pub size: u32x2,
    }

    #[gpu(compute(8, 8, 1))]
    fn gradient(root: Gpu<'static, Data>, #[thread_id] tid: u32x3) {
        if tid.x >= root.size.x || tid.y >= root.size.y {
            return;
        }
        let uv = f32x2::new(tid.x as f32, tid.y as f32)
            / f32x2::new(root.size.x as f32, root.size.y as f32);
        root.target
            .store(tid.xy(), f32x4::new(uv.x, uv.y, 0.0, 1.0));
    }
}

pub mod sample {
    use super::*;

    /// Every entry point in this module, which is what one library holds.
    pub const ENTRIES: &[&nga::ir::GpuFn] = &[&SAMPLE];

    /// Note what is absent: arithmetic on a handle. `base + layer` works on Vulkan, where a
    /// handle is an index, and means nothing on Metal, where it is an opaque `MTLResourceID`.
    /// An array of handles is the portable spelling.
    #[gpu_struct]
    pub struct Data {
        pub layers: [Texture2d; 3],
        pub samp: Sampler,
        pub target: Texture2dRW,
        pub size: u32x2,
    }

    #[gpu(compute(8, 8, 1))]
    fn sample(root: Gpu<'static, Data>, #[thread_id] tid: u32x3) {
        if tid.x >= root.size.x || tid.y >= root.size.y {
            return;
        }
        let uv = (f32x2::new(tid.x as f32, tid.y as f32) + 0.5)
            / f32x2::new(root.size.x as f32, root.size.y as f32);
        let layer = ((uv.x * 3.0) as u32).min(2);
        let c = root.layers[layer as usize].sample_level(root.samp, uv, 0.0);
        root.target.store(tid.xy(), c);
    }
}

pub mod material {
    use super::*;

    /// Every entry point in this module, which is what one library holds.
    pub const ENTRIES: &[&nga::ir::GpuFn] = &[&SHADE];

    /// Exercises the one place the two targets genuinely disagree: a struct holding bindless
    /// handles has to land on the same offsets whether a handle is a descriptor index or an
    /// `MTLResourceID`.
    #[gpu_struct]
    pub struct Material {
        pub albedo: Texture2d,
        pub samp: Sampler,
        pub target: Texture2dRW,
        pub uv: f32x2,
    }

    /// `sample_level`, not `sample`: a compute stage has no derivatives, so an implicit level
    /// of detail has nothing to compute from. The Slang original asked for one and neither the
    /// validator nor Metal would have accepted it — the old test only ever read the struct
    /// layout out of the module, never ran it.
    #[gpu(compute(8, 8, 1))]
    fn shade(root: Gpu<'static, Material>, #[thread_id] tid: u32x3) {
        root.target
            .store(tid.xy(), root.albedo.sample_level(root.samp, root.uv, 0.0));
    }
}

pub mod triangle {
    use super::*;

    /// Every entry point in this module, which is what one library holds.
    pub const ENTRIES: &[&nga::ir::GpuFn] = &[&VERTEX, &FRAGMENT];

    #[gpu_struct]
    pub struct Vertex {
        pub position: f32x3,
        pub color: f32x3,
    }

    #[gpu_struct(align(16))]
    pub struct Data {
        pub vertices: Gpu<'static, [Vertex]>,
        pub mvp: f32x4x4,
    }

    /// The interface between the two stages. `position` is the clip-space builtin on both
    /// sides; every other member is a varying, in declaration order.
    #[gpu_struct]
    pub struct VertexOut {
        pub position: f32x4,
        pub color: f32x3,
    }

    #[gpu(vertex)]
    fn vertex(root: Gpu<'static, Data>, #[vertex_id] vid: u32) -> VertexOut {
        let v = root.vertices[vid];
        VertexOut {
            position: root
                .mvp
                .mul(f32x4::new(v.position.x, v.position.y, v.position.z, 1.0)),
            color: v.color,
        }
    }

    #[gpu(fragment)]
    fn fragment(input: VertexOut) -> f32x4 {
        f32x4::new(input.color.x, input.color.y, input.color.z, 1.0)
    }
}

pub mod raster {
    use super::*;

    /// Every entry point in this module, which is what one library holds.
    pub const ENTRIES: &[&nga::ir::GpuFn] = &[&VERTEX, &FRAGMENT, &FRAGMENT_SCREEN];

    #[gpu_struct]
    pub struct Vertex {
        pub position: f32x3,
        pub color: f32x3,
    }

    #[gpu_struct(align(16))]
    pub struct Data {
        pub vertices: Gpu<'static, [Vertex]>,
        pub mvp: f32x4x4,
        pub tint: f32x4,
    }

    /// One specialization constant per field, numbered by field index — the same number
    /// `Constants::values` pushes, so the host and the module cannot disagree about which is
    /// which.
    #[gpu_struct]
    #[derive(Constants)]
    pub struct Options {
        pub boost: bool,
        pub gamma: f32,
    }

    #[gpu_struct]
    pub struct VertexOut {
        pub position: f32x4,
        pub color: f32x3,
        pub alpha: f32,
    }

    #[gpu(vertex)]
    fn vertex(root: Gpu<'static, Data>, #[vertex_id] vid: u32) -> VertexOut {
        let v = root.vertices[vid];
        VertexOut {
            position: root
                .mvp
                .mul(f32x4::new(v.position.x, v.position.y, v.position.z, 1.0)),
            color: v.color * root.tint.xyz(),
            alpha: root.tint.w,
        }
    }

    #[gpu(fragment)]
    fn fragment(input: VertexOut, #[constants] options: Options) -> f32x4 {
        let mut c = input.color.pow(f32x3::splat(options.gamma));
        if options.boost {
            c = (c * 1.6).min(f32x3::splat(1.0));
        }
        f32x4::new(c.x, c.y, c.z, input.alpha)
    }

    /// The Metal original read `float4 dst [[color(0)]]` — a framebuffer fetch — and computed
    /// screen blending itself. Desktop Vulkan has no framebuffer fetch, so this emits `src`
    /// only and the screen formula is a fixed blend equation instead: `ONE`,
    /// `ONE_MINUS_SRC_COLOR` is `src + dst * (1 - src)`, which is exactly screen.
    #[gpu(fragment)]
    fn fragment_screen(input: VertexOut) -> f32x4 {
        let c = input.color * input.alpha;
        f32x4::new(c.x, c.y, c.z, 1.0)
    }
}

pub mod sync {
    use super::*;

    /// Every entry point in this module, which is what one library holds.
    pub const ENTRIES: &[&nga::ir::GpuFn] = &[&SIMULATE, &BUILD_VERTICES, &VERTEX, &FRAGMENT];

    #[gpu_struct]
    pub struct Instance {
        pub center: f32x2,
        pub angle: f32,
        pub scale: f32,
        pub color: f32x4,
    }

    #[gpu_struct]
    pub struct Vertex {
        pub position: f32x3,
        pub color: f32x3,
    }

    #[gpu_struct]
    pub struct SimData {
        pub instances: GpuMut<'static, [Instance]>,
        pub time: f32,
        pub count: u32,
    }

    #[gpu_struct]
    pub struct BuildData {
        pub instances: Gpu<'static, [Instance]>,
        pub vertices: GpuMut<'static, [Vertex]>,
        pub count: u32,
        pub pad: u32,
    }

    #[gpu_struct(align(16))]
    pub struct DrawData {
        pub vertices: Gpu<'static, [Vertex]>,
        pub mvp: f32x4x4,
    }

    #[gpu(compute(64))]
    fn simulate(root: Gpu<'static, SimData>, #[thread_id] tid: u32x3) {
        let i = tid.x;
        if i >= root.count {
            return;
        }
        let f = i as f32 / root.count as f32;
        let orbit = 0.25 + 0.65 * (f * 7.0).fract();
        let phase = f * TAU + root.time * (0.2 + 0.9 * (f * 5.0).fract());
        let wheel = (f32x3::splat(TAU * f) + f32x3::new(0.0, 2.0, 4.0)).cos() * 0.5 + 0.5;

        let mut out = root.instances;
        out[i] = Instance {
            center: f32x2::new(phase.cos(), phase.sin()) * orbit,
            angle: phase * 2.0 + root.time,
            scale: 0.015 + 0.045 * (f * 3.0).fract(),
            color: f32x4::new(wheel.x, wheel.y, wheel.z, 1.0),
        };
    }

    #[gpu(compute(64))]
    fn build_vertices(root: Gpu<'static, BuildData>, #[thread_id] tid: u32x3) {
        let i = tid.x;
        if i >= root.count {
            return;
        }
        let inst = root.instances[i];
        let mut out = root.vertices;
        for k in 0..3u32 {
            let a = inst.angle + k as f32 * 2.0943951;
            let p = inst.center + f32x2::new(a.cos(), a.sin()) * inst.scale;
            out[i * 3 + k] = Vertex {
                position: f32x3::new(p.x, p.y, 0.5),
                color: inst.color.xyz(),
            };
        }
    }

    #[gpu_struct]
    pub struct VertexOut {
        pub position: f32x4,
        pub color: f32x3,
    }

    #[gpu(vertex)]
    fn vertex(root: Gpu<'static, DrawData>, #[vertex_id] vid: u32) -> VertexOut {
        let v = root.vertices[vid];
        VertexOut {
            position: root
                .mvp
                .mul(f32x4::new(v.position.x, v.position.y, v.position.z, 1.0)),
            color: v.color,
        }
    }

    #[gpu(fragment)]
    fn fragment(input: VertexOut) -> f32x4 {
        f32x4::new(input.color.x, input.color.y, input.color.z, 1.0)
    }
}

pub mod bindless {
    use super::*;

    /// Every entry point in this module, which is what one library holds.
    pub const ENTRIES: &[&nga::ir::GpuFn] = &[&PAINT, &VERTEX, &FRAGMENT];

    #[gpu_struct]
    pub struct Vertex {
        pub position: f32x2,
        pub uv: f32x2,
    }

    #[gpu_struct]
    pub struct Material {
        pub layers: [Texture2d; 3],
        pub samp: Sampler,
        pub tint: f32x3,
    }

    #[gpu_struct(align(16))]
    pub struct Data {
        pub vertices: Gpu<'static, [Vertex]>,
        pub materials: Gpu<'static, [Material]>,
        pub mvp: f32x4x4,
    }

    #[gpu_struct]
    pub struct Paint {
        pub target: Texture2dRW,
        pub size: u32,
        pub time: f32,
    }

    #[gpu(compute(8, 8, 1))]
    fn paint(root: Gpu<'static, Paint>, #[thread_id] tid: u32x3) {
        if tid.x >= root.size || tid.y >= root.size {
            return;
        }
        let p = (f32x2::new(tid.x as f32, tid.y as f32) + 0.5) / root.size as f32 * 2.0 - 1.0;
        let r = p.length();
        let wave = (r * 18.0 - root.time * 6.0).sin() * 0.5 + 0.5;
        let prev = root.target.load(tid.xy());
        root.target.store(
            tid.xy(),
            f32x4::new(wave, wave * (1.0 - r), prev.z * 0.9 + 0.1 * r, 1.0),
        );
    }

    /// `material` is an integer varying, so it is flat on both targets — that is not a choice
    /// either language leaves open.
    #[gpu_struct]
    pub struct VertexOut {
        pub position: f32x4,
        pub uv: f32x2,
        pub material: u32,
    }

    #[gpu(vertex)]
    fn vertex(root: Gpu<'static, Data>, #[vertex_id] vid: u32) -> VertexOut {
        let v = root.vertices[vid];
        VertexOut {
            position: root
                .mvp
                .mul(f32x4::new(v.position.x, v.position.y, 0.0, 1.0)),
            uv: v.uv,
            material: vid / 6,
        }
    }

    #[gpu(fragment)]
    fn fragment(root: Gpu<'static, Data>, input: VertexOut) -> f32x4 {
        let m = root.materials[input.material];
        let layer = ((input.uv.x * 3.0) as u32).min(2);
        let c = m.layers[layer as usize].sample(m.samp, input.uv).xyz() * m.tint;
        f32x4::new(c.x, c.y, c.z, 1.0)
    }
}

pub mod gpu_driven {
    use super::*;

    /// Every entry point in this module, which is what one library holds.
    pub const ENTRIES: &[&nga::ir::GpuFn] = &[
        &SIMULATE,
        &ENCODE,
        &VERTEX,
        &FRAGMENT,
        &BOUNDARY,
        &BOUNDARY_FRAGMENT,
    ];

    #[gpu_struct]
    pub struct Instance {
        pub center: f32x2,
        pub angle: f32,
        pub scale: f32,
        pub color: f32x4,
    }

    /// `VkDrawIndirectCommand`, laid out by the spec rather than by us.
    #[gpu_struct]
    pub struct DrawArgs {
        pub vertex_count: u32,
        pub instance_count: u32,
        pub first_vertex: u32,
        pub first_instance: u32,
    }

    #[gpu_struct]
    pub struct SimData {
        pub instances: GpuMut<'static, [Instance]>,
        pub time: f32,
        pub count: u32,
    }

    #[gpu_struct]
    pub struct EncodeData {
        pub instances: Gpu<'static, [Instance]>,
        pub draws: GpuMut<'static, [DrawArgs]>,
        pub count: GpuMut<'static, AtomicU32>,
        pub cull_radius: f32,
        pub total: u32,
    }

    #[gpu_struct(align(16))]
    pub struct DrawData {
        pub instances: Gpu<'static, [Instance]>,
        pub mvp: f32x4x4,
    }

    #[gpu_struct(align(16))]
    pub struct BoundaryData {
        pub mvp: f32x4x4,
        pub radius: f32,
        pub segments: u32,
    }

    #[gpu(compute(64))]
    fn simulate(root: Gpu<'static, SimData>, #[thread_id] tid: u32x3) {
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
    #[gpu(compute(64))]
    fn encode(root: Gpu<'static, EncodeData>, #[thread_id] tid: u32x3) {
        let i = tid.x;
        if i >= root.total {
            return;
        }
        let inst = root.instances[i];
        if inst.center.length() > root.cull_radius {
            return;
        }
        let mut counter = root.count;
        let slot = counter.atomic_add(1);
        let mut draws = root.draws;
        draws[slot] = DrawArgs {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: i,
        };
    }

    #[gpu_struct]
    pub struct VertexOut {
        pub position: f32x4,
        pub color: f32x4,
    }

    #[gpu(vertex)]
    fn vertex(
        root: Gpu<'static, DrawData>,
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

    #[gpu(fragment)]
    fn fragment(input: VertexOut) -> f32x4 {
        input.color
    }

    /// A vertex stage that returns a bare `f32x4` is returning clip space and nothing else.
    #[gpu(vertex)]
    fn boundary(root: Gpu<'static, BoundaryData>, #[vertex_id] vid: u32) -> f32x4 {
        let a = (vid / 2) as f32 / root.segments as f32 * TAU;
        let mut r = root.radius + 0.004;
        if vid % 2 == 0 {
            r = root.radius - 0.004;
        }
        root.mvp
            .mul(f32x4::new(a.cos() * r, a.sin() * r, 0.25, 1.0))
    }

    #[gpu(fragment)]
    fn boundary_fragment() -> f32x4 {
        f32x4::new(1.0, 1.0, 1.0, 0.6)
    }
}
