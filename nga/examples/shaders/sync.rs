//! The shader `sync.rs` and `threaded.rs` both draw with, in Rust.
//!
//! It lives in its own file because two examples run the same simulation: `sync` on one thread
//! and `threaded` on several. A directory under `examples/` with no `main.rs` in it is not an
//! example, so cargo leaves this alone and each of them pulls it in with `#[path]`.

use nga::gpu_mod;

#[gpu_mod]
pub mod shader {
    use nga::*;

    const TAU: f32 = 6.2831853;

    pub struct Instance {
        pub center: f32x2,
        pub angle: f32,
        pub scale: f32,
        pub color: f32x4,
    }

    pub struct Vertex {
        pub position: f32x3,
        pub color: f32x3,
    }

    pub struct SimData {
        pub instances: GpuMut<'static, [Instance]>,
        pub time: f32,
        pub count: u32,
    }

    pub struct BuildData {
        pub instances: Gpu<'static, [Instance]>,
        pub vertices: GpuMut<'static, [Vertex]>,
        pub count: u32,
        pub pad: u32,
    }

    #[repr(C, align(16))]
    pub struct DrawData {
        pub vertices: Gpu<'static, [Vertex]>,
        pub mvp: f32x4x4,
    }

    #[compute(64)]
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

    #[compute(64)]
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

    pub struct VertexOut {
        pub position: f32x4,
        pub color: f32x3,
    }

    #[vertex]
    fn vertex(root: Gpu<'static, DrawData>, #[vertex_id] vid: u32) -> VertexOut {
        let v = root.vertices[vid];
        VertexOut {
            position: root
                .mvp
                .mul(f32x4::new(v.position.x, v.position.y, v.position.z, 1.0)),
            color: v.color,
        }
    }

    #[fragment]
    fn fragment(input: VertexOut) -> f32x4 {
        f32x4::new(input.color.x, input.color.y, input.color.z, 1.0)
    }
}

pub use shader::*;
