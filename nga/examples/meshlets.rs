fn main() {
    #[cfg(target_vendor = "apple")]
    if let Err(error) = meshlets::run() {
        eprintln!("{error}");
    }
    #[cfg(not(target_vendor = "apple"))]
    eprintln!("meshlets needs mesh pipelines, which only the Metal backend has today");
}

#[cfg(target_vendor = "apple")]
mod meshlets {
    use minwin::*;
    use nga::*;

    #[gpu_mod]
    mod shader {
        use nga::*;

        const PER_GROUP: u32 = 32;
        const RING: u32 = 8;

        pub struct Meshlet {
            pub bounds: f32x4,
            pub color: [f32; 3],
        }

        #[repr(C, align(16))]
        pub struct Scene {
            pub meshlets: Gpu<[Meshlet]>,
            pub mvp: f32x4x4,
            pub cull_radius: f32,
            pub time: f32,
            pub meshlet_count: u32,
        }

        #[repr(C, align(16))]
        pub struct Payload {
            pub meshlets: [u32; 32],
            pub mvp: f32x4x4,
            pub time: f32,
        }

        pub struct VertexOut {
            pub position: f32x4,
            pub world: f32x3,
            pub color: f32x4,
        }

        #[object]
        fn meshlet_object(
            root: Gpu<Scene>,
            #[payload] mut payload: Payload,
            #[group_id] group: u32x3,
            #[thread_index_in_group] lane: u32,
        ) {
            let mut visible = ThreadgroupAtomicU32::new(0);
            if lane == 0 {
                visible.atomic_exchange(0);
            }
            threadgroup_barrier();

            let index = group.x * PER_GROUP + lane;
            if index < root.meshlet_count {
                let bounds = root.meshlets[index].bounds;
                if bounds.xyz().length() - bounds.w <= root.cull_radius {
                    let slot = visible.atomic_add(1);
                    payload.meshlets[slot as usize] = index;
                }
            }
            threadgroup_barrier();

            if lane == 0 {
                payload.mvp = root.mvp;
                payload.time = root.time;
                mesh_set_threadgroups(u32x3::new(visible.atomic_add(0), 1, 1));
            }
        }

        #[mesh(VertexOut, 9, 8)]
        fn meshlet_mesh(
            root: Gpu<Scene>,
            #[payload] payload: Payload,
            #[group_id] group: u32x3,
            #[thread_index_in_group] lane: u32,
        ) {
            let index = payload.meshlets[group.x as usize];
            let meshlet = root.meshlets[index];
            let centre = meshlet.bounds.xyz();
            let radius = meshlet.bounds.w;
            if lane == 0 {
                mesh_set_primitive_count(RING);
            }
            if lane <= RING {
                let mut position = centre;
                if lane > 0 {
                    let angle = (lane - 1) as f32 / RING as f32 * 6.2831853 + payload.time * 0.5;
                    position += f32x3::new(angle.cos(), angle.sin(), 0.0) * radius;
                }
                let hash = index * 2654435761u32;
                let tint = f32x3::new(
                    ((hash >> 16) & 255) as f32,
                    ((hash >> 8) & 255) as f32,
                    (hash & 255) as f32,
                ) / 255.0;
                let base = f32x3::new(meshlet.color[0], meshlet.color[1], meshlet.color[2]);
                let color = base.mix(tint, 0.35);
                mesh_set_vertex(
                    lane,
                    VertexOut {
                        position: payload
                            .mvp
                            .mul(f32x4::new(position.x, position.y, position.z, 1.0)),
                        world: position,
                        color: f32x4::new(color.x, color.y, color.z, 1.0),
                    },
                );
            }
            if lane < RING {
                mesh_set_index(lane * 3, 0);
                mesh_set_index(lane * 3 + 1, 1 + lane);
                mesh_set_index(lane * 3 + 2, 1 + (lane + 1) % RING);
            }
        }

        #[fragment]
        fn meshlet_fragment(input: VertexOut) -> f32x4 {
            let shade = 0.55 + 0.45 * (input.world.length() * 6.0).cos();
            f32x4::new(
                input.color.x * shade,
                input.color.y * shade,
                input.color.z * shade,
                1.0,
            )
        }
    }

    use shader::*;

    const PER_GROUP: u32 = 32;
    const MESHLETS: u32 = 24 * PER_GROUP;

    const PAYLOAD: usize = 32 * 4 + 64 + 16;

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut window = create_window(
            "nga M6 — meshlets",
            None,
            900,
            900,
            true,
            WindowStyle::Standard,
        );

        let device = Device::new(cfg!(debug_assertions))?;
        eprintln!("device: {}", device.name());
        if !device.supports(Feature::MeshShaders) {
            return Err(Error::new("this device has no mesh pipeline").into());
        }

        let (mut width, mut height) = window.scaled_size();
        let mut surface = device.surface(window.handle(), width as u32, height as u32)?;

        let mut shaders = device.shaders(shader::MODULE)?;
        let pipeline = shaders.mesh(&MeshDesc::new(
            "meshlet_object",
            "meshlet_mesh",
            "meshlet_fragment",
            &[ColorTarget::new(surface.format())],
            PAYLOAD,
        ))?;

        let mut list = Vec::with_capacity(MESHLETS as usize);
        for index in 0..MESHLETS {
            let t = index as f32 / MESHLETS as f32;
            let angle = t * 6.2831853 * 6.0;
            let radius = 0.08 + 0.85 * t;
            let hue = t * 6.2831853;
            list.push(Meshlet {
                bounds: f32x4([
                    angle.cos() * radius,
                    angle.sin() * radius,
                    0.0,
                    0.012 + 0.05 * t,
                ]),
                color: [
                    0.5 + 0.5 * hue.cos(),
                    0.5 + 0.5 * (hue + 2.0).cos(),
                    0.5 + 0.5 * (hue + 4.0).cos(),
                ],
            });
        }

        let storage = Arena::new(&device, 1 << 20, Memory::Device)?;
        let meshlets = storage.alloc_slice(&list).unwrap();
        let meshlets: Gpu<[Meshlet]> = meshlets.gpu;

        let mut ctx = Context::new(&device, &shaders, 64 * 1024, 1)?;

        let mut timers = Timers::new(&device, 2)?;

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

                let mut frame = ctx.begin_frame().unwrap();
                timers.begin(&frame);
                let rec = frame.recorder();

                let pass = Pass::new().color(0, &drawable, Some([0.02, 0.02, 0.04, 1.0]));
                let raster = rec.raster(&pass);
                raster.timestamp(&timers, 0);

                raster
                    .draw_mesh(
                        pipeline,
                        Scene {
                            meshlets,
                            mvp,

                            cull_radius: 0.55 + 0.45 * (time * 0.5).sin(),
                            time,
                            meshlet_count: MESHLETS,
                        },
                        [MESHLETS.div_ceil(PER_GROUP), 1, 1],
                        [PER_GROUP, 1, 1],
                        [16, 1, 1],
                    )
                    .unwrap();
                raster.timestamp(&timers, 1);

                frame.submit(Some(&drawable)).unwrap();
                surface.present(&device, &drawable);

                if timers.written(0) && timers.written(1) {
                    win.set_title(&format!("nga M6 — meshlets — {:.3} ms", timers.ms(0, 1)));
                }
                time += 1.0 / 60.0;
            });
        }

        device.wait_idle()?;
        Ok(())
    }
}
