use ash::vk;
use ash::vk::TaggedStructure as _;
use nga_vk::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(false)?;
    eprintln!("device: {}", device.name);

    let combos: [(&str, vk::ImageUsageFlags); 4] = [
        (
            "SAMPLED|TRANSFER_DST",
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        ),
        (
            "SAMPLED|TRANSFER_DST|STORAGE",
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE,
        ),
        (
            "SAMPLED|TRANSFER_DST|HOST_TRANSFER",
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
        ),
        (
            "SAMPLED|HOST_TRANSFER|STORAGE",
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::HOST_TRANSFER_EXT
                | vk::ImageUsageFlags::STORAGE,
        ),
    ];

    for (name, usage) in combos {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: 64,
                height: 64,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let mut from_info = vk::MemoryRequirements2::default();
        unsafe {
            device.device.get_device_image_memory_requirements(
                &vk::DeviceImageMemoryRequirements::default().create_info(&info),
                &mut from_info,
            )
        };

        let image = unsafe { device.device.create_image(&info, None) };
        let from_object = match image {
            Ok(image) => {
                let requirements = unsafe { device.device.get_image_memory_requirements(image) };
                unsafe { device.device.destroy_image(image, None) };
                Some(requirements)
            }
            Err(e) => {
                eprintln!("{name:38} create_image failed: {e:?}");
                None
            }
        };

        eprintln!(
            "{name:38} maintenance4 bits {:#06x} size {:6}   object bits {:#06x} size {:6}",
            from_info.memory_requirements.memory_type_bits,
            from_info.memory_requirements.size,
            from_object.map_or(0, |r| r.memory_type_bits),
            from_object.map_or(0, |r| r.size),
        );
    }

    eprintln!();
    eprintln!("--- queue families ---");
    let families = unsafe {
        device
            .instance
            .get_physical_device_queue_family_properties(device.physical)
    };
    for (index, family) in families.iter().enumerate() {
        let mut flags = Vec::new();
        for (bit, name) in [
            (vk::QueueFlags::GRAPHICS, "graphics"),
            (vk::QueueFlags::COMPUTE, "compute"),
            (vk::QueueFlags::TRANSFER, "transfer"),
            (vk::QueueFlags::SPARSE_BINDING, "sparse"),
            (vk::QueueFlags::VIDEO_DECODE_KHR, "video-decode"),
            (vk::QueueFlags::VIDEO_ENCODE_KHR, "video-encode"),
        ] {
            if family.queue_flags.contains(bit) {
                flags.push(name);
            }
        }
        // A family with TRANSFER and no GRAPHICS is a dedicated copy engine: it can DMA while
        // the graphics queue renders. `nga-vk` does not use one yet — see NOTES.md.
        let dedicated = family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && !family.queue_flags.intersects(
                vk::QueueFlags::GRAPHICS
                    | vk::QueueFlags::COMPUTE
                    | vk::QueueFlags::VIDEO_DECODE_KHR
                    | vk::QueueFlags::VIDEO_ENCODE_KHR,
            );
        eprintln!(
            "family {index}: {:2} queues  {:34} {}",
            family.queue_count,
            flags.join("|"),
            if dedicated {
                "<- dedicated copy engine"
            } else {
                ""
            },
        );
    }

    eprintln!();
    eprintln!("--- memory heaps ---");
    for heap in 0..device.memory.memory_heap_count {
        let properties = device.memory.memory_heaps[heap as usize];
        let local = properties.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
        eprintln!(
            "heap {heap}: {:>9.2} MiB total, {:>9.2} MiB free  {}",
            properties.size as f64 / (1 << 20) as f64,
            device.budget(heap) as f64 / (1 << 20) as f64,
            if local { "DEVICE_LOCAL" } else { "" },
        );
    }

    eprintln!();
    eprintln!("--- placements ---");
    // What `Placement` actually resolves to on this machine. The BAR line is the one that
    // decides how much per-frame arena the library can afford: without ReBAR it is ~214 MiB
    // shared with every other process, and the free figure is what is left right now.
    let mask = u32::MAX;
    for (name, flags, without) in [
        (
            "Bar      (host-visible VRAM)",
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        ),
        (
            "Upload   (write-combined RAM)",
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_CACHED,
        ),
        (
            "Readback (cached RAM)",
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED,
            vk::MemoryPropertyFlags::empty(),
        ),
    ] {
        match device.memory_types(mask, flags, without).first() {
            Some(&index) => {
                let heap = device.heap_of(index);
                eprintln!(
                    "{name:30} -> type {index:2} heap {heap}, {:>9.2} MiB free",
                    device.budget(heap) as f64 / (1 << 20) as f64,
                );
            }
            None => eprintln!("{name:30} -> unavailable"),
        }
    }

    eprintln!();
    let mut host = vk::FormatProperties3::default();
    let mut props = vk::FormatProperties2::default().push(&mut host);
    unsafe {
        device.instance.get_physical_device_format_properties2(
            device.physical,
            vk::Format::R8G8B8A8_UNORM,
            &mut props,
        )
    };
    eprintln!(
        "R8G8B8A8_UNORM optimal supports HOST_IMAGE_TRANSFER: {}",
        host.optimal_tiling_features
            .contains(vk::FormatFeatureFlags2::HOST_IMAGE_TRANSFER_EXT)
    );
    Ok(())
}
