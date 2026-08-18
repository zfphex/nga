use ash::vk;
use nga_vk::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(false)?;
    eprintln!("device: {}", device.name);

    let combos: [(&str, vk::ImageUsageFlags); 4] = [
        ("SAMPLED|TRANSFER_DST", vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
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

    let mut host = vk::FormatProperties3::default();
    let mut props = vk::FormatProperties2::default().push_next(&mut host);
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
