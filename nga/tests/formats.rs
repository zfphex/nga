use nga::*;

#[test]
fn block_layout() {
    for format in [Format::Bc1RgbaUnorm, Format::Bc4RUnorm, Format::Bc4RSnorm] {
        assert_eq!(format.block_extent(), (4, 4));
        assert_eq!(format.block_size(), 8);
        assert_eq!(format.row_size(5), 16);
        assert_eq!(format.image_size(5, 7), 32);
        assert_eq!(format.image_size(1, 1), 8);
        assert_eq!(format.image_size(0, 7), 0);
    }
    for format in [Format::Bc7RgbaUnorm, Format::Astc4x4UnormSrgb] {
        assert_eq!(format.block_extent(), (4, 4));
        assert_eq!(format.block_size(), 16);
        assert_eq!(format.row_size(5), 32);
        assert_eq!(format.image_size(5, 7), 64);
        assert_eq!(format.image_size(1, 1), 16);
        assert_eq!(format.image_size(7, 0), 0);
    }
    assert_eq!(Format::Rgba8Unorm.block_extent(), (1, 1));
    assert_eq!(Format::Rgba8Unorm.size(), 4);
    assert_eq!(Format::Rgba8Unorm.image_size(5, 7), 140);
    assert_eq!(Format::Rgba16Float.image_size(5, 7), 280);
}

#[test]
#[should_panic]
fn compressed_texel_size_is_not_defined() {
    Format::Astc4x4Unorm.size();
}

#[test]
#[ignore = "requires a GPU"]
fn compressed_upload_readback() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(true)?;
    let shaders = device.shaders(&[])?;
    let heap = GpuHeap::with_ring(&device, 8 * 1024 * 1024, 96)?;
    let readback = Arena::new(&device, 4096, Memory::Readback)?;
    let pixels = readback.alloc_slice(&[0u8; 4096]).unwrap();
    let mut context = Context::new(&device, &shaders, 4096, 1)?;
    for format in [
        Format::Bc1RgbaUnorm,
        Format::Bc1RgbaUnormSrgb,
        Format::Bc2RgbaUnorm,
        Format::Bc2RgbaUnormSrgb,
        Format::Bc3RgbaUnorm,
        Format::Bc3RgbaUnormSrgb,
        Format::Bc4RUnorm,
        Format::Bc4RSnorm,
        Format::Bc5RgUnorm,
        Format::Bc5RgSnorm,
        Format::Bc6hRgbFloat,
        Format::Bc6hRgbUfloat,
        Format::Bc7RgbaUnorm,
        Format::Bc7RgbaUnormSrgb,
        Format::Astc4x4Unorm,
        Format::Astc4x4UnormSrgb,
    ] {
        let feature = match format {
            Format::Astc4x4Unorm | Format::Astc4x4UnormSrgb => Feature::TextureCompressionAstc,
            _ => Feature::TextureCompressionBc,
        };
        let desc = TextureDesc::texture_2d(format, 17, 13)
            .mips(5)
            .layers(2)
            .usage(Usage::COPY_SRC);
        if !device.supports(feature) {
            assert!(heap.alloc_texture(&desc).is_err());
            continue;
        }
        assert!(heap.alloc_texture(&desc.usage(Usage::COLOR)).is_err());
        assert!(heap.alloc_texture(&desc.usage(Usage::STORAGE)).is_err());
        assert!(heap.alloc_texture(&desc.samples(4)).is_err());
        let texture = heap.alloc_texture(&desc)?;
        assert_eq!(texture.format(), format);
        assert!(heap.upload(&texture, 0, &[]).is_err());
        assert!(heap.upload(&texture, 0, &[0; 17 * 13 * 4]).is_err());
        assert!(heap.upload_layer(&texture, 5, 0, &[0; 16]).is_err());
        assert!(heap.upload_layer(&texture, 0, 2, &[0; 16]).is_err());
        for layer in 0..2 {
            for level in 0..5 {
                let size = format.image_size((17 >> level).max(1), (13 >> level).max(1));
                let mut bytes = vec![0u8; size];
                for (index, byte) in bytes.iter_mut().enumerate() {
                    *byte = (index as u8).wrapping_add((layer * 71 + level * 13) as u8);
                }
                heap.upload_layer(&texture, level, layer, &bytes)?;
                heap.flush()?;
                let mut frame = context.begin_frame()?;
                frame.recorder().compute().read_texture_at(
                    &texture,
                    level,
                    layer,
                    &readback,
                    pixels.gpu.as_rw(),
                );
                frame.submit(None)?;
                device.wait_idle()?;
                assert_eq!(
                    &pixels[..size],
                    &bytes,
                    "{format:?}, mip {level}, layer {layer}"
                );
            }
        }
        heap.free_texture(texture);
    }
    assert_eq!(device.validation_errors(), 0);
    Ok(())
}
