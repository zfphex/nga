//! The Vulkan 1.4 backend.
//!
//! ## Backend contract
//!
//! `lib.rs` ends with `pub use backend::*`, so this file and `metal.rs` must define the same
//! names with the same signatures. Neither is compiled on the other's platform, so the
//! compiler cannot check that — this list is the checklist, and it is kept in the same order
//! in both files:
//!
//! `Device`, `Arena`, `GpuHeap`, `HeapBuffer`, `HeapTexture`, `TextureDesc`, `ResourceHeap`,
//! `HeapRange`, `Shaders`, `Pipeline`, `Surface`, `Drawable`, `Pass`, `Raster`, `Compute`,
//! `Context`, `Frame`, `Recorder`, `Fence`.
//!
//! Nothing from `ash` appears in a public signature. Every handle that crosses the boundary is
//! a newtype declared here with private fields, so `pub use backend::*` cannot leak one.

use crate::{
    AddressMode, Alloc, BlendFactor, BlendMode, ColorTarget, CompareOp, Constant, Constants, Cull,
    DepthState, Error, FRAMES_IN_FLIGHT, Feature, Filter, Format, Gpu, GpuBox, GpuMut, Handle,
    Hazard, IndexFormat, Memory, RasterDesc, Sampler, SamplerDesc, Scalar, Scissor, Stage,
    Texture2d, Texture2dRW, Tlsf, Topology, Usage, Viewport, Winding,
};
use ash::vk;
use ash::vk::TaggedStructure as _;
use std::cell::Cell;
use std::ffi::{CStr, c_char};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A compiled pipeline. Opaque so that `VkPipeline` never reaches a caller, and `Copy` so it
/// can be passed around as freely as the `MTLRenderPipelineState` the Metal side hands out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pipeline(vk::Pipeline);

/// A sampler *object* living in the heap. `Sampler` in `lib.rs` is the bindless handle that
/// points at one of these, which is why this needs a different name.
pub struct SamplerState(vk::Sampler, ash::Device);

impl Drop for SamplerState {
    fn drop(&mut self) {
        unsafe { self.1.destroy_sampler(self.0, None) };
    }
}

/// One run of a multi-draw. `VK_EXT_multi_draw` collapses a run of draws into a single call.
/// Field order and layout match `VkMultiDrawInfoEXT`, so a slice passes straight through.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MultiDraw {
    pub first_vertex: u32,
    pub vertex_count: u32,
}

/// Matches `VkMultiDrawIndexedInfoEXT`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MultiDrawIndexed {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
}

impl From<vk::Result> for Error {
    fn from(result: vk::Result) -> Self {
        Error::new(format!("{result:?}"))
    }
}

// -- enum translation --------------------------------------------------------------------
//
// One `match` per public enum. Adding a variant in `lib.rs` breaks these, which is the point:
// the compiler points at every backend that has not caught up.

pub(crate) fn format(format: Format) -> vk::Format {
    match format {
        Format::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        Format::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        Format::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        Format::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        Format::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        Format::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        Format::Rg11B10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
        Format::Rgb10A2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        Format::Rg8Unorm => vk::Format::R8G8_UNORM,
        Format::Rg16Float => vk::Format::R16G16_SFLOAT,
        Format::R8Unorm => vk::Format::R8_UNORM,
        Format::R16Float => vk::Format::R16_SFLOAT,
        Format::R32Float => vk::Format::R32_SFLOAT,
        Format::R32Uint => vk::Format::R32_UINT,
        Format::Rg32Uint => vk::Format::R32G32_UINT,
        Format::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
        Format::D32Float => vk::Format::D32_SFLOAT,
        Format::D32FloatS8 => vk::Format::D32_SFLOAT_S8_UINT,
        Format::D16Unorm => vk::Format::D16_UNORM,
    }
}

pub(crate) fn from_vk_format(raw: vk::Format) -> Option<Format> {
    Some(match raw {
        vk::Format::R8G8B8A8_UNORM => Format::Rgba8Unorm,
        vk::Format::R8G8B8A8_SRGB => Format::Rgba8UnormSrgb,
        vk::Format::B8G8R8A8_UNORM => Format::Bgra8Unorm,
        vk::Format::B8G8R8A8_SRGB => Format::Bgra8UnormSrgb,
        vk::Format::R16G16B16A16_SFLOAT => Format::Rgba16Float,
        vk::Format::R32G32B32A32_SFLOAT => Format::Rgba32Float,
        vk::Format::B10G11R11_UFLOAT_PACK32 => Format::Rg11B10Float,
        vk::Format::A2B10G10R10_UNORM_PACK32 => Format::Rgb10A2Unorm,
        vk::Format::R8G8_UNORM => Format::Rg8Unorm,
        vk::Format::R16G16_SFLOAT => Format::Rg16Float,
        vk::Format::R8_UNORM => Format::R8Unorm,
        vk::Format::R16_SFLOAT => Format::R16Float,
        vk::Format::R32_SFLOAT => Format::R32Float,
        vk::Format::R32_UINT => Format::R32Uint,
        vk::Format::R32G32_UINT => Format::Rg32Uint,
        vk::Format::R32G32B32A32_UINT => Format::Rgba32Uint,
        vk::Format::D32_SFLOAT => Format::D32Float,
        vk::Format::D32_SFLOAT_S8_UINT => Format::D32FloatS8,
        vk::Format::D16_UNORM => Format::D16Unorm,
        _ => return None,
    })
}

pub(crate) fn compare_op(op: CompareOp) -> vk::CompareOp {
    match op {
        CompareOp::Never => vk::CompareOp::NEVER,
        CompareOp::Less => vk::CompareOp::LESS,
        CompareOp::Equal => vk::CompareOp::EQUAL,
        CompareOp::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareOp::Greater => vk::CompareOp::GREATER,
        CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareOp::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareOp::Always => vk::CompareOp::ALWAYS,
    }
}

pub(crate) fn topology(topology: Topology) -> vk::PrimitiveTopology {
    match topology {
        Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        Topology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        Topology::LineList => vk::PrimitiveTopology::LINE_LIST,
        Topology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        Topology::PointList => vk::PrimitiveTopology::POINT_LIST,
    }
}

pub(crate) fn cull(cull: Cull) -> vk::CullModeFlags {
    match cull {
        Cull::None => vk::CullModeFlags::NONE,
        Cull::Front => vk::CullModeFlags::FRONT,
        Cull::Back => vk::CullModeFlags::BACK,
    }
}

pub(crate) fn winding(winding: Winding) -> vk::FrontFace {
    match winding {
        Winding::Clockwise => vk::FrontFace::CLOCKWISE,
        Winding::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
    }
}

pub(crate) fn blend_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
    }
}

pub(crate) fn filter(filter: Filter) -> vk::Filter {
    match filter {
        Filter::Nearest => vk::Filter::NEAREST,
        Filter::Linear => vk::Filter::LINEAR,
    }
}

pub(crate) fn mip_mode(filter: Filter) -> vk::SamplerMipmapMode {
    match filter {
        Filter::Nearest => vk::SamplerMipmapMode::NEAREST,
        Filter::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

pub(crate) fn address_mode(mode: AddressMode) -> vk::SamplerAddressMode {
    match mode {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

/// The one place Vulkan's per-image layout state is decided. Metal has no equivalent, which
/// is why this is a backend detail and `Usage` is what the caller names.
pub(crate) fn image_layout(usage: Usage) -> vk::ImageLayout {
    if usage.has(Usage::STORAGE) {
        vk::ImageLayout::GENERAL
    } else if usage.has(Usage::COLOR) {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    } else if usage.has(Usage::DEPTH_STENCIL) {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    } else if usage.has(Usage::COPY_SRC) {
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
    } else if usage.has(Usage::COPY_DST) {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL
    } else {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    }
}

pub(crate) fn image_barrier(
    texture: &HeapTexture,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(old)
        .new_layout(new)
        .image(texture.image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        )
}

pub(crate) fn usage(usage: Usage) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    if usage.has(Usage::SAMPLED) {
        flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.has(Usage::STORAGE) {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    if usage.has(Usage::COLOR) {
        flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if usage.has(Usage::DEPTH_STENCIL) {
        flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    if usage.has(Usage::COPY_SRC) {
        flags |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.has(Usage::COPY_DST) {
        flags |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    flags
}

/// The blog's `STAGE` values, widened to `VkPipelineStageFlags2`. Metal takes the same mask
/// almost directly; Vulkan needs the two-sided access masks that `hazard` supplies.
pub(crate) fn stage(stage: Stage) -> vk::PipelineStageFlags2 {
    let mut flags = vk::PipelineStageFlags2::NONE;
    if stage.has(Stage::TRANSFER) {
        flags |= vk::PipelineStageFlags2::ALL_TRANSFER;
    }
    if stage.has(Stage::COMPUTE) {
        flags |= vk::PipelineStageFlags2::COMPUTE_SHADER;
    }
    if stage.has(Stage::VERTEX) {
        flags |= vk::PipelineStageFlags2::VERTEX_SHADER;
    }
    if stage.has(Stage::PIXEL) {
        flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
    }
    if stage.has(Stage::COLOR_OUT) {
        flags |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
    }
    if stage.has(Stage::DEPTH_STENCIL) {
        flags |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
    }
    if stage.has(Stage::INDIRECT) {
        flags |= vk::PipelineStageFlags2::DRAW_INDIRECT;
    }
    flags
}

/// Metal's visibility flag is stage-agnostic: one bit means the same thing whatever stages the
/// barrier joins. Vulkan access masks are not — pairing `SHADER_WRITE` with `ALL_TRANSFER`, or
/// `SHADER_READ` with `DRAW_INDIRECT`, is invalid. `MEMORY_READ`/`MEMORY_WRITE` are the only
/// masks legal with every stage, so they are what keeps `barrier` portable.
pub(crate) fn hazard(hazard: Hazard) -> (vk::AccessFlags2, vk::AccessFlags2) {
    if hazard == Hazard::NONE {
        return (vk::AccessFlags2::NONE, vk::AccessFlags2::NONE);
    }
    (
        vk::AccessFlags2::MEMORY_WRITE,
        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
    )
}

/// SPIR-V specialization payloads are a flat blob plus an entry table, and a `bool` occupies
/// a full `VkBool32` rather than the one byte Rust gives it.
pub(crate) fn specialize(values: &[Constant]) -> (Vec<vk::SpecializationMapEntry>, Vec<u8>) {
    let mut entries = Vec::with_capacity(values.len());
    let mut data = Vec::new();
    for value in values {
        let width = match value.scalar {
            Scalar::Bool => 4,
            _ => value.len as usize,
        };
        entries.push(
            vk::SpecializationMapEntry::default()
                .constant_id(value.index)
                .offset(data.len() as u32)
                .size(width),
        );
        data.extend_from_slice(&value.bytes[..value.len as usize]);
        data.resize(data.len() + width - value.len as usize, 0);
    }
    debug_assert_eq!(
        data.len() % 4,
        0,
        "specialization payload must stay word-aligned"
    );
    (entries, data)
}

pub(crate) fn index_type<I: IndexFormat>() -> vk::IndexType {
    match I::WIDE {
        true => vk::IndexType::UINT32,
        false => vk::IndexType::UINT16,
    }
}

/// Negative height flips clip space so the two backends agree on which way is up without the
/// shader or the caller knowing which one is running.
pub(crate) fn viewport(v: Viewport) -> vk::Viewport {
    vk::Viewport {
        x: v.x,
        y: v.y + v.height,
        width: v.width,
        height: -v.height,
        min_depth: v.min_depth,
        max_depth: v.max_depth,
    }
}

pub(crate) fn scissor(s: Scissor) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D {
            x: s.x as i32,
            y: s.y as i32,
        },
        extent: vk::Extent2D {
            width: s.width,
            height: s.height,
        },
    }
}

pub(crate) fn blend_attachment(target: &ColorTarget) -> vk::PipelineColorBlendAttachmentState {
    let mask = vk::ColorComponentFlags::from_raw(target.write_mask as u32);
    let state = vk::PipelineColorBlendAttachmentState::default().color_write_mask(mask);
    let (src_color, dst_color, src_alpha, dst_alpha) = match target.blend {
        BlendMode::Opaque => return state,
        BlendMode::Alpha => (
            BlendFactor::SrcAlpha,
            BlendFactor::OneMinusSrcAlpha,
            BlendFactor::One,
            BlendFactor::OneMinusSrcAlpha,
        ),
        BlendMode::Premultiplied => (
            BlendFactor::One,
            BlendFactor::OneMinusSrcAlpha,
            BlendFactor::One,
            BlendFactor::OneMinusSrcAlpha,
        ),
        BlendMode::Additive => (
            BlendFactor::One,
            BlendFactor::One,
            BlendFactor::One,
            BlendFactor::One,
        ),
        BlendMode::Screen => (
            BlendFactor::One,
            BlendFactor::OneMinusSrcColor,
            BlendFactor::One,
            BlendFactor::OneMinusSrcAlpha,
        ),
    };
    state
        .blend_enable(true)
        .color_blend_op(vk::BlendOp::ADD)
        .alpha_blend_op(vk::BlendOp::ADD)
        .src_color_blend_factor(blend_factor(src_color))
        .dst_color_blend_factor(blend_factor(dst_color))
        .src_alpha_blend_factor(blend_factor(src_alpha))
        .dst_alpha_blend_factor(blend_factor(dst_alpha))
}

/// Metal binds two argument-table addresses; Vulkan gets one push-constant range instead.
/// Every pipeline layout in `nga-vk` is this range plus the two descriptor-buffer sets, so
/// pipelines are freely interchangeable the way they are under one `MTL4ArgumentTable`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Push {
    pub(crate) root: u64,
    pub(crate) resources: u32,
    pub(crate) samplers: u32,
}

pub(crate) const PUSH_SIZE: u32 = size_of::<Push>() as u32;

pub struct Device {
    /// Never read. Held because dropping it unloads `vulkan-1.dll`.
    #[allow(dead_code)]
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) queue_family: u32,
    pub(crate) memory: vk::PhysicalDeviceMemoryProperties,
    pub(crate) limits: vk::PhysicalDeviceLimits,
    pub(crate) name: String,
    pub(crate) surface_ext: ash::khr::surface::Instance,
    pub(crate) win32_surface_ext: ash::khr::win32_surface::Instance,
    pub(crate) swapchain_ext: ash::khr::swapchain::Device,
    pub(crate) descriptor_buffer_ext: ash::ext::descriptor_buffer::Device,
    pub(crate) commands: Commands,
    pub(crate) descriptor_sizes: DescriptorSizes,
    /// Sampled images, storage images, samplers. Fixed by `SLANG_PRELUDE`, created once, and
    /// shared by every pipeline layout so pipelines stay interchangeable under one binding
    /// model — the property Metal gets from having a single argument table.
    pub(crate) set_layouts: [vk::DescriptorSetLayout; 3],
    pub(crate) bindless_cap: u32,
    pub(crate) max_multi_draw_count: u32,
    pub(crate) timestamp_period: f32,
    pub(crate) debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

/// The command-level extension a `Raster` reaches for, cloned out of `Device` so that recording
/// does not need to borrow it. `VK_EXT_multi_draw` collapses a run of draws into one, spending
/// fewer driver calls per draw.
#[derive(Clone)]
pub(crate) struct Commands {
    pub(crate) multi_draw: ash::ext::multi_draw::Device,
}

/// `VK_EXT_descriptor_buffer` descriptors are driver-sized, unlike Metal's fixed 8-byte
/// `MTLResourceID`. On this NVIDIA driver a sampled image is 4 bytes, not 8.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DescriptorSizes {
    pub(crate) sampled_image: usize,
    pub(crate) storage_image: usize,
    pub(crate) sampler: usize,
    pub(crate) offset_alignment: u64,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*data).p_message) };
    let level = match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "error",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "warning",
        _ => "info",
    };
    let scope = match kind {
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "validation",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "performance",
        _ => "general",
    };
    eprintln!("[{level} {scope}] {}", message.to_string_lossy());
    vk::FALSE
}

impl Device {
    pub fn new(validation: bool) -> Result<Self, Error> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| Error::new(format!("could not load vulkan-1.dll: {e}")))?;

        let loader_version =
            unsafe { entry.try_enumerate_instance_version()? }.unwrap_or(vk::API_VERSION_1_0);
        if loader_version < vk::API_VERSION_1_4 {
            return Err(Error::new(format!(
                "loader supports only vulkan {}.{}, 1.4 required",
                vk::api_version_major(loader_version),
                vk::api_version_minor(loader_version)
            )));
        }

        let mut layers: Vec<*const c_char> = Vec::new();
        let mut extensions = vec![
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::win32_surface::NAME.as_ptr(),
        ];
        if validation {
            layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let app = vk::ApplicationInfo::default()
            .application_name(c"nga-vk")
            .api_version(vk::API_VERSION_1_4);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_layer_names(&layers)
                    .enabled_extension_names(&extensions),
                None,
            )?
        };

        let debug = validation.then(|| {
            let loader = ash::ext::debug_utils::Instance::load(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger = unsafe { loader.create_debug_utils_messenger(&info, None).unwrap() };
            (loader, messenger)
        });

        let surface_ext = ash::khr::surface::Instance::load(&entry, &instance);
        let win32_surface_ext = ash::khr::win32_surface::Instance::load(&entry, &instance);

        let devices = unsafe { instance.enumerate_physical_devices()? };
        let (physical, queue_family) = devices
            .iter()
            .find_map(|&physical| {
                if unsafe { instance.get_physical_device_properties(physical) }.api_version
                    < vk::API_VERSION_1_4
                {
                    return None;
                }
                let families =
                    unsafe { instance.get_physical_device_queue_family_properties(physical) };
                let family = families.iter().position(|family| {
                    family.queue_flags.contains(
                        vk::QueueFlags::GRAPHICS
                            | vk::QueueFlags::COMPUTE
                            | vk::QueueFlags::TRANSFER,
                    )
                })?;
                Some((physical, family as u32))
            })
            .ok_or(Error::new(
                "no vulkan 1.4 device with a graphics+compute queue",
            ))?;

        let device_extensions = [
            ash::khr::swapchain::NAME.as_ptr(),
            ash::ext::descriptor_buffer::NAME.as_ptr(),
            ash::ext::memory_budget::NAME.as_ptr(),
            ash::ext::multi_draw::NAME.as_ptr(),
            ash::ext::dynamic_rendering_unused_attachments::NAME.as_ptr(),
        ];

        let mut features14 = vk::PhysicalDeviceVulkan14Features::default()
            .maintenance5(true)
            .maintenance6(true)
            .host_image_copy(true);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .draw_indirect_count(true)
            .scalar_block_layout(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true);
        let mut features11 =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        let mut descriptor_buffer =
            vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);
        let mut multi_draw = vk::PhysicalDeviceMultiDrawFeaturesEXT::default().multi_draw(true);
        let mut unused_attachments =
            vk::PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT::default()
                .dynamic_rendering_unused_attachments(true);
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .multi_draw_indirect(true)
            .draw_indirect_first_instance(true)
            .fill_mode_non_solid(true)
            .independent_blend(true);

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_info)
                    .enabled_extension_names(&device_extensions)
                    .enabled_features(&features)
                    .push(&mut features11)
                    .push(&mut features12)
                    .push(&mut features13)
                    .push(&mut features14)
                    .push(&mut descriptor_buffer)
                    .push(&mut multi_draw)
                    .push(&mut unused_attachments),
                None,
            )?
        };

        let mut descriptor_props = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut multi_draw_props = vk::PhysicalDeviceMultiDrawPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceProperties2::default()
            .push(&mut descriptor_props)
            .push(&mut multi_draw_props);
        unsafe { instance.get_physical_device_properties2(physical, &mut props) };
        let name = unsafe { CStr::from_ptr(props.properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        // Read out of `props` before touching the structs it holds mutable borrows of.
        let limits = props.properties.limits;
        let timestamp_period = limits.timestamp_period;

        let bindless_cap = 1024;
        let set_layouts = [
            Self::set_layout(&device, vk::DescriptorType::SAMPLED_IMAGE, bindless_cap)?,
            Self::set_layout(&device, vk::DescriptorType::STORAGE_IMAGE, bindless_cap)?,
            Self::set_layout(&device, vk::DescriptorType::SAMPLER, bindless_cap)?,
        ];

        Ok(Self {
            set_layouts,
            bindless_cap,
            queue_family,
            memory: unsafe { instance.get_physical_device_memory_properties(physical) },
            limits,
            name,
            descriptor_sizes: DescriptorSizes {
                sampled_image: descriptor_props.sampled_image_descriptor_size,
                storage_image: descriptor_props.storage_image_descriptor_size,
                sampler: descriptor_props.sampler_descriptor_size,
                offset_alignment: descriptor_props.descriptor_buffer_offset_alignment,
            },
            commands: Commands {
                multi_draw: ash::ext::multi_draw::Device::load(&instance, &device),
            },
            max_multi_draw_count: multi_draw_props.max_multi_draw_count,
            timestamp_period,
            swapchain_ext: ash::khr::swapchain::Device::load(&instance, &device),
            descriptor_buffer_ext: ash::ext::descriptor_buffer::Device::load(&instance, &device),
            surface_ext,
            win32_surface_ext,
            physical,
            device,
            instance,
            entry,
            debug,
        })
    }

    fn set_layout(
        device: &ash::Device,
        kind: vk::DescriptorType,
        count: u32,
    ) -> Result<vk::DescriptorSetLayout, Error> {
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(kind)
            .descriptor_count(count)
            .stage_flags(vk::ShaderStageFlags::ALL)];
        let layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT)
                    .bindings(&bindings),
                None,
            )?
        };
        Ok(layout)
    }

    /// The adapter name, for the one line of diagnostics every example prints.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Blocks until the device is idle. The one piece of teardown a caller has to drive, and
    /// the only reason `vkDeviceWaitIdle` used to appear in an example.
    pub fn wait_idle(&self) -> Result<(), Error> {
        unsafe { self.device.device_wait_idle()? };
        Ok(())
    }

    /// Whether this adapter can do something outside the shared surface.
    ///
    /// The answers are deliberately conservative: a feature reports `true` only when the
    /// backend actually implements a path for it today, not merely when the hardware could.
    pub fn supports(&self, feature: Feature) -> bool {
        match feature {
            // Metal-only. Not an API gap — this is TBDR hardware showing through.
            Feature::TileShaders => false,
            // `VK_EXT_mesh_shader` is not requested at device creation, so no.
            Feature::MeshShaders => false,
            // Metal's ICBs have no Vulkan counterpart of the same shape; `draw_indirect_count`
            // and `draw_multi` cover the same ground differently and are always available.
            Feature::IndirectCommands => false,
            Feature::MultiDraw => self.max_multi_draw_count > 1,
            Feature::Timestamps => self.timestamp_period > 0.0,
            Feature::Occlusion => true,
        }
    }

    /// How many descriptors a [`ResourceHeap`] may hold. Metal's argument buffer has no such
    /// ceiling, so this is the binding one.
    pub fn max_resources(&self) -> u32 {
        self.bindless_cap
    }

    /// The longest run `Raster::draw_multi` will accept.
    pub fn max_multi_draw(&self) -> u32 {
        self.max_multi_draw_count
    }

    /// A sampler object, to be written into a heap slot. The returned value owns a
    /// `VkSampler`, so it must outlive every heap slot pointing at it.
    pub fn sampler(&self, desc: &SamplerDesc) -> Result<SamplerState, Error> {
        let address = address_mode(desc.address);
        let mut info = vk::SamplerCreateInfo::default()
            .min_filter(filter(desc.min))
            .mag_filter(filter(desc.mag))
            .mipmap_mode(mip_mode(desc.mip))
            .address_mode_u(address)
            .address_mode_v(address)
            .address_mode_w(address)
            .max_lod(vk::LOD_CLAMP_NONE);
        if desc.anisotropy > 1 {
            info = info
                .anisotropy_enable(true)
                .max_anisotropy((desc.anisotropy as f32).min(self.limits.max_sampler_anisotropy));
        }
        if let Some(op) = desc.compare {
            info = info.compare_enable(true).compare_op(compare_op(op));
        }
        let sampler = unsafe { self.device.create_sampler(&info, None)? };
        Ok(SamplerState(sampler, self.device.clone()))
    }

    pub(crate) fn queue(&self) -> vk::Queue {
        unsafe { self.device.get_device_queue(self.queue_family, 0) }
    }

    /// Memory types matching `flags` and carrying none of `without`, best first. A list
    /// rather than one index, because the heap being full is a different failure from the
    /// heap not existing: the 214 MiB BAR window fills long before it disappears, so the
    /// caller retries the next candidate on `ERROR_OUT_OF_DEVICE_MEMORY`.
    pub(crate) fn memory_types(
        &self,
        mask: u32,
        flags: vk::MemoryPropertyFlags,
        without: vk::MemoryPropertyFlags,
    ) -> Vec<u32> {
        (0..self.memory.memory_type_count)
            .filter(|&index| {
                let properties = self.memory.memory_types[index as usize].property_flags;
                mask & (1 << index) != 0
                    && properties.contains(flags)
                    && !properties.intersects(without)
            })
            .collect()
    }

    /// `VK_EXT_memory_budget` reports what is actually free right now, which for the BAR heap
    /// is the number that matters — it is shared with every other process on the desktop, so
    /// its 214 MiB is never all ours.
    pub fn budget(&self, heap: u32) -> u64 {
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push(&mut budget);
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.physical, &mut properties)
        };
        budget.heap_budget[heap as usize].saturating_sub(budget.heap_usage[heap as usize])
    }

    pub fn heap_of(&self, memory_type: u32) -> u32 {
        self.memory.memory_types[memory_type as usize].heap_index
    }

    pub(crate) fn memory_type(&self, mask: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory.memory_type_count).find(|&index| {
            mask & (1 << index) != 0
                && self.memory.memory_types[index as usize]
                    .property_flags
                    .contains(flags)
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            for &layout in &self.set_layouts {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.device.destroy_device(None);
            if let Some((loader, messenger)) = &self.debug {
                loader.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Where an `Arena`'s memory lives. Metal has one answer to this question — `Shared` under
/// UMA — so `nga` has no such enum. On discrete Windows the three are physically different
/// memory with opposite performance, and picking the wrong one is silent, so it is explicit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Placement {
    /// Host-visible *device-local*: the BAR window. CPU writes land in VRAM and the GPU reads
    /// them at full speed. Without ReBAR this is 214 MiB shared with every other process, so
    /// it is reserved for per-frame root data and nothing else.
    Bar,
    /// Host-visible system RAM without `HOST_CACHED` — write-combined. Uncached CPU reads from
    /// it are catastrophically slow, sequential writes are fast, and the GPU reads it once over
    /// PCIe. That is exactly the staging-source access pattern, and it costs no BAR.
    Upload,
    /// Host-visible system RAM *with* `HOST_CACHED`, for GPU-to-CPU readback, where the CPU is
    /// the reader and wants a cache.
    Readback,
}

impl Placement {
    /// Preferred property flags first, then the flags every host-visible type has. The BAR
    /// fallback matters on a machine whose host-visible device-local heap does not exist;
    /// the `Upload`/`Readback` fallback matters because `HOST_CACHED` is not guaranteed.
    fn candidates(self) -> [(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags); 2] {
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let none = vk::MemoryPropertyFlags::empty();
        match self {
            Placement::Bar => [
                (host | vk::MemoryPropertyFlags::DEVICE_LOCAL, none),
                (host, none),
            ],
            Placement::Upload => [(host, vk::MemoryPropertyFlags::HOST_CACHED), (host, none)],
            Placement::Readback => [
                (host | vk::MemoryPropertyFlags::HOST_CACHED, none),
                (host, none),
            ],
        }
    }
}

/// The Metal `Arena` writes structs straight into GPU memory and hands out the address.
/// On discrete Windows that only works inside the host-visible device-local BAR window,
/// which is 214 MiB on this GPU with no ReBAR — enough for root data, too small for meshes.
/// So the arena is no longer one kind of memory: see `Placement`.
pub struct Arena {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) base: u64,
    pub(crate) cpu: *mut u8,
    pub(crate) size: usize,
    pub(crate) head: AtomicUsize,
    pub(crate) device: ash::Device,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new(device: &Device, size: usize, memory: Memory) -> Result<Self, Error> {
        // An arena is host-mapped by definition, so `Memory::Device` asks for the closest a
        // host-mapped allocation gets to the GPU: the BAR window.
        let placement = match memory {
            Memory::Device => Placement::Bar,
            Memory::Upload => Placement::Upload,
            Memory::Readback => Placement::Readback,
        };
        let usage = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER;
        let info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.device.create_buffer(&info, None)? };
        let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };

        let mut indices = Vec::new();
        for (flags, without) in placement.candidates() {
            for index in device.memory_types(requirements.memory_type_bits, flags, without) {
                if !indices.contains(&index) {
                    indices.push(index);
                }
            }
        }
        if indices.is_empty() {
            unsafe { device.device.destroy_buffer(buffer, None) };
            return Err(Error::new(format!(
                "no host-visible memory type for a {placement:?} arena"
            )));
        }

        // A full heap and an absent heap are different failures. The BAR window is 214 MiB and
        // shared with the desktop, so the first candidate can simply be out of room — that is a
        // reason to take the next one, not to give up.
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut allocated = Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        for &index in &indices {
            allocated = unsafe {
                device.device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(index)
                        .push(&mut flags),
                    None,
                )
            };
            match allocated {
                Ok(_) => break,
                Err(
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                ) => continue,
                Err(_) => break,
            }
        }
        let memory = match allocated {
            Ok(memory) => memory,
            Err(result) => {
                unsafe { device.device.destroy_buffer(buffer, None) };
                let free = indices
                    .iter()
                    .map(|&index| device.budget(device.heap_of(index)))
                    .max()
                    .unwrap_or(0);
                return Err(Error::new(format!(
                    "{placement:?} arena of {size} bytes did not fit: {result:?}, largest                      candidate heap has {free} bytes free"
                )));
            }
        };
        unsafe { device.device.bind_buffer_memory(buffer, memory, 0)? };
        let cpu = unsafe {
            device
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
        };
        let base = unsafe {
            device
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        Ok(Self {
            buffer,
            memory,
            base,
            cpu: cpu as *mut u8,
            size,
            head: AtomicUsize::new(0),
            device: device.device.clone(),
        })
    }

    pub fn alloc_bytes(&self, size: usize, align: usize) -> Option<(*mut u8, u64)> {
        let mut head = self.head.load(Ordering::Relaxed);
        loop {
            let offset = (head + align - 1) & !(align - 1);
            if offset + size > self.size {
                return None;
            }
            match self.head.compare_exchange_weak(
                head,
                offset + size,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return unsafe { Some((self.cpu.add(offset), self.base + offset as u64)) },
                Err(current) => head = current,
            }
        }
    }

    pub fn alloc<T>(&self, value: T) -> Option<GpuBox<'_, T>> {
        let (cpu, gpu) = self.alloc_bytes(size_of::<T>(), align_of::<T>())?;
        let cpu = cpu as *mut T;
        unsafe { cpu.write(value) };
        Some(GpuBox {
            cpu: NonNull::new(cpu)?,
            gpu: Gpu::new(gpu),
        })
    }

    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> Option<GpuBox<'_, [T]>> {
        let (cpu, gpu) = self.alloc_bytes(size_of_val(values), align_of::<T>())?;
        let cpu = cpu as *mut T;
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), cpu, values.len()) };
        Some(GpuBox {
            cpu: NonNull::new(std::ptr::slice_from_raw_parts_mut(cpu, values.len()))?,
            gpu: Gpu::new(gpu),
        })
    }

    pub fn alloc_scratch<T>(&self, len: usize) -> Option<GpuBox<'_, [T]>> {
        let (cpu, gpu) = self.alloc_bytes(size_of::<T>() * len, align_of::<T>())?;
        Some(GpuBox {
            cpu: NonNull::new(std::ptr::slice_from_raw_parts_mut(cpu as *mut T, len))?,
            gpu: Gpu::new(gpu),
        })
    }

    pub fn reset(&self) {
        self.head.store(0, Ordering::Relaxed);
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            self.device.unmap_memory(self.memory);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub struct HeapBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) gpu: u64,
    pub(crate) len: usize,
    pub(crate) alloc: Alloc,
}

impl HeapBuffer {
    /// The device address, for putting in a struct a shader dereferences.
    pub fn gpu<T>(&self) -> Gpu<'_, T> {
        Gpu::new(self.gpu)
    }

    /// The raw address, for the cases where the borrow has to be discarded — a heap allocation
    /// outlives every frame, so `Gpu<'static, T>` is the honest type and only the caller knows.
    pub fn address(&self) -> u64 {
        self.gpu
    }

    pub fn gpu_mut<T>(&mut self) -> GpuMut<'_, T> {
        GpuMut::new(self.gpu)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct HeapTexture {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) format: vk::Format,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) alloc: Alloc,
    pub(crate) layout: Cell<vk::ImageLayout>,
}

impl HeapTexture {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> Format {
        from_vk_format(self.format).expect("texture format is outside nga::Format")
    }
}

#[derive(Clone, Copy)]
pub struct TextureDesc {
    pub(crate) format: vk::Format,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) usage: vk::ImageUsageFlags,
    pub(crate) aspect: vk::ImageAspectFlags,
}

impl TextureDesc {
    pub fn texture_2d(fmt: Format, width: u32, height: u32) -> Self {
        Self {
            format: format(fmt),
            width,
            height,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
        }
    }

    pub fn depth(fmt: Format, width: u32, height: u32) -> Self {
        Self {
            format: format(fmt),
            width,
            height,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            aspect: vk::ImageAspectFlags::DEPTH,
        }
    }

    /// Extra usage beyond what the constructor implies — storage writes, colour attachment.
    pub fn usage(mut self, extra: Usage) -> Self {
        self.usage |= usage(extra);
        self
    }
}

/// Batches ahead of this many submissions are made to wait, so a command buffer is never
/// re-recorded while the GPU is still reading it.
pub(crate) const UPLOAD_BATCHES: usize = 4;

/// Staging ring size. Large enough that a frame's worth of uploads never wraps, small enough
/// that it is unremarkable in system RAM — and it costs no BAR, which is the whole point.
pub(crate) const UPLOAD_RING: usize = 32 * 1024 * 1024;

/// `vkCmdUpdateBuffer`'s hard limit. Below it the data rides in the command buffer and the
/// staging ring is not touched at all.
pub(crate) const UPDATE_LIMIT: usize = 65536;

/// The thing Metal never needs: a machine for getting bytes into memory the CPU cannot see.
///
/// Under UMA an upload is a memcpy that has already happened by the time the call returns.
/// On discrete hardware without ReBAR the destination is unmappable, so every byte goes
/// CPU -> write-combined system RAM -> DMA -> VRAM. The only question is what that costs, and
/// the answer is entirely about batching: a copy recorded here is a memcpy plus a few words in
/// a command buffer, whereas a copy that submits and waits is a full device drain. So nothing
/// here submits until `flush`, and `flush` signals a timeline instead of idling the queue.
///
/// The ring is `Placement::Upload` — deliberately *not* the BAR window. Staging memory is read
/// once by the DMA engine and never by a shader, so spending the scarce 214 MiB on it would be
/// the worst trade available.
pub(crate) struct Uploader {
    pub(crate) ring: Arena,
    pub(crate) pool: vk::CommandPool,
    pub(crate) buffers: Vec<vk::CommandBuffer>,
    pub(crate) timeline: vk::Semaphore,
    pub(crate) queue: vk::Queue,
    pub(crate) device: ash::Device,
    pub(crate) head: usize,
    pub(crate) submitted: u64,
    pub(crate) recording: bool,
}

impl Uploader {
    pub fn new(device: &Device, ring: usize) -> Result<Self, Error> {
        let pool = unsafe {
            device.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(device.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let buffers = unsafe {
            device.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(UPLOAD_BATCHES as u32),
            )?
        };
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline = unsafe {
            device
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push(&mut kind), None)?
        };
        Ok(Self {
            ring: Arena::new(device, ring, Memory::Upload)?,
            pool,
            buffers,
            timeline,
            queue: device.queue(),
            device: device.device.clone(),
            head: 0,
            submitted: 0,
            recording: false,
        })
    }

    /// The open batch, started on first use. Recording is free; only `flush` costs a submission.
    pub(crate) fn open(&mut self) -> Result<vk::CommandBuffer, Error> {
        let command_buffer = self.buffers[self.submitted as usize % UPLOAD_BATCHES];
        if !self.recording {
            let batches = UPLOAD_BATCHES as u64;
            if self.submitted >= batches {
                self.wait(self.submitted + 1 - batches)?;
            }
            unsafe {
                self.device
                    .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
                self.device.begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?
            };
            self.recording = true;
        }
        Ok(command_buffer)
    }

    /// Bump-allocate staging space, returning the offset into `ring.buffer` and where to write.
    ///
    /// The ring wraps by draining: when a request does not fit, the open batch is submitted and
    /// every outstanding one is waited on, then the head resets. That is the one blocking path
    /// left, it is bounded by the work already recorded rather than by the whole device, and a
    /// ring sized for the frame's uploads never reaches it. Chunking is what keeps a single
    /// oversized upload from needing a ring as large as itself.
    pub fn stage(&mut self, size: usize, align: usize) -> Result<(u64, *mut u8), Error> {
        if size > self.ring.size {
            return Err(Error::new(format!(
                "staged copy of {size} bytes exceeds the {} byte upload ring",
                self.ring.size
            )));
        }
        let mut offset = (self.head + align - 1) & !(align - 1);
        if offset + size > self.ring.size {
            self.flush()?;
            self.wait(self.submitted)?;
            self.head = 0;
            offset = 0;
        }
        self.head = offset + size;
        Ok((offset as u64, unsafe { self.ring.cpu.add(offset) }))
    }

    /// How much ring is left before a wrap forces a wait.
    pub fn available(&self) -> usize {
        self.ring.size - self.head
    }

    /// How much of `want` to stage next, so a copy larger than the ring streams through it.
    /// The remaining tail is used when it is worth using; below a quarter of the ring it is
    /// not, since a runt copy would wrap on the very next iteration anyway — better to ask for
    /// a full chunk and let `stage` wrap once.
    pub fn chunk(&self, want: usize) -> usize {
        let tail = self.available();
        let room = match tail * 4 >= self.ring.size {
            true => tail,
            false => self.ring.size,
        };
        want.min(room)
    }

    /// Submits whatever is recorded and returns the timeline value that marks it complete.
    /// Callers that can express "the GPU waits" should wait on that value rather than blocking.
    pub fn flush(&mut self) -> Result<u64, Error> {
        if !self.recording {
            return Ok(self.submitted);
        }
        let command_buffer = self.buffers[self.submitted as usize % UPLOAD_BATCHES];

        // One barrier per batch rather than per copy: everything staged here is read later by
        // some stage or other, and at batch granularity the coarse mask costs nothing.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(
                        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                    )]),
            );
            self.device.end_command_buffer(command_buffer)?;
        }

        let value = self.submitted + 1;
        let infos = [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        unsafe {
            self.device.queue_submit2(
                self.queue,
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&infos)
                    .signal_semaphore_infos(&signals)],
                vk::Fence::null(),
            )?
        };
        self.submitted = value;
        self.recording = false;
        Ok(value)
    }

    pub fn wait(&self, value: u64) -> Result<(), Error> {
        let semaphores = [self.timeline];
        let values = [value];
        unsafe {
            self.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&semaphores)
                    .values(&values),
                u64::MAX,
            )?
        };
        Ok(())
    }

    /// Load-time convenience: submit the batch and block until it lands. One wait for an entire
    /// asset load, where the old immediate path drained the device once per resource.
    pub fn flush_wait(&mut self) -> Result<(), Error> {
        let value = self.flush()?;
        self.wait(value)
    }
}

impl Drop for Uploader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_semaphore(self.timeline, None);
        }
    }
}

/// `MTLHeap` + TLSF, rebuilt on one `VkDeviceMemory`.
///
/// Two Vulkan-only concerns that `nga` never meets: every allocation is aligned to
/// `bufferImageGranularity` so a buffer and an image can never share a granularity page, and
/// the memory is `DEVICE_LOCAL` without `HOST_VISIBLE`, because the host-visible device-local
/// heap on this GPU is only 214 MiB. Both buffer and texture contents therefore arrive by
/// staged copy — `VK_EXT_host_image_copy` would have spared textures that, but it is unusable
/// on this driver (see `upload` and `examples/probe.rs`).
///
/// Every staged copy is *recorded*, not submitted. Call `flush` (or `flush_wait`) once the
/// batch is built; nothing reaches the GPU before that.
pub struct GpuHeap {
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) tlsf: Tlsf,
    pub(crate) granularity: u64,
    pub(crate) device: ash::Device,
    pub(crate) uploader: Uploader,
}

impl GpuHeap {
    pub const BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw()
            | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
            | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
            | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw()
            | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
            | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
    );

    pub fn new(device: &Device, size: u64) -> Result<Self, Error> {
        Self::with_ring(device, size, UPLOAD_RING)
    }

    pub fn with_ring(device: &Device, size: u64, ring: usize) -> Result<Self, Error> {
        // `VK_KHR_maintenance4` answers "which heaps fit this?" from a create-info, with no
        // object created — the exact complaint the post levels at Vulkan's allocator design.
        let probe = vk::BufferCreateInfo::default()
            .size(size.min(1 << 20))
            .usage(Self::BUFFER_USAGE);
        let mut requirements = vk::MemoryRequirements2::default();
        unsafe {
            device.device.get_device_buffer_memory_requirements(
                &vk::DeviceBufferMemoryRequirements::default().create_info(&probe),
                &mut requirements,
            )
        };
        let index = device
            .memory_type(
                requirements.memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or(Error::new("no device-local memory type for the heap"))?;

        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = unsafe {
            device.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(size)
                    .memory_type_index(index)
                    .push(&mut flags),
                None,
            )?
        };
        let granularity = device.limits.buffer_image_granularity.max(256);
        Ok(Self {
            memory,
            tlsf: Tlsf::new(size, granularity),
            granularity,
            device: device.device.clone(),
            uploader: Uploader::new(device, ring)?,
        })
    }

    pub fn alloc_buffer(&mut self, len: usize) -> Result<HeapBuffer, Error> {
        let info = vk::BufferCreateInfo::default()
            .size(len as u64)
            .usage(Self::BUFFER_USAGE);
        let mut requirements = vk::MemoryRequirements2::default();
        unsafe {
            self.device.get_device_buffer_memory_requirements(
                &vk::DeviceBufferMemoryRequirements::default().create_info(&info),
                &mut requirements,
            )
        };
        let requirements = requirements.memory_requirements;

        let alloc = self
            .tlsf
            .alloc(requirements.size.max(1), self.granularity)
            .ok_or(Error::new("GPU heap is out of space"))?;
        let buffer = unsafe { self.device.create_buffer(&info, None)? };
        unsafe {
            self.device
                .bind_buffer_memory(buffer, self.memory, alloc.offset)?
        };
        let gpu = unsafe {
            self.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        Ok(HeapBuffer {
            buffer,
            gpu,
            len,
            alloc,
        })
    }

    pub fn alloc_texture(&mut self, desc: &TextureDesc) -> Result<HeapTexture, Error> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(desc.format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(desc.usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let mut requirements = vk::MemoryRequirements2::default();
        unsafe {
            self.device.get_device_image_memory_requirements(
                &vk::DeviceImageMemoryRequirements::default().create_info(&info),
                &mut requirements,
            )
        };
        let requirements = requirements.memory_requirements;

        let alloc = self
            .tlsf
            .alloc(
                requirements.size,
                requirements.alignment.max(self.granularity),
            )
            .ok_or(Error::new("GPU heap is out of space"))?;
        let image = unsafe { self.device.create_image(&info, None)? };
        unsafe {
            self.device
                .bind_image_memory(image, self.memory, alloc.offset)?
        };
        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(desc.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(desc.aspect)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )?
        };
        Ok(HeapTexture {
            image,
            view,
            format: desc.format,
            width: desc.width,
            height: desc.height,
            alloc,
            layout: Cell::new(vk::ImageLayout::UNDEFINED),
        })
    }

    /// `nga`'s `HeapBuffer::write` is a memcpy, because Metal heaps are host-visible under UMA.
    /// Here the destination is device-local, so the bytes go through the upload ring and a
    /// recorded copy. This is the half of `StorageMode::Shared` that Vulkan cannot give back —
    /// but it is recorded, not submitted, so what it costs is a memcpy and not a device drain.
    ///
    /// Writes at or below `UPDATE_LIMIT` skip the ring entirely: `vkCmdUpdateBuffer` embeds the
    /// data in the command buffer itself, which is both fewer moving parts and no ring pressure.
    pub fn write<T: Copy>(&mut self, buffer: &HeapBuffer, offset: usize, values: &[T]) {
        self.try_write(buffer, offset, values)
            .expect("staged buffer write failed")
    }

    pub fn try_write<T: Copy>(
        &mut self,
        buffer: &HeapBuffer,
        offset: usize,
        values: &[T],
    ) -> Result<(), Error> {
        let bytes = size_of_val(values);
        debug_assert!(offset + bytes <= buffer.len);
        if bytes == 0 {
            return Ok(());
        }
        let source = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, bytes) };

        if bytes <= UPDATE_LIMIT && bytes.is_multiple_of(4) && offset.is_multiple_of(4) {
            let command_buffer = self.uploader.open()?;
            unsafe {
                self.device
                    .cmd_update_buffer(command_buffer, buffer.buffer, offset as u64, source)
            };
            return Ok(());
        }

        // Chunked against whatever the ring has left, so a write larger than the ring streams
        // through it rather than failing — the old path simply panicked past 8 MiB.
        let mut written = 0;
        while written < bytes {
            let chunk = self.uploader.chunk(bytes - written);
            let (staged, cpu) = self.uploader.stage(chunk, 16)?;
            unsafe { std::ptr::copy_nonoverlapping(source.as_ptr().add(written), cpu, chunk) };
            let command_buffer = self.uploader.open()?;
            unsafe {
                self.device.cmd_copy_buffer(
                    command_buffer,
                    self.uploader.ring.buffer,
                    buffer.buffer,
                    &[vk::BufferCopy {
                        src_offset: staged,
                        dst_offset: (offset + written) as u64,
                        size: chunk as u64,
                    }],
                )
            };
            written += chunk;
        }
        Ok(())
    }

    /// `VK_EXT_host_image_copy` was meant to go here: CPU memory straight into the image with
    /// no staging buffer and no command buffer, which is the closest Vulkan gets to Metal's
    /// `replace_region`. On this driver it is unusable — an image created with
    /// `HOST_TRANSFER_EXT` reports `memoryTypeBits == 0`, so no memory can back it, even though
    /// the extension is advertised and `R8G8B8A8_UNORM` claims `HOST_IMAGE_TRANSFER` support.
    /// See `examples/probe.rs`. So uploads stage through a buffer, exactly like `write`.
    pub fn upload(&mut self, texture: &HeapTexture, bytes: &[u8]) -> Result<(), Error> {
        let (width, height) = (texture.width, texture.height);
        let texels = width as usize * height as usize;
        if texels == 0 || bytes.is_empty() {
            return Ok(());
        }
        if !bytes.len().is_multiple_of(texels) {
            return Err(Error::new(format!(
                "{} bytes is not a whole number of texels for a {width}x{height} image",
                bytes.len()
            )));
        }
        // The texel size comes from the caller's own slice rather than a format table: an upload
        // is by definition the whole image, so the division is exact and every format works.
        let row = bytes.len() / height as usize;
        if row > self.uploader.ring.size {
            return Err(Error::new(format!(
                "one image row is {row} bytes, larger than the {} byte upload ring",
                self.uploader.ring.size
            )));
        }

        let image = texture.image;
        let old = texture.layout.get();
        let command_buffer = self.uploader.open()?;
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[Self::barrier(
                    image,
                    old,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                )]),
            )
        };
        texture.layout.set(vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        // Rows are the chunking unit, so a texture far larger than the ring still streams
        // through it. `bufferRowLength` stays 0 because each chunk is tightly packed.
        let mut first = 0u32;
        while first < height {
            let remaining = (height - first) as usize * row;
            let rows = ((self.uploader.chunk(remaining) / row).max(1) as u32).min(height - first);
            let size = rows as usize * row;
            let (staged, cpu) = self.uploader.stage(size, 16)?;
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(first as usize * row), cpu, size)
            };
            let command_buffer = self.uploader.open()?;
            unsafe {
                self.device.cmd_copy_buffer_to_image(
                    command_buffer,
                    self.uploader.ring.buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy {
                        buffer_offset: staged,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D {
                            x: 0,
                            y: first as i32,
                            z: 0,
                        },
                        image_extent: vk::Extent3D {
                            width,
                            height: rows,
                            depth: 1,
                        },
                        ..Default::default()
                    }],
                )
            };
            first += rows;
        }
        Ok(())
    }

    /// Submit the recorded batch and block until it lands. The load-time entry point: one wait
    /// for a whole scene, where the old path drained the device once per resource.
    pub fn flush(&mut self) -> Result<(), Error> {
        self.uploader.flush_wait()
    }

    /// Submit without blocking, returning the timeline value to wait on. Pair with
    /// `Frame::after_uploads` to let the GPU do the waiting instead of the CPU.
    pub fn flush_async(&mut self) -> Result<u64, Error> {
        self.uploader.flush()
    }

    /// Metal has no image layouts, so this method has no counterpart at all. Without
    /// `VK_KHR_unified_image_layouts` — absent on this driver — a transition is unavoidable,
    /// but it is only a barrier: recording it into the open upload batch makes it free, where
    /// submitting one command buffer per transition made it the most expensive call here.
    fn barrier(
        image: vk::Image,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
    ) -> vk::ImageMemoryBarrier2<'static> {
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(old)
            .new_layout(new)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            )
    }

    pub fn free(&mut self, alloc: Alloc) {
        self.tlsf.free(alloc.block);
    }

    /// Moves a texture into the layout `usage` implies, on the upload queue rather than in a
    /// frame. Used at load time, before any frame has been recorded.
    ///
    /// Metal needs no equivalent — there is no layout to choose — which is why the argument is
    /// a `Usage` and not something more Vulkan-shaped.
    pub fn transition(&mut self, texture: &HeapTexture, usage: Usage) {
        let layout = image_layout(usage);
        let old = texture.layout.get();
        if old == layout {
            return;
        }
        let command_buffer = self.uploader.open().expect("upload batch");
        let barriers = [image_barrier(texture, old, layout)];
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            )
        };
        texture.layout.set(layout);
    }

    /// The buffer counterpart of [`GpuHeap::free_texture`].
    pub fn free_buffer(&mut self, buffer: HeapBuffer) {
        unsafe { self.device.destroy_buffer(buffer.buffer, None) };
        self.tlsf.free(buffer.alloc.block);
    }

    /// Destroys the image and view as well as releasing the block. Validation reports a leaked
    /// `VkImage` without this, because `HeapTexture` has no `Drop` of its own — it cannot, it
    /// does not know which device it came from.
    pub fn free_texture(&mut self, texture: HeapTexture) {
        unsafe {
            self.device.destroy_image_view(texture.view, None);
            self.device.destroy_image(texture.image, None);
        }
        self.tlsf.free(texture.alloc.block);
    }
}

impl Drop for GpuHeap {
    fn drop(&mut self) {
        unsafe {
            self.device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HeapRange {
    pub(crate) base: u32,
    pub(crate) len: u32,
}

/// Metal's bindless heap is one array of 8-byte `MTLResourceID`s that MSL reinterprets per use
/// (`heap[i].t`, `nga_heap_as(nga_sampler, heap)[i].s`). Vulkan cannot do that: a descriptor's
/// size is driver-defined, and sampler descriptors must live in a buffer created with
/// `SAMPLER_DESCRIPTOR_BUFFER_BIT` while image descriptors need `RESOURCE_DESCRIPTOR_BUFFER_BIT`.
///
/// So one Metal heap becomes three parallel arrays — sampled images, storage images, samplers —
/// across two buffers. The index space stays unified so `range.base + slot` arithmetic ports
/// unchanged; the cost is that a slot reserves room in all three arrays.
pub struct ResourceHeap {
    pub(crate) resources: vk::Buffer,
    pub(crate) resources_memory: vk::DeviceMemory,
    pub(crate) resources_cpu: NonNull<u8>,
    pub(crate) resources_address: u64,
    pub(crate) samplers: vk::Buffer,
    pub(crate) samplers_memory: vk::DeviceMemory,
    pub(crate) samplers_cpu: NonNull<u8>,
    pub(crate) samplers_address: u64,
    pub(crate) sampled_offset: u64,
    pub(crate) storage_offset: u64,
    pub(crate) sizes: DescriptorSizes,
    pub(crate) cap: u32,
    pub(crate) head: u32,
    pub(crate) device: ash::Device,
    pub(crate) ext: ash::ext::descriptor_buffer::Device,
}

impl ResourceHeap {
    pub fn new(device: &Device, cap: u32) -> Result<Self, Error> {
        let sampled_size = unsafe {
            device
                .descriptor_buffer_ext
                .get_descriptor_set_layout_size(device.set_layouts[0])
        };
        let storage_size = unsafe {
            device
                .descriptor_buffer_ext
                .get_descriptor_set_layout_size(device.set_layouts[1])
        };
        let sampler_size = unsafe {
            device
                .descriptor_buffer_ext
                .get_descriptor_set_layout_size(device.set_layouts[2])
        };

        let align = device.descriptor_sizes.offset_alignment;
        let sampled_offset = 0;
        let storage_offset = (sampled_size + align - 1) & !(align - 1);
        let resources_size = storage_offset + storage_size;

        let (resources, resources_memory, resources_cpu, resources_address) = Self::buffer(
            device,
            resources_size,
            vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
        )?;
        let (samplers, samplers_memory, samplers_cpu, samplers_address) = Self::buffer(
            device,
            sampler_size.max(align),
            vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT,
        )?;

        Ok(Self {
            resources,
            resources_memory,
            resources_cpu,
            resources_address,
            samplers,
            samplers_memory,
            samplers_cpu,
            samplers_address,
            sampled_offset,
            storage_offset,
            sizes: device.descriptor_sizes,
            cap,
            head: 0,
            device: device.device.clone(),
            ext: device.descriptor_buffer_ext.clone(),
        })
    }

    fn buffer(
        device: &Device,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, NonNull<u8>, u64), Error> {
        let info = vk::BufferCreateInfo::default()
            .size(size.max(4))
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let buffer = unsafe { device.device.create_buffer(&info, None)? };
        let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };
        let index = device
            .memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or(Error::new(
                "no host-visible memory for the descriptor buffer",
            ))?;
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = unsafe {
            device.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(index)
                    .push(&mut flags),
                None,
            )?
        };
        unsafe { device.device.bind_buffer_memory(buffer, memory, 0)? };
        let cpu = unsafe {
            device
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
        };
        let address = unsafe {
            device
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        Ok((
            buffer,
            memory,
            NonNull::new(cpu as *mut u8).unwrap(),
            address,
        ))
    }

    pub fn alloc_range(&mut self, len: u32) -> Result<HeapRange, Error> {
        if self.head + len > self.cap {
            return Err(Error::new("resource heap is full"));
        }
        let base = self.head;
        self.head += len;
        Ok(HeapRange { base, len })
    }

    /// `vkGetDescriptorEXT` is the direct analogue of reading `texture.gpuResourceID()`: it hands
    /// back the opaque descriptor bytes for the caller to place. The difference is that the size
    /// is a runtime property and the destination is a typed array, not one shared `u64` table.
    pub fn write_texture(&self, range: HeapRange, slot: u32, texture: &HeapTexture) -> Texture2d {
        debug_assert!(slot < range.len);
        let image = vk::DescriptorImageInfo::default()
            .image_view(texture.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let data = vk::DescriptorDataEXT {
            p_sampled_image: &image,
        };
        self.write(
            vk::DescriptorType::SAMPLED_IMAGE,
            data,
            self.sampled_offset + (range.base + slot) as u64 * self.sizes.sampled_image as u64,
            self.sizes.sampled_image,
            self.resources_cpu,
        );
        Texture2d(Handle((range.base + slot) as u64))
    }

    /// A texture that is written and sampled in the same frame stays in `GENERAL`, so its
    /// sampled descriptor has to record that layout. Metal has no equivalent choice to make.
    /// Writes a *sampled* descriptor for a texture that is not in the usual read-only layout.
    /// The case that needs it: a texture written by compute and sampled in the same frame stays
    /// in `Usage::STORAGE`, and sampling it from a read-only descriptor would be invalid.
    pub fn write_texture_in(
        &self,
        range: HeapRange,
        slot: u32,
        texture: &HeapTexture,
        usage: Usage,
    ) -> Texture2d {
        debug_assert!(slot < range.len);
        let image = vk::DescriptorImageInfo::default()
            .image_view(texture.view)
            .image_layout(image_layout(usage));
        let data = vk::DescriptorDataEXT {
            p_sampled_image: &image,
        };
        self.write(
            vk::DescriptorType::SAMPLED_IMAGE,
            data,
            self.sampled_offset + (range.base + slot) as u64 * self.sizes.sampled_image as u64,
            self.sizes.sampled_image,
            self.resources_cpu,
        );
        Texture2d(Handle((range.base + slot) as u64))
    }

    pub fn write_storage(
        &self,
        range: HeapRange,
        slot: u32,
        texture: &HeapTexture,
    ) -> Texture2dRW {
        debug_assert!(slot < range.len);
        let image = vk::DescriptorImageInfo::default()
            .image_view(texture.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let data = vk::DescriptorDataEXT {
            p_storage_image: &image,
        };
        self.write(
            vk::DescriptorType::STORAGE_IMAGE,
            data,
            self.storage_offset + (range.base + slot) as u64 * self.sizes.storage_image as u64,
            self.sizes.storage_image,
            self.resources_cpu,
        );
        Texture2dRW(Handle((range.base + slot) as u64))
    }

    pub fn write_sampler(&self, range: HeapRange, slot: u32, sampler: &SamplerState) -> Sampler {
        debug_assert!(slot < range.len);
        let data = vk::DescriptorDataEXT {
            p_sampler: &sampler.0,
        };
        self.write(
            vk::DescriptorType::SAMPLER,
            data,
            (range.base + slot) as u64 * self.sizes.sampler as u64,
            self.sizes.sampler,
            self.samplers_cpu,
        );
        Sampler(Handle((range.base + slot) as u64))
    }

    fn write(
        &self,
        kind: vk::DescriptorType,
        data: vk::DescriptorDataEXT,
        offset: u64,
        size: usize,
        base: NonNull<u8>,
    ) {
        let info = vk::DescriptorGetInfoEXT::default().ty(kind);
        let info = vk::DescriptorGetInfoEXT { data, ..info };
        let destination =
            unsafe { std::slice::from_raw_parts_mut(base.as_ptr().add(offset as usize), size) };
        unsafe { self.ext.get_descriptor(&info, destination) };
    }

    /// Metal binds the heap as one more `u64` in the argument table. Vulkan needs the buffers
    /// bound and then a per-set offset, which is why `Push` carries no heap address at all.
    pub(crate) fn bind(
        &self,
        command_buffer: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        graphics: bool,
    ) {
        let bindings = [
            vk::DescriptorBufferBindingInfoEXT::default()
                .address(self.resources_address)
                .usage(
                    vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                ),
            vk::DescriptorBufferBindingInfoEXT::default()
                .address(self.samplers_address)
                .usage(
                    vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                ),
        ];
        let point = match graphics {
            true => vk::PipelineBindPoint::GRAPHICS,
            false => vk::PipelineBindPoint::COMPUTE,
        };
        unsafe {
            self.ext
                .cmd_bind_descriptor_buffers(command_buffer, &bindings);
            self.ext.cmd_set_descriptor_buffer_offsets(
                command_buffer,
                point,
                layout,
                0,
                &[0, 0, 1],
                &[self.sampled_offset, self.storage_offset, 0],
            );
        }
    }
}

impl Drop for ResourceHeap {
    fn drop(&mut self) {
        unsafe {
            self.device.unmap_memory(self.resources_memory);
            self.device.destroy_buffer(self.resources, None);
            self.device.free_memory(self.resources_memory, None);
            self.device.unmap_memory(self.samplers_memory);
            self.device.destroy_buffer(self.samplers, None);
            self.device.free_memory(self.samplers_memory, None);
        }
    }
}

pub struct Surface {
    pub(crate) surface_khr: vk::SurfaceKHR,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) images: Vec<vk::Image>,
    pub(crate) views: Vec<vk::ImageView>,
    pub(crate) format: vk::Format,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) acquired: Vec<vk::Semaphore>,
    pub(crate) released: Vec<vk::Semaphore>,
    pub(crate) cursor: usize,
}

impl Surface {
    pub fn new(device: &Device, hwnd: isize, width: u32, height: u32) -> Result<Self, Error> {
        let surface = unsafe {
            device.win32_surface_ext.create_win32_surface(
                &vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(0)
                    .hwnd(hwnd),
                None,
            )?
        };
        let supported = unsafe {
            device.surface_ext.get_physical_device_surface_support(
                device.physical,
                device.queue_family,
                surface,
            )?
        };
        if !supported {
            return Err(Error::new("queue family cannot present to this surface"));
        }

        let formats = unsafe {
            device
                .surface_ext
                .get_physical_device_surface_formats(device.physical, surface)?
        };
        let format = formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
            .copied()
            .unwrap_or(formats[0]);

        let mut swapchain = Self {
            surface_khr: surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            views: Vec::new(),
            format: format.format,
            width,
            height,
            acquired: Vec::new(),
            released: Vec::new(),
            cursor: 0,
        };
        swapchain.rebuild(device, width, height)?;
        Ok(swapchain)
    }

    pub fn rebuild(&mut self, device: &Device, width: u32, height: u32) -> Result<(), Error> {
        unsafe { device.device.device_wait_idle()? };
        let caps = unsafe {
            device
                .surface_ext
                .get_physical_device_surface_capabilities(device.physical, self.surface_khr)?
        };
        let extent = match caps.current_extent.width {
            u32::MAX => vk::Extent2D { width, height },
            _ => caps.current_extent,
        };
        if extent.width == 0 || extent.height == 0 {
            return Ok(());
        }
        let count = match caps.max_image_count {
            0 => caps.min_image_count + 1,
            max => (caps.min_image_count + 1).min(max),
        };

        let old = self.swapchain;
        self.swapchain = unsafe {
            device.swapchain_ext.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface_khr)
                    .min_image_count(count)
                    .image_format(self.format)
                    .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                    )
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(vk::PresentModeKHR::FIFO)
                    .clipped(true)
                    .old_swapchain(old),
                None,
            )?
        };
        self.destroy_views(device);
        if old != vk::SwapchainKHR::null() {
            unsafe { device.swapchain_ext.destroy_swapchain(old, None) };
        }

        self.width = extent.width;
        self.height = extent.height;
        self.images = unsafe { device.swapchain_ext.get_swapchain_images(self.swapchain)? };
        for &image in &self.images {
            let view = unsafe {
                device.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )?
            };
            self.views.push(view);
            self.acquired.push(unsafe {
                device
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
            });
            self.released.push(unsafe {
                device
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
            });
        }
        Ok(())
    }

    fn destroy_views(&mut self, device: &Device) {
        unsafe {
            for &view in &self.views {
                device.device.destroy_image_view(view, None);
            }
            for &semaphore in self.acquired.iter().chain(&self.released) {
                device.device.destroy_semaphore(semaphore, None);
            }
        }
        self.views.clear();
        self.acquired.clear();
        self.released.clear();
    }

    /// Metal hands back a `Drawable` the queue waits on. Vulkan needs an explicit binary
    /// semaphore pair, so `Frame::submit` takes them rather than a drawable.
    pub fn acquire(&mut self, device: &Device) -> Option<Drawable> {
        let semaphore = self.acquired[self.cursor];
        let result = unsafe {
            device.swapchain_ext.acquire_next_image(
                self.swapchain,
                u64::MAX,
                semaphore,
                vk::Fence::null(),
            )
        };
        let (index, suboptimal) = match result {
            Ok(pair) => pair,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return None,
            Err(_) => return None,
        };
        self.cursor = (self.cursor + 1) % self.acquired.len();
        Some(Drawable {
            index,
            image: self.images[index as usize],
            view: self.views[index as usize],
            acquired: semaphore,
            released: self.released[index as usize],
            suboptimal,
        })
    }

    /// The swapchain's colour format, for building a matching [`ColorTarget`].
    pub fn format(&self) -> Format {
        from_vk_format(self.format).expect("swapchain format is outside nga::Format")
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Presenting used to be a raw `queue_present` at the call site, which is exactly the kind
    /// of leak this crate exists to close. Returns `true` when the surface needs rebuilding.
    pub fn present(&mut self, device: &Device, drawable: &Drawable) -> bool {
        let indices = [drawable.index];
        let swapchains = [self.swapchain];
        let waits = [drawable.released];
        let result = unsafe {
            device.swapchain_ext.queue_present(
                device.queue(),
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&waits)
                    .swapchains(&swapchains)
                    .image_indices(&indices),
            )
        };
        matches!(result, Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Ok(true)) || drawable.suboptimal
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            self.destroy_views(device);
            device.swapchain_ext.destroy_swapchain(self.swapchain, None);
            device.surface_ext.destroy_surface(self.surface_khr, None);
        }
    }
}

#[derive(Clone, Copy)]
pub struct Drawable {
    pub(crate) index: u32,
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) acquired: vk::Semaphore,
    pub(crate) released: vk::Semaphore,
    pub(crate) suboptimal: bool,
}


/// One SPIR-V module holds every entry point, the way one `MTLLibrary` holds every function,
/// so pipelines name their stages by string exactly as they do under Metal.
pub struct Shaders {
    pub(crate) module: vk::ShaderModule,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) device: ash::Device,
    pub(crate) pipelines: Vec<vk::Pipeline>,
}

impl Shaders {
    pub fn new(device: &Device, spirv: &[u8]) -> Result<Self, Error> {
        debug_assert!(
            spirv.len() % 4 == 0,
            "SPIR-V is not a whole number of words"
        );
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        let module = unsafe {
            device
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
        };
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .size(PUSH_SIZE)];
        let layout = unsafe {
            device.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&device.set_layouts)
                    .push_constant_ranges(&ranges),
                None,
            )?
        };
        Ok(Self {
            module,
            layout,
            device: device.device.clone(),
            pipelines: Vec::new(),
        })
    }

    pub fn raster(&mut self, desc: &RasterDesc) -> Result<Pipeline, Error> {
        let vertex = std::ffi::CString::new(desc.vertex).unwrap();
        let fragment = std::ffi::CString::new(desc.fragment).unwrap();
        let specialization = desc.constants.map(|c| specialize(&c.values()));
        let info = specialization.as_ref().map(|(entries, data)| {
            vk::SpecializationInfo::default()
                .map_entries(entries)
                .data(data)
        });

        let mut stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.module)
                .name(&vertex),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.module)
                .name(&fragment),
        ];
        if let Some(info) = info.as_ref() {
            stages[0] = stages[0].specialization_info(info);
            stages[1] = stages[1].specialization_info(info);
        }

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology(desc.topology));
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();

        let attachments: Vec<_> = desc.colors.iter().map(blend_attachment).collect();
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);

        // Depth and stencil stay dynamic so `depth_state()` remains a state object bound at
        // record time, the way `MTLDepthStencilState` is, instead of multiplying pipelines.
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::CULL_MODE,
            vk::DynamicState::FRONT_FACE,
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::STENCIL_REFERENCE,
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let formats: Vec<vk::Format> = desc.colors.iter().map(|t| format(t.format)).collect();
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&formats)
            .depth_attachment_format(desc.depth.map_or(vk::Format::UNDEFINED, format));

        let create = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .layout(self.layout)
            .push(&mut rendering);

        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create], None)
                .map_err(|(_, result)| Error::from(result))?
        };
        self.pipelines.push(pipelines[0]);
        Ok(Pipeline(pipelines[0]))
    }

    pub fn compute(
        &mut self,
        entry: &str,
        constants: Option<&dyn Constants>,
    ) -> Result<Pipeline, Error> {
        let entry = std::ffi::CString::new(entry).unwrap();
        let specialization = constants.map(|c| specialize(&c.values()));
        let info = specialization.as_ref().map(|(entries, data)| {
            vk::SpecializationInfo::default()
                .map_entries(entries)
                .data(data)
        });
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.module)
            .name(&entry);
        if let Some(info) = info.as_ref() {
            stage = stage.specialization_info(info);
        }
        let create = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .layout(self.layout);
        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create], None)
                .map_err(|(_, result)| Error::from(result))?
        };
        self.pipelines.push(pipelines[0]);
        Ok(Pipeline(pipelines[0]))
    }
}

impl Drop for Shaders {
    fn drop(&mut self) {
        unsafe {
            for &pipeline in &self.pipelines {
                self.device.destroy_pipeline(pipeline, None);
            }
            self.device.destroy_shader_module(self.module, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Attachment {
    pub(crate) view: vk::ImageView,
    pub(crate) image: vk::Image,
    pub(crate) clear: Option<vk::ClearValue>,
    pub(crate) store: bool,
}

#[derive(Default)]
pub struct Pass {
    pub(crate) colors: Vec<Attachment>,
    pub(crate) depth: Option<Attachment>,
    pub(crate) layers: u32,
}

impl Pass {
    pub fn new() -> Self {
        Self {
            colors: Vec::new(),
            depth: None,
            layers: 1,
        }
    }

    pub fn color(mut self, index: usize, drawable: &Drawable, clear: Option<[f32; 4]>) -> Self {
        let attachment = Attachment {
            view: drawable.view,
            image: drawable.image,
            clear: clear.map(|float32| vk::ClearValue {
                color: vk::ClearColorValue { float32 },
            }),
            store: true,
        };
        if self.colors.len() <= index {
            self.colors.resize(
                index + 1,
                Attachment {
                    view: vk::ImageView::null(),
                    image: vk::Image::null(),
                    clear: None,
                    store: false,
                },
            );
        }
        self.colors[index] = attachment;
        self
    }

    /// Renders into a texture rather than the surface. The two differ only in where the image
    /// came from, so offscreen rendering needs no separate pass type.
    ///
    /// Recording the pass is what moves the image to `COLOR_ATTACHMENT_OPTIMAL`, so the
    /// texture's tracked layout is updated here. Without this a later `read_texture` would
    /// transition it out of `UNDEFINED` and discard everything the pass just drew.
    pub fn color_texture(
        mut self,
        index: usize,
        texture: &HeapTexture,
        clear: Option<[f32; 4]>,
    ) -> Self {
        texture.layout.set(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let attachment = Attachment {
            view: texture.view,
            image: texture.image,
            clear: clear.map(|float32| vk::ClearValue {
                color: vk::ClearColorValue { float32 },
            }),
            store: true,
        };
        if self.colors.len() <= index {
            self.colors.resize(
                index + 1,
                Attachment {
                    view: vk::ImageView::null(),
                    image: vk::Image::null(),
                    clear: None,
                    store: false,
                },
            );
        }
        self.colors[index] = attachment;
        self
    }

    pub fn depth(mut self, texture: &HeapTexture, clear: Option<f32>, store: bool) -> Self {
        texture
            .layout
            .set(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
        self.depth = Some(Attachment {
            view: texture.view,
            image: texture.image,
            clear: clear.map(|depth| vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
            }),
            store,
        });
        self
    }
}

pub struct Raster<'a> {
    pub(crate) device: &'a ash::Device,
    pub(crate) commands: &'a Commands,
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) push: Cell<Push>,
}

impl<'a> Raster<'a> {
    pub(crate) fn new(
        device: &'a ash::Device,
        commands: &'a Commands,
        command_buffer: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        pass: &Pass,
        width: u32,
        height: u32,
    ) -> Self {
        // Vulkan has no `MTLStorageMode`-style implicit transition: every attachment has to be
        // moved to its rendering layout by hand. `VK_KHR_unified_image_layouts` would delete
        // this block entirely, and it is absent on this driver.
        let mut barriers: Vec<_> = pass
            .colors
            .iter()
            .filter(|attachment| attachment.image != vk::Image::null())
            .map(|attachment| {
                // A pass that clears may discard whatever was there; a pass that loads must
                // keep it, and must additionally order its writes after the previous pass's.
                // Metal's `Pass` needs neither distinction — this is the layout tax again.
                let (old_layout, src_stage, src_access) = match attachment.clear {
                    Some(_) => (
                        vk::ImageLayout::UNDEFINED,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                    ),
                    None => (
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    ),
                };
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                    )
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(attachment.image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    )
            })
            .collect();
        if let Some(attachment) = pass.depth {
            let old_layout = match attachment.clear {
                Some(_) => vk::ImageLayout::UNDEFINED,
                None => vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            };
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                    .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                    .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .image(attachment.image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .level_count(1)
                            .layer_count(1),
                    ),
            );
        }
        unsafe {
            device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            )
        };

        let colors: Vec<_> = pass
            .colors
            .iter()
            .map(|attachment| {
                let mut info = vk::RenderingAttachmentInfo::default()
                    .image_view(attachment.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .store_op(match attachment.store {
                        true => vk::AttachmentStoreOp::STORE,
                        false => vk::AttachmentStoreOp::DONT_CARE,
                    });
                info = match attachment.clear {
                    Some(clear) => info.load_op(vk::AttachmentLoadOp::CLEAR).clear_value(clear),
                    None => info.load_op(vk::AttachmentLoadOp::LOAD),
                };
                info
            })
            .collect();

        let area = vk::Rect2D::default().extent(vk::Extent2D { width, height });
        let mut rendering = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(pass.layers)
            .color_attachments(&colors);
        let depth;
        if let Some(attachment) = pass.depth {
            depth = vk::RenderingAttachmentInfo::default()
                .image_view(attachment.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(match attachment.clear {
                    Some(_) => vk::AttachmentLoadOp::CLEAR,
                    None => vk::AttachmentLoadOp::LOAD,
                })
                .store_op(match attachment.store {
                    true => vk::AttachmentStoreOp::STORE,
                    false => vk::AttachmentStoreOp::DONT_CARE,
                })
                .clear_value(attachment.clear.unwrap_or_default());
            rendering = rendering.depth_attachment(&depth);
        }
        unsafe { device.cmd_begin_rendering(command_buffer, &rendering) };

        let raster = Self {
            device,
            commands,
            command_buffer,
            layout,
            push: Cell::new(Push::default()),
        };
        raster.viewport(
            Viewport::new(width, height),
            Scissor {
                x: 0,
                y: 0,
                width,
                height,
            },
        );
        raster.cull(Cull::None, Winding::CounterClockwise);
        raster.depth(&DepthState::new(CompareOp::Always, false));
        unsafe {
            device.cmd_set_stencil_test_enable(command_buffer, false);
            device.cmd_set_stencil_reference(
                command_buffer,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                0,
            );
            device.cmd_set_stencil_op(
                command_buffer,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                vk::StencilOp::KEEP,
                vk::StencilOp::KEEP,
                vk::StencilOp::KEEP,
                vk::CompareOp::ALWAYS,
            );
        }
        raster
    }

    pub fn pipeline(&self, pipeline: Pipeline) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.0,
            )
        };
    }

    pub fn heap(&self, heap: &ResourceHeap) {
        heap.bind(self.command_buffer, self.layout, true);
    }

    pub fn depth(&self, state: &DepthState) {
        unsafe {
            self.device
                .cmd_set_depth_test_enable(self.command_buffer, state.test());
            self.device
                .cmd_set_depth_write_enable(self.command_buffer, state.write);
            self.device
                .cmd_set_depth_compare_op(self.command_buffer, compare_op(state.compare));
        }
    }

    pub fn cull(&self, mode: Cull, front: Winding) {
        unsafe {
            self.device
                .cmd_set_cull_mode(self.command_buffer, cull(mode));
            self.device
                .cmd_set_front_face(self.command_buffer, winding(front));
        }
    }

    pub fn viewport(&self, view: Viewport, clip: Scissor) {
        unsafe {
            self.device
                .cmd_set_viewport(self.command_buffer, 0, &[viewport(view)]);
            self.device
                .cmd_set_scissor(self.command_buffer, 0, &[scissor(clip)]);
        }
    }

    pub fn draw_indexed<T, I: IndexFormat>(
        &self,
        root: Gpu<'_, T>,
        indices: &HeapBuffer,
        count: u32,
    ) {
        self.bind_indices::<I>(indices);
        self.bind(root);
        unsafe {
            self.device
                .cmd_draw_indexed(self.command_buffer, count, 1, 0, 0, 0)
        };
    }

    /// `vkCmdBindIndexBuffer2` takes the range explicitly, so an out-of-bounds index faults
    /// against the allocation rather than against whatever the heap suballocated after it.
    fn bind_indices<I: IndexFormat>(&self, indices: &HeapBuffer) {
        unsafe {
            self.device.cmd_bind_index_buffer2(
                self.command_buffer,
                indices.buffer,
                0,
                indices.len as u64,
                index_type::<I>(),
            )
        };
    }

    /// One driver call for a run of draws that share every piece of state, including the root
    /// pointer. This is the CPU-side counterpart to `draw_indirect_count`: the ranges are known
    /// at record time rather than computed on the GPU, but the per-draw cost is a `[u32; 2]`
    /// write instead of a command-buffer round trip. `Device::max_multi_draw` bounds `draws`.
    pub fn draw_multi<T>(&self, root: Gpu<'_, T>, draws: &[MultiDraw]) {
        self.bind(root);
        unsafe {
            (self.commands.multi_draw.fp().cmd_draw_multi_ext)(
                self.command_buffer,
                draws.len() as u32,
                draws.as_ptr() as *const vk::MultiDrawInfoEXT,
                1,
                0,
                size_of::<vk::MultiDrawInfoEXT>() as u32,
            )
        };
    }

    pub fn draw_multi_indexed<T, I: IndexFormat>(
        &self,
        root: Gpu<'_, T>,
        indices: &HeapBuffer,
        draws: &[MultiDrawIndexed],
    ) {
        self.bind_indices::<I>(indices);
        self.bind(root);
        unsafe {
            (self.commands.multi_draw.fp().cmd_draw_multi_indexed_ext)(
                self.command_buffer,
                draws.len() as u32,
                draws.as_ptr() as *const vk::MultiDrawIndexedInfoEXT,
                1,
                0,
                size_of::<vk::MultiDrawIndexedInfoEXT>() as u32,
                std::ptr::null(),
            )
        };
    }

    fn bind<T>(&self, root: Gpu<'_, T>) {
        let mut push = self.push.get();
        push.root = root.0.get();
        self.push.set(push);
        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                self.layout,
                vk::ShaderStageFlags::ALL,
                0,
                std::slice::from_raw_parts(&push as *const Push as *const u8, PUSH_SIZE as usize),
            )
        };
    }

    /// Metal replays an indirect command buffer the GPU filled in, with each encoded draw
    /// carrying its own root pointer. Vulkan has no GPU-side encoder: the closest portable
    /// equivalent is a GPU-written `VkDrawIndirectCommand` array plus a GPU-written count,
    /// with every draw sharing one push constant. Per-draw data has to travel some other way
    /// — here, through `first_instance`.
    pub fn draw_indirect_count<T>(
        &self,
        root: Gpu<'_, T>,
        draws: &HeapBuffer,
        count: &HeapBuffer,
        max: u32,
    ) {
        self.bind(root);
        unsafe {
            self.device.cmd_draw_indirect_count(
                self.command_buffer,
                draws.buffer,
                0,
                count.buffer,
                0,
                max,
                size_of::<[u32; 4]>() as u32,
            )
        };
    }

    pub fn draw<T>(&self, root: Gpu<'_, T>, start: u32, count: u32) {
        self.bind(root);
        unsafe {
            self.device
                .cmd_draw(self.command_buffer, count, 1, start, 0)
        };
    }
}

impl Drop for Raster<'_> {
    fn drop(&mut self) {
        unsafe { self.device.cmd_end_rendering(self.command_buffer) };
    }
}

pub struct Compute<'a> {
    pub(crate) device: &'a ash::Device,
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) push: Cell<Push>,
}

impl<'a> Compute<'a> {
    pub(crate) fn new(
        device: &'a ash::Device,
        command_buffer: vk::CommandBuffer,
        layout: vk::PipelineLayout,
    ) -> Self {
        Self {
            device,
            command_buffer,
            layout,
            push: Cell::new(Push::default()),
        }
    }

    pub fn pipeline(&self, pipeline: Pipeline) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.0,
            )
        };
    }

    pub fn heap(&self, heap: &ResourceHeap) {
        heap.bind(self.command_buffer, self.layout, false);
    }

    pub fn dispatch<T>(&self, root: Gpu<'_, T>, threads: usize, group: usize) {
        let mut push = self.push.get();
        push.root = root.0.get();
        self.push.set(push);
        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                self.layout,
                vk::ShaderStageFlags::ALL,
                0,
                std::slice::from_raw_parts(&push as *const Push as *const u8, PUSH_SIZE as usize),
            );
            self.device.cmd_dispatch(
                self.command_buffer,
                ((threads + group - 1) / group) as u32,
                1,
                1,
            );
        }
    }

    pub fn dispatch_2d<T>(&self, root: Gpu<'_, T>, width: u32, height: u32, group: u32) {
        let mut push = self.push.get();
        push.root = root.0.get();
        self.push.set(push);
        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                self.layout,
                vk::ShaderStageFlags::ALL,
                0,
                std::slice::from_raw_parts(&push as *const Push as *const u8, PUSH_SIZE as usize),
            );
            self.device.cmd_dispatch(
                self.command_buffer,
                width.div_ceil(group),
                height.div_ceil(group),
                1,
            );
        }
    }

    /// The whole barrier, with no resource list — exactly the design the post argues for, and
    /// exactly what `VkMemoryBarrier2` provides. Buffers need nothing else; only images drag
    /// per-resource layout state back in.
    pub fn barrier(&self, before: Stage, after: Stage, hazards: Hazard) {
        let (src_access, dst_access) = hazard(hazards);
        let barriers = [vk::MemoryBarrier2::default()
            .src_stage_mask(stage(before))
            .src_access_mask(src_access)
            .dst_stage_mask(stage(after))
            .dst_access_mask(dst_access)];
        unsafe {
            self.device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().memory_barriers(&barriers),
            )
        };
    }

    pub fn copy_buffer(
        &self,
        source: &HeapBuffer,
        source_offset: usize,
        destination: &HeapBuffer,
        destination_offset: usize,
        size: usize,
    ) {
        unsafe {
            self.device.cmd_copy_buffer(
                self.command_buffer,
                source.buffer,
                destination.buffer,
                &[vk::BufferCopy {
                    src_offset: source_offset as u64,
                    dst_offset: destination_offset as u64,
                    size: size as u64,
                }],
            )
        };
    }

    /// Moves a texture into the layout a shader or a copy needs. Metal tracks this itself;
    /// Vulkan does not, so it is the one piece of per-resource state the API cannot delete.
    pub fn transition(&self, texture: &HeapTexture, usage: Usage) {
        let layout = image_layout(usage);
        let old = texture.layout.get();
        if old == layout {
            return;
        }
        let barriers = [image_barrier(texture, old, layout)];
        unsafe {
            self.device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            )
        };
        texture.layout.set(layout);
    }

    /// Reads a whole texture back into host-visible memory, tightly packed. The counterpart of
    /// `GpuHeap::upload`, and what makes a headless render checkable without a screenshot.
    ///
    /// `dest` must live in `arena` — a `Memory::Readback` one, if the CPU is going to read it
    /// more than once. The device address is resolved back to a buffer offset here rather than
    /// carried around, because that is the only thing Vulkan's copy needs that Metal's does not.
    pub fn read_texture(&self, texture: &HeapTexture, arena: &Arena, dest: GpuMut<'_, [u8]>) {
        let offset = dest.0.get().checked_sub(arena.base).expect(
            "the readback destination is not in this arena",
        );
        self.transition(texture, Usage::COPY_SRC);
        let region = vk::BufferImageCopy2::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: texture.width,
                height: texture.height,
                depth: 1,
            })
            .buffer_offset(offset);
        let regions = [region];
        unsafe {
            self.device.cmd_copy_image_to_buffer2(
                self.command_buffer,
                &vk::CopyImageToBufferInfo2::default()
                    .src_image(texture.image)
                    .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .dst_buffer(arena.buffer)
                    .regions(&regions),
            )
        };
    }

    pub fn fill_buffer(&self, buffer: &HeapBuffer, offset: usize, size: usize, value: u32) {
        unsafe {
            self.device.cmd_fill_buffer(
                self.command_buffer,
                buffer.buffer,
                offset as u64,
                size as u64,
                value,
            )
        };
    }
}

pub struct Context {
    pub(crate) device: ash::Device,
    pub(crate) commands: Commands,
    pub(crate) queue: vk::Queue,
    pub(crate) pools: Vec<vk::CommandPool>,
    pub(crate) command_buffers: Vec<vk::CommandBuffer>,
    pub(crate) arenas: Vec<Arena>,
    pub(crate) timeline: vk::Semaphore,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) threads: usize,
    pub(crate) frame: u64,
}

impl Context {
    pub fn new(
        device: &Device,
        shaders: &Shaders,
        arena_size: usize,
        threads: usize,
    ) -> Result<Self, Error> {
        Self::with_layout(device, shaders.layout, arena_size, threads)
    }

    /// Every pipeline layout in `nga-vk` is the same shape, so a `Context` only needs one —
    /// which is what lets the shader-object path reuse all of this unchanged.
    pub(crate) fn with_layout(
        device: &Device,
        layout: vk::PipelineLayout,
        arena_size: usize,
        threads: usize,
    ) -> Result<Self, Error> {
        debug_assert!(threads > 0);
        let slots = FRAMES_IN_FLIGHT as usize * threads;

        // Arenas are BAR-resident, and there are `FRAMES_IN_FLIGHT * threads` of them. Without
        // ReBAR that window is 214 MiB *shared with every other process*, so the interesting
        // number is the live budget, not the heap size. Failing here beats an
        // ERROR_OUT_OF_DEVICE_MEMORY several arenas into the loop with no hint as to why.
        let total = arena_size * slots;
        let probe = vk::BufferCreateInfo::default()
            .size(arena_size.max(1) as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let mut requirements = vk::MemoryRequirements2::default();
        unsafe {
            device.device.get_device_buffer_memory_requirements(
                &vk::DeviceBufferMemoryRequirements::default().create_info(&probe),
                &mut requirements,
            )
        };
        let bar = device.memory_types(
            requirements.memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        );
        if let Some(&index) = bar.first() {
            let free = device.budget(device.heap_of(index));
            if total as u64 > free {
                return Err(Error::new(format!(
                    "{slots} arenas of {arena_size} bytes need {total} bytes of the host-visible \
                     device-local heap, which has {free} free. Shrink the arena, lower the thread \
                     count, or move bulk data to a GpuHeap."
                )));
            }
        }

        let mut pools = Vec::new();
        let mut command_buffers = Vec::new();
        let mut arenas = Vec::new();
        for _ in 0..slots {
            let pool = unsafe {
                device.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(device.queue_family),
                    None,
                )?
            };
            let buffers = unsafe {
                device.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?
            };
            pools.push(pool);
            command_buffers.push(buffers[0]);
            arenas.push(Arena::new(device, arena_size, Memory::Device)?);
        }

        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline = unsafe {
            device
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push(&mut kind), None)?
        };

        Ok(Self {
            device: device.device.clone(),
            commands: device.commands.clone(),
            queue: device.queue(),
            pools,
            command_buffers,
            arenas,
            timeline,
            layout,
            threads,
            frame: 0,
        })
    }

    /// How many frames have been submitted. Callers use it to pick a per-frame slot; the
    /// context uses it to decide which arena and command pool are free.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn begin_frame(&mut self) -> Result<Frame<'_>, Error> {
        let index = self.frame;
        self.frame += 1;
        if index >= FRAMES_IN_FLIGHT {
            let value = [index + 1 - FRAMES_IN_FLIGHT];
            let semaphores = [self.timeline];
            unsafe {
                self.device.wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&semaphores)
                        .values(&value),
                    u64::MAX,
                )?
            };
        }

        let base = (index % FRAMES_IN_FLIGHT) as usize * self.threads;
        let slots = base..base + self.threads;
        for slot in slots.clone() {
            unsafe {
                self.device
                    .reset_command_pool(self.pools[slot], vk::CommandPoolResetFlags::empty())?
            };
            self.arenas[slot].reset();
        }

        Ok(Frame {
            device: &self.device,
            commands: &self.commands,
            queue: self.queue,
            arenas: &self.arenas[slots.clone()],
            command_buffers: &self.command_buffers[slots],
            timeline: self.timeline,
            layout: self.layout,
            recording: 0,
            value: index + 1,
            upload: None,
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_semaphore(self.timeline, None);
            for &pool in &self.pools {
                self.device.destroy_command_pool(pool, None);
            }
        }
    }
}

pub struct Frame<'a> {
    pub(crate) device: &'a ash::Device,
    pub(crate) commands: &'a Commands,
    pub(crate) queue: vk::Queue,
    pub(crate) arenas: &'a [Arena],
    pub(crate) command_buffers: &'a [vk::CommandBuffer],
    pub(crate) timeline: vk::Semaphore,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) recording: usize,
    pub(crate) value: u64,
    pub(crate) upload: Option<(vk::Semaphore, u64)>,
}

impl<'a> Frame<'a> {
    /// This frame's per-thread arena. Everything allocated here is valid until the frame is
    /// submitted and its slot comes round again, which is what the `Gpu<'a, T>` lifetime says.
    ///
    /// The returned borrow is tied to the *context*, not to `self`, so allocating from it does
    /// not lock the frame — recording can still take `&mut self` afterwards.
    pub fn arena(&self, thread: usize) -> &'a Arena {
        &self.arenas[thread]
    }
}

impl Frame<'_> {
    /// One recorder per thread, each with its own `VkCommandPool`-backed command buffer and
    /// its own arena slot. `nga` takes a count per queue; with one queue this is `&[n]`.
    pub fn recorders(&mut self, counts: &[usize]) -> Vec<Recorder<'_>> {
        debug_assert!(
            self.recording == 0,
            "recorders already taken for this frame"
        );
        let count: usize = counts.iter().sum();
        debug_assert!(
            count <= self.command_buffers.len(),
            "{count} recorders requested, Context::new was given {}",
            self.command_buffers.len()
        );
        self.recording = count;
        (0..count)
            .map(|index| {
                let command_buffer = self.command_buffers[index];
                unsafe {
                    self.device
                        .begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo::default()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        )
                        .unwrap()
                };
                Recorder {
                    device: self.device,
                    commands: self.commands,
                    arena: &self.arenas[index],
                    command_buffer,
                    layout: self.layout,
                    not_sync: PhantomData,
                }
            })
            .collect()
    }

    pub fn recorder(&mut self) -> Recorder<'_> {
        let index = self.recording;
        self.recording += 1;
        debug_assert!(index < self.command_buffers.len());
        let command_buffer = self.command_buffers[index];
        unsafe {
            self.device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap()
        };
        Recorder {
            device: self.device,
            commands: self.commands,
            arena: &self.arenas[index],
            command_buffer,
            layout: self.layout,
            not_sync: PhantomData,
        }
    }

    /// Make this frame wait on an upload batch on the GPU rather than on the CPU. Pair with
    /// `GpuHeap::flush_async`: the uploads are in flight while the frame is being recorded, and
    /// nothing blocks. `GpuHeap::flush` is the blocking equivalent, for load time.
    pub fn after_uploads(&mut self, value: u64, heap: &GpuHeap) {
        self.upload = Some((heap.uploader.timeline, value));
    }

    pub fn submit(self, drawable: Option<&Drawable>) -> Result<(), Error> {
        if let Some(drawable) = drawable {
            let command_buffer = self.command_buffers[self.recording - 1];
            unsafe {
                self.device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                            .image(drawable.image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .level_count(1)
                                    .layer_count(1),
                            ),
                    ]),
                )
            };
        }
        for &command_buffer in &self.command_buffers[..self.recording] {
            unsafe { self.device.end_command_buffer(command_buffer)? };
        }

        let command_infos: Vec<_> = self.command_buffers[..self.recording]
            .iter()
            .map(|&command_buffer| {
                vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)
            })
            .collect();

        let mut waits = Vec::new();
        let mut signals = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline)
                .value(self.value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        if let Some((semaphore, value)) = self.upload {
            waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .value(value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        if let Some(drawable) = drawable {
            waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(drawable.acquired)
                    .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            );
            signals.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(drawable.released)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }

        unsafe {
            self.device.queue_submit2(
                self.queue,
                &[vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&waits)
                    .command_buffer_infos(&command_infos)
                    .signal_semaphore_infos(&signals)],
                vk::Fence::null(),
            )?
        };
        Ok(())
    }
}

pub struct Recorder<'a> {
    pub(crate) device: &'a ash::Device,
    pub(crate) commands: &'a Commands,
    pub(crate) arena: &'a Arena,
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) not_sync: PhantomData<Cell<()>>,
}

impl<'a> Recorder<'a> {
    /// This recorder's slice of the frame arena — where root data for the commands being
    /// recorded belongs. Borrowed from the context, not from `self`, so allocating does not
    /// prevent recording.
    pub fn arena(&self) -> &'a Arena {
        self.arena
    }
}

impl Recorder<'_> {
    pub fn compute(&mut self) -> Compute<'_> {
        Compute::new(self.device, self.command_buffer, self.layout)
    }

    pub fn raster(&mut self, pass: &Pass, width: u32, height: u32) -> Raster<'_> {
        Raster::new(
            self.device,
            self.commands,
            self.command_buffer,
            self.layout,
            pass,
            width,
            height,
        )
    }
}
