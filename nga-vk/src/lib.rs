#![allow(non_camel_case_types)]

pub mod abi;
pub mod tlsf;

use ash::vk;
use std::cell::Cell;
use std::ffi::{CStr, c_char};
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;

pub use abi::*;
pub use tlsf::{Alloc, Tlsf};

#[derive(Debug)]
pub struct VkError(pub String);

impl VkError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for VkError {}

impl From<vk::Result> for VkError {
    fn from(result: vk::Result) -> Self {
        Self(format!("{result:?}"))
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Gpu<'a, T: ?Sized>(pub NonZeroU64, pub PhantomData<&'a T>);

impl<T: ?Sized> Gpu<'_, T> {
    pub fn new(address: u64) -> Self {
        Self(NonZeroU64::new(address).unwrap(), PhantomData)
    }
}

impl<'a, T> Gpu<'a, [T]> {
    pub fn elements(self) -> Gpu<'a, T> {
        Gpu(self.0, PhantomData)
    }
}

impl<T: ?Sized> Clone for Gpu<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Gpu<'_, T> {}

#[repr(transparent)]
#[derive(Debug)]
pub struct GpuMut<'a, T: ?Sized>(pub NonZeroU64, pub PhantomData<&'a Cell<T>>);

impl<T: ?Sized> GpuMut<'_, T> {
    pub fn new(address: u64) -> Self {
        Self(NonZeroU64::new(address).unwrap(), PhantomData)
    }
}

impl<'a, T> GpuMut<'a, [T]> {
    pub fn elements(self) -> GpuMut<'a, T> {
        GpuMut(self.0, PhantomData)
    }
}

impl<T: ?Sized> Clone for GpuMut<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for GpuMut<'_, T> {}

pub struct GpuBox<'a, T: ?Sized> {
    pub cpu: NonNull<T>,
    pub gpu: Gpu<'a, T>,
}

impl<T: ?Sized> Deref for GpuBox<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { self.cpu.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for GpuBox<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { self.cpu.as_mut() }
    }
}

/// Metal binds two argument-table addresses; Vulkan gets one push-constant range instead.
/// Every pipeline layout in `nga-vk` is this range plus the two descriptor-buffer sets, so
/// pipelines are freely interchangeable the way they are under one `MTL4ArgumentTable`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Push {
    pub root: u64,
    pub resources: u32,
    pub samplers: u32,
}

pub const PUSH_SIZE: u32 = size_of::<Push>() as u32;

pub struct Device {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family: u32,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    pub limits: vk::PhysicalDeviceLimits,
    pub timestamp_period: f32,
    pub name: String,
    pub surface_ext: ash::khr::surface::Instance,
    pub win32_surface_ext: ash::khr::win32_surface::Instance,
    pub swapchain_ext: ash::khr::swapchain::Device,
    pub descriptor_buffer_ext: ash::ext::descriptor_buffer::Device,
    pub host_image_copy_ext: ash::ext::host_image_copy::Device,
    pub descriptor_sizes: DescriptorSizes,
    pub debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

/// `VK_EXT_descriptor_buffer` descriptors are driver-sized, unlike Metal's fixed 8-byte
/// `MTLResourceID`. On this NVIDIA driver a sampled image is 4 bytes, not 8.
#[derive(Clone, Copy, Debug, Default)]
pub struct DescriptorSizes {
    pub sampled_image: usize,
    pub storage_image: usize,
    pub sampler: usize,
    pub offset_alignment: u64,
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
    pub fn new(validation: bool) -> Result<Self, VkError> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| VkError::new(format!("could not load vulkan-1.dll: {e}")))?;

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
            .api_version(vk::API_VERSION_1_3);
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
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
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

        let surface_ext = ash::khr::surface::Instance::new(&entry, &instance);
        let win32_surface_ext = ash::khr::win32_surface::Instance::new(&entry, &instance);

        let devices = unsafe { instance.enumerate_physical_devices()? };
        let (physical, queue_family) = devices
            .iter()
            .find_map(|&physical| {
                let families =
                    unsafe { instance.get_physical_device_queue_family_properties(physical) };
                let family = families.iter().position(|family| {
                    family.queue_flags.contains(
                        vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                    )
                })?;
                Some((physical, family as u32))
            })
            .ok_or(VkError::new("no device with a graphics+compute queue"))?;

        let device_extensions = [
            ash::khr::swapchain::NAME.as_ptr(),
            ash::ext::descriptor_buffer::NAME.as_ptr(),
            ash::ext::host_image_copy::NAME.as_ptr(),
            ash::ext::extended_dynamic_state3::NAME.as_ptr(),
        ];

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .scalar_block_layout(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true);
        let mut features11 =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        let mut descriptor_buffer =
            vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);
        let mut host_image_copy =
            vk::PhysicalDeviceHostImageCopyFeaturesEXT::default().host_image_copy(true);
        let mut dynamic_state3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
            .extended_dynamic_state3_color_blend_enable(true)
            .extended_dynamic_state3_color_blend_equation(true)
            .extended_dynamic_state3_color_write_mask(true);
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
                    .push_next(&mut features11)
                    .push_next(&mut features12)
                    .push_next(&mut features13)
                    .push_next(&mut descriptor_buffer)
                    .push_next(&mut host_image_copy)
                    .push_next(&mut dynamic_state3),
                None,
            )?
        };

        let mut descriptor_props = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut descriptor_props);
        unsafe { instance.get_physical_device_properties2(physical, &mut props) };
        let name = unsafe { CStr::from_ptr(props.properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        Ok(Self {
            queue_family,
            memory: unsafe { instance.get_physical_device_memory_properties(physical) },
            limits: props.properties.limits,
            timestamp_period: props.properties.limits.timestamp_period,
            name,
            descriptor_sizes: DescriptorSizes {
                sampled_image: descriptor_props.sampled_image_descriptor_size,
                storage_image: descriptor_props.storage_image_descriptor_size,
                sampler: descriptor_props.sampler_descriptor_size,
                offset_alignment: descriptor_props.descriptor_buffer_offset_alignment,
            },
            swapchain_ext: ash::khr::swapchain::Device::new(&instance, &device),
            descriptor_buffer_ext: ash::ext::descriptor_buffer::Device::new(&instance, &device),
            host_image_copy_ext: ash::ext::host_image_copy::Device::new(&instance, &device),
            surface_ext,
            win32_surface_ext,
            physical,
            device,
            instance,
            entry,
            debug,
        })
    }

    pub fn queue(&self) -> vk::Queue {
        unsafe { self.device.get_device_queue(self.queue_family, 0) }
    }

    pub fn memory_type(&self, mask: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
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
            self.device.destroy_device(None);
            if let Some((loader, messenger)) = &self.debug {
                loader.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// The Metal `Arena` writes structs straight into GPU memory and hands out the address.
/// On discrete Windows that only works inside the host-visible device-local BAR window,
/// which is 214 MiB on this GPU with no ReBAR — enough for root data, too small for meshes.
pub struct Arena {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub base: u64,
    pub cpu: NonNull<u8>,
    pub size: usize,
    pub head: Cell<usize>,
    pub device: ash::Device,
}

impl Arena {
    pub fn new(device: &Device, size: usize) -> Result<Self, VkError> {
        let usage = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER;
        let info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.device.create_buffer(&info, None)? };
        let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };

        let host_local = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let index = device
            .memory_type(requirements.memory_type_bits, host_local)
            .or_else(|| {
                device.memory_type(
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .ok_or(VkError::new("no host-visible memory type for the arena"))?;

        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = unsafe {
            device.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(index)
                    .push_next(&mut flags),
                None,
            )?
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
            cpu: NonNull::new(cpu as *mut u8).unwrap(),
            size,
            head: Cell::new(0),
            device: device.device.clone(),
        })
    }

    pub fn alloc_bytes(&self, size: usize, align: usize) -> Option<(*mut u8, u64)> {
        let offset = (self.head.get() + align - 1) & !(align - 1);
        if offset + size > self.size {
            return None;
        }
        self.head.set(offset + size);
        Some((
            unsafe { self.cpu.as_ptr().add(offset) },
            self.base + offset as u64,
        ))
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
        self.head.set(0);
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

pub struct Pool<'a, T> {
    pub cpu: NonNull<T>,
    pub gpu: u64,
    pub free: Vec<u32>,
    pub len: u32,
    pub cap: u32,
    pub arena: PhantomData<&'a Arena>,
}

impl<'a, T> Pool<'a, T> {
    pub fn new(arena: &'a Arena, cap: u32) -> Option<Self> {
        let (cpu, gpu) = arena.alloc_bytes(size_of::<T>() * cap as usize, align_of::<T>())?;
        Some(Self {
            cpu: NonNull::new(cpu as *mut T)?,
            gpu,
            free: Vec::new(),
            len: 0,
            cap,
            arena: PhantomData,
        })
    }

    pub fn alloc(&mut self, value: T) -> Option<u32> {
        let index = match self.free.pop() {
            Some(index) => index,
            None if self.len < self.cap => {
                self.len += 1;
                self.len - 1
            }
            None => return None,
        };
        unsafe { self.cpu.as_ptr().add(index as usize).write(value) };
        Some(index)
    }

    pub fn free(&mut self, index: u32) {
        debug_assert!(index < self.len);
        debug_assert!(!self.free.contains(&index));
        self.free.push(index);
    }

    pub fn gpu(&self, index: u32) -> Gpu<'a, T> {
        debug_assert!(index < self.len);
        Gpu::new(self.gpu + (index as usize * size_of::<T>()) as u64)
    }
}

impl<T> Index<u32> for Pool<'_, T> {
    type Output = T;
    fn index(&self, index: u32) -> &T {
        debug_assert!(index < self.len);
        unsafe { &*self.cpu.as_ptr().add(index as usize) }
    }
}

impl<T> IndexMut<u32> for Pool<'_, T> {
    fn index_mut(&mut self, index: u32) -> &mut T {
        debug_assert!(index < self.len);
        unsafe { &mut *self.cpu.as_ptr().add(index as usize) }
    }
}

pub struct Swapchain {
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub acquired: Vec<vk::Semaphore>,
    pub released: Vec<vk::Semaphore>,
    pub cursor: usize,
}

impl Swapchain {
    pub fn new(device: &Device, hwnd: isize, width: u32, height: u32) -> Result<Self, VkError> {
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
            return Err(VkError::new("queue family cannot present to this surface"));
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
            surface,
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

    pub fn rebuild(&mut self, device: &Device, width: u32, height: u32) -> Result<(), VkError> {
        unsafe { device.device.device_wait_idle()? };
        let caps = unsafe {
            device
                .surface_ext
                .get_physical_device_surface_capabilities(device.physical, self.surface)?
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
                    .surface(self.surface)
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

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            self.destroy_views(device);
            device
                .swapchain_ext
                .destroy_swapchain(self.swapchain, None);
            device.surface_ext.destroy_surface(self.surface, None);
        }
    }
}

#[derive(Clone, Copy)]
pub struct Drawable {
    pub index: u32,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub acquired: vk::Semaphore,
    pub released: vk::Semaphore,
    pub suboptimal: bool,
}

pub struct ColorTarget {
    pub format: vk::Format,
    pub blend: Option<vk::PipelineColorBlendAttachmentState>,
}

impl ColorTarget {
    pub fn new(format: vk::Format) -> Self {
        Self {
            format,
            blend: None,
        }
    }
}

pub struct RasterDesc<'a> {
    pub vertex: &'a str,
    pub fragment: &'a str,
    pub colors: &'a [ColorTarget],
    pub depth: vk::Format,
    pub topology: vk::PrimitiveTopology,
    pub constants: Option<&'a dyn Constants>,
}

impl<'a> RasterDesc<'a> {
    pub fn new(vertex: &'a str, fragment: &'a str, colors: &'a [ColorTarget]) -> Self {
        Self {
            vertex,
            fragment,
            colors,
            depth: vk::Format::UNDEFINED,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            constants: None,
        }
    }
}

/// One SPIR-V module holds every entry point, the way one `MTLLibrary` holds every function,
/// so pipelines name their stages by string exactly as they do under Metal.
pub struct Shaders {
    pub module: vk::ShaderModule,
    pub layout: vk::PipelineLayout,
    pub device: ash::Device,
    pub pipelines: Vec<vk::Pipeline>,
}

impl Shaders {
    pub fn new(device: &Device, spirv: &[u8]) -> Result<Self, VkError> {
        debug_assert!(spirv.len() % 4 == 0, "SPIR-V is not a whole number of words");
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
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
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

    pub fn raster(&mut self, desc: &RasterDesc) -> Result<vk::Pipeline, VkError> {
        let vertex = std::ffi::CString::new(desc.vertex).unwrap();
        let fragment = std::ffi::CString::new(desc.fragment).unwrap();
        let specialization = desc.constants.map(|c| c.values());
        let info = specialization.as_ref().map(|s| {
            vk::SpecializationInfo::default()
                .map_entries(&s.entries)
                .data(&s.data)
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
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(desc.topology);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();

        let attachments: Vec<_> = desc
            .colors
            .iter()
            .map(|target| {
                target.blend.unwrap_or(
                    vk::PipelineColorBlendAttachmentState::default()
                        .color_write_mask(vk::ColorComponentFlags::RGBA),
                )
            })
            .collect();
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
        let dynamic =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let formats: Vec<_> = desc.colors.iter().map(|target| target.format).collect();
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&formats)
            .depth_attachment_format(desc.depth);

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
            .layout(self.layout)
            .push_next(&mut rendering);

        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create], None)
                .map_err(|(_, result)| VkError::from(result))?
        };
        self.pipelines.push(pipelines[0]);
        Ok(pipelines[0])
    }

    pub fn compute(
        &mut self,
        entry: &str,
        constants: Option<&dyn Constants>,
    ) -> Result<vk::Pipeline, VkError> {
        let entry = std::ffi::CString::new(entry).unwrap();
        let specialization = constants.map(|c| c.values());
        let info = specialization.as_ref().map(|s| {
            vk::SpecializationInfo::default()
                .map_entries(&s.entries)
                .data(&s.data)
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
            .layout(self.layout);
        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create], None)
                .map_err(|(_, result)| VkError::from(result))?
        };
        self.pipelines.push(pipelines[0]);
        Ok(pipelines[0])
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
pub struct DepthState {
    pub test: bool,
    pub write: bool,
    pub compare: vk::CompareOp,
}

pub fn depth_state(compare: vk::CompareOp, write: bool) -> DepthState {
    DepthState {
        test: compare != vk::CompareOp::ALWAYS || write,
        write,
        compare,
    }
}

#[derive(Clone, Copy)]
pub struct Attachment {
    pub view: vk::ImageView,
    pub image: vk::Image,
    pub clear: Option<vk::ClearValue>,
    pub store: bool,
}

#[derive(Default)]
pub struct Pass {
    pub colors: Vec<Attachment>,
    pub depth: Option<Attachment>,
    pub layers: u32,
}

impl Pass {
    pub fn new() -> Self {
        Self {
            colors: Vec::new(),
            depth: None,
            layers: 1,
        }
    }

    pub fn color(
        mut self,
        index: usize,
        drawable: &Drawable,
        clear: Option<[f32; 4]>,
    ) -> Self {
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
}

pub struct Raster<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub layout: vk::PipelineLayout,
    pub push: Cell<Push>,
}

impl<'a> Raster<'a> {
    pub fn new(
        device: &'a ash::Device,
        command_buffer: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        pass: &Pass,
        width: u32,
        height: u32,
    ) -> Self {
        // Vulkan has no `MTLStorageMode`-style implicit transition: every colour attachment
        // has to be moved to ATTACHMENT_OPTIMAL by hand. `VK_KHR_unified_image_layouts`
        // would delete this block, and it is absent on this driver.
        let barriers: Vec<_> = pass
            .colors
            .iter()
            .filter(|attachment| attachment.image != vk::Image::null())
            .map(|attachment| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
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
                    Some(clear) => info
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .clear_value(clear),
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
            command_buffer,
            layout,
            push: Cell::new(Push::default()),
        };
        raster.viewport(width, height);
        raster.cull(vk::CullModeFlags::NONE, vk::FrontFace::COUNTER_CLOCKWISE);
        raster.depth(&depth_state(vk::CompareOp::ALWAYS, false));
        unsafe {
            device.cmd_set_stencil_test_enable(command_buffer, false);
            device.cmd_set_stencil_reference(command_buffer, vk::StencilFaceFlags::FRONT_AND_BACK, 0);
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

    pub fn pipeline(&self, pipeline: vk::Pipeline) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            )
        };
    }

    pub fn depth(&self, state: &DepthState) {
        unsafe {
            self.device
                .cmd_set_depth_test_enable(self.command_buffer, state.test);
            self.device
                .cmd_set_depth_write_enable(self.command_buffer, state.write);
            self.device
                .cmd_set_depth_compare_op(self.command_buffer, state.compare);
        }
    }

    pub fn cull(&self, mode: vk::CullModeFlags, winding: vk::FrontFace) {
        unsafe {
            self.device.cmd_set_cull_mode(self.command_buffer, mode);
            self.device.cmd_set_front_face(self.command_buffer, winding);
        }
    }

    pub fn viewport(&self, width: u32, height: u32) {
        // Flipped so clip space matches Metal's: +Y up, origin bottom-left.
        let viewport = vk::Viewport {
            x: 0.0,
            y: height as f32,
            width: width as f32,
            height: -(height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });
        unsafe {
            self.device
                .cmd_set_viewport(self.command_buffer, 0, &[viewport]);
            self.device
                .cmd_set_scissor(self.command_buffer, 0, &[scissor]);
        }
    }

    pub fn draw<T>(&self, root: Gpu<'_, T>, start: u32, count: u32) {
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
            self.device
                .cmd_draw(self.command_buffer, count, 1, start, 0);
        }
    }
}

impl Drop for Raster<'_> {
    fn drop(&mut self) {
        unsafe { self.device.cmd_end_rendering(self.command_buffer) };
    }
}

pub const FRAMES_IN_FLIGHT: u64 = 2;

pub struct Context {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub arenas: Vec<Arena>,
    pub timeline: vk::Semaphore,
    pub layout: vk::PipelineLayout,
    pub threads: usize,
    pub frame: u64,
}

impl Context {
    pub fn new(
        device: &Device,
        shaders: &Shaders,
        arena_size: usize,
        threads: usize,
    ) -> Result<Self, VkError> {
        debug_assert!(threads > 0);
        let slots = FRAMES_IN_FLIGHT as usize * threads;
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
            arenas.push(Arena::new(device, arena_size)?);
        }

        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline = unsafe {
            device
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut kind), None)?
        };

        Ok(Self {
            device: device.device.clone(),
            queue: device.queue(),
            pools,
            command_buffers,
            arenas,
            timeline,
            layout: shaders.layout,
            threads,
            frame: 0,
        })
    }

    pub fn begin_frame(&mut self) -> Result<Frame<'_>, VkError> {
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
            queue: self.queue,
            arenas: &self.arenas[slots.clone()],
            command_buffers: &self.command_buffers[slots],
            timeline: self.timeline,
            layout: self.layout,
            recording: 0,
            value: index + 1,
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
    pub device: &'a ash::Device,
    pub queue: vk::Queue,
    pub arenas: &'a [Arena],
    pub command_buffers: &'a [vk::CommandBuffer],
    pub timeline: vk::Semaphore,
    pub layout: vk::PipelineLayout,
    pub recording: usize,
    pub value: u64,
}

impl Frame<'_> {
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
            arena: &self.arenas[index],
            command_buffer,
            layout: self.layout,
            not_sync: PhantomData,
        }
    }

    pub fn submit(self, drawable: Option<&Drawable>) -> Result<(), VkError> {
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
    pub device: &'a ash::Device,
    pub arena: &'a Arena,
    pub command_buffer: vk::CommandBuffer,
    pub layout: vk::PipelineLayout,
    pub not_sync: PhantomData<Cell<()>>,
}

impl Recorder<'_> {
    pub fn raster(&mut self, pass: &Pass, width: u32, height: u32) -> Raster<'_> {
        Raster::new(
            self.device,
            self.command_buffer,
            self.layout,
            pass,
            width,
            height,
        )
    }
}
