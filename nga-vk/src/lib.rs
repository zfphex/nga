#![allow(non_camel_case_types)]

pub mod abi;
pub mod slang;
pub mod tlsf;

use ash::vk;
use std::cell::Cell;
use std::ffi::{CStr, c_char};
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

pub use abi::*;
pub use nga_vk_macros::{Constants, gpu_struct};
pub use slang::SLANG_PRELUDE;
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
    pub shader_object_ext: ash::ext::shader_object::Device,
    pub dynamic_state3_ext: ash::ext::extended_dynamic_state3::Device,
    pub descriptor_sizes: DescriptorSizes,
    /// Sampled images, storage images, samplers. Fixed by `SLANG_PRELUDE`, created once, and
    /// shared by every pipeline layout so pipelines stay interchangeable under one binding
    /// model — the property Metal gets from having a single argument table.
    pub set_layouts: [vk::DescriptorSetLayout; 3],
    pub bindless_cap: u32,
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
                        vk::QueueFlags::GRAPHICS
                            | vk::QueueFlags::COMPUTE
                            | vk::QueueFlags::TRANSFER,
                    )
                })?;
                Some((physical, family as u32))
            })
            .ok_or(VkError::new("no device with a graphics+compute queue"))?;

        let device_extensions = [
            ash::khr::swapchain::NAME.as_ptr(),
            ash::ext::descriptor_buffer::NAME.as_ptr(),
            ash::ext::host_image_copy::NAME.as_ptr(),
            ash::ext::memory_budget::NAME.as_ptr(),
            ash::ext::extended_dynamic_state3::NAME.as_ptr(),
            ash::ext::shader_object::NAME.as_ptr(),
        ];

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
        let mut host_image_copy =
            vk::PhysicalDeviceHostImageCopyFeaturesEXT::default().host_image_copy(true);
        let mut dynamic_state3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
            .extended_dynamic_state3_color_blend_enable(true)
            .extended_dynamic_state3_color_blend_equation(true)
            .extended_dynamic_state3_color_write_mask(true)
            .extended_dynamic_state3_polygon_mode(true)
            .extended_dynamic_state3_rasterization_samples(true)
            .extended_dynamic_state3_sample_mask(true)
            .extended_dynamic_state3_alpha_to_coverage_enable(true)
            .extended_dynamic_state3_logic_op_enable(true);
        let mut shader_object =
            vk::PhysicalDeviceShaderObjectFeaturesEXT::default().shader_object(true);
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
                    .push_next(&mut dynamic_state3)
                    .push_next(&mut shader_object),
                None,
            )?
        };

        let mut descriptor_props = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut descriptor_props);
        unsafe { instance.get_physical_device_properties2(physical, &mut props) };
        let name = unsafe { CStr::from_ptr(props.properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

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
            limits: props.properties.limits,
            timestamp_period: props.properties.limits.timestamp_period,
            name,
            descriptor_sizes: DescriptorSizes {
                sampled_image: descriptor_props.sampled_image_descriptor_size,
                storage_image: descriptor_props.storage_image_descriptor_size,
                sampler: descriptor_props.sampler_descriptor_size,
                offset_alignment: descriptor_props.descriptor_buffer_offset_alignment,
            },
            shader_object_ext: ash::ext::shader_object::Device::new(&instance, &device),
            dynamic_state3_ext: ash::ext::extended_dynamic_state3::Device::new(&instance, &device),
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

    fn set_layout(
        device: &ash::Device,
        kind: vk::DescriptorType,
        count: u32,
    ) -> Result<vk::DescriptorSetLayout, VkError> {
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

    pub fn queue(&self) -> vk::Queue {
        unsafe { self.device.get_device_queue(self.queue_family, 0) }
    }

    /// Memory types matching `flags` and carrying none of `without`, best first. A list
    /// rather than one index, because the heap being full is a different failure from the
    /// heap not existing: the 214 MiB BAR window fills long before it disappears, so the
    /// caller retries the next candidate on `ERROR_OUT_OF_DEVICE_MEMORY`.
    pub fn memory_types(
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
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.physical, &mut properties)
        };
        budget.heap_budget[heap as usize].saturating_sub(budget.heap_usage[heap as usize])
    }

    pub fn heap_of(&self, memory_type: u32) -> u32 {
        self.memory.memory_types[memory_type as usize].heap_index
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
pub enum Placement {
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
    pub placement: Placement,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub base: u64,
    pub cpu: *mut u8,
    pub size: usize,
    pub head: AtomicUsize,
    pub device: ash::Device,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new(device: &Device, size: usize, placement: Placement) -> Result<Self, VkError> {
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
            return Err(VkError::new(format!(
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
                        .push_next(&mut flags),
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
                return Err(VkError::new(format!(
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
            placement,
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

pub struct HeapBuffer {
    pub buffer: vk::Buffer,
    pub gpu: u64,
    pub len: usize,
    pub alloc: Alloc,
}

pub struct HeapTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub alloc: Alloc,
    pub layout: Cell<vk::ImageLayout>,
}

#[derive(Clone, Copy)]
pub struct TextureDesc {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
}

impl TextureDesc {
    pub fn texture_2d(format: vk::Format, width: u32, height: u32) -> Self {
        Self {
            format,
            width,
            height,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
        }
    }

    pub fn depth(format: vk::Format, width: u32, height: u32) -> Self {
        Self {
            format,
            width,
            height,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            aspect: vk::ImageAspectFlags::DEPTH,
        }
    }
}

/// Batches ahead of this many submissions are made to wait, so a command buffer is never
/// re-recorded while the GPU is still reading it.
pub const UPLOAD_BATCHES: usize = 4;

/// Staging ring size. Large enough that a frame's worth of uploads never wraps, small enough
/// that it is unremarkable in system RAM — and it costs no BAR, which is the whole point.
pub const UPLOAD_RING: usize = 32 * 1024 * 1024;

/// `vkCmdUpdateBuffer`'s hard limit. Below it the data rides in the command buffer and the
/// staging ring is not touched at all.
pub const UPDATE_LIMIT: usize = 65536;

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
pub struct Uploader {
    pub ring: Arena,
    pub pool: vk::CommandPool,
    pub buffers: Vec<vk::CommandBuffer>,
    pub timeline: vk::Semaphore,
    pub queue: vk::Queue,
    pub device: ash::Device,
    pub head: usize,
    pub submitted: u64,
    pub recording: bool,
}

impl Uploader {
    pub fn new(device: &Device, ring: usize) -> Result<Self, VkError> {
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
            device.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )?
        };
        Ok(Self {
            ring: Arena::new(device, ring, Placement::Upload)?,
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
    pub fn open(&mut self) -> Result<vk::CommandBuffer, VkError> {
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
    pub fn stage(&mut self, size: usize, align: usize) -> Result<(u64, *mut u8), VkError> {
        if size > self.ring.size {
            return Err(VkError::new(format!(
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
    pub fn flush(&mut self) -> Result<u64, VkError> {
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

    pub fn wait(&self, value: u64) -> Result<(), VkError> {
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
    pub fn flush_wait(&mut self) -> Result<(), VkError> {
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
    pub memory: vk::DeviceMemory,
    pub tlsf: Tlsf,
    pub granularity: u64,
    pub device: ash::Device,
    pub uploader: Uploader,
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

    pub fn new(device: &Device, size: u64) -> Result<Self, VkError> {
        Self::with_ring(device, size, UPLOAD_RING)
    }

    pub fn with_ring(device: &Device, size: u64, ring: usize) -> Result<Self, VkError> {
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
            .ok_or(VkError::new("no device-local memory type for the heap"))?;

        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = unsafe {
            device.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(size)
                    .memory_type_index(index)
                    .push_next(&mut flags),
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

    pub fn alloc_buffer(&mut self, len: usize) -> Result<HeapBuffer, VkError> {
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
            .ok_or(VkError::new("GPU heap is out of space"))?;
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

    pub fn alloc_texture(&mut self, desc: &TextureDesc) -> Result<HeapTexture, VkError> {
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
            .ok_or(VkError::new("GPU heap is out of space"))?;
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
    ) -> Result<(), VkError> {
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
    pub fn upload(&mut self, texture: &HeapTexture, bytes: &[u8]) -> Result<(), VkError> {
        let (width, height) = (texture.width, texture.height);
        let texels = width as usize * height as usize;
        if texels == 0 || bytes.is_empty() {
            return Ok(());
        }
        if !bytes.len().is_multiple_of(texels) {
            return Err(VkError::new(format!(
                "{} bytes is not a whole number of texels for a {width}x{height} image",
                bytes.len()
            )));
        }
        // The texel size comes from the caller's own slice rather than a format table: an upload
        // is by definition the whole image, so the division is exact and every format works.
        let row = bytes.len() / height as usize;
        if row > self.uploader.ring.size {
            return Err(VkError::new(format!(
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
    pub fn flush(&mut self) -> Result<(), VkError> {
        self.uploader.flush_wait()
    }

    /// Submit without blocking, returning the timeline value to wait on. Pair with
    /// `Frame::after_uploads` to let the GPU do the waiting instead of the CPU.
    pub fn flush_async(&mut self) -> Result<u64, VkError> {
        self.uploader.flush()
    }

    /// Metal has no image layouts, so this method has no counterpart at all. Without
    /// `VK_KHR_unified_image_layouts` — absent on this driver — a transition is unavoidable,
    /// but it is only a barrier: recording it into the open upload batch makes it free, where
    /// submitting one command buffer per transition made it the most expensive call here.
    pub fn transition(&mut self, texture: &HeapTexture, layout: vk::ImageLayout) {
        let image = texture.image;
        let old = texture.layout.get();
        let command_buffer = self.uploader.open().expect("upload batch");
        let device = &self.device;
        unsafe {
            device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&[Self::barrier(image, old, layout)]),
            )
        };
        texture.layout.set(layout);
    }

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
    pub base: u32,
    pub len: u32,
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
    pub resources: vk::Buffer,
    pub resources_memory: vk::DeviceMemory,
    pub resources_cpu: NonNull<u8>,
    pub resources_address: u64,
    pub samplers: vk::Buffer,
    pub samplers_memory: vk::DeviceMemory,
    pub samplers_cpu: NonNull<u8>,
    pub samplers_address: u64,
    pub sampled_offset: u64,
    pub storage_offset: u64,
    pub sizes: DescriptorSizes,
    pub cap: u32,
    pub head: u32,
    pub device: ash::Device,
    pub ext: ash::ext::descriptor_buffer::Device,
}

impl ResourceHeap {
    pub fn new(device: &Device, cap: u32) -> Result<Self, VkError> {
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
    ) -> Result<(vk::Buffer, vk::DeviceMemory, NonNull<u8>, u64), VkError> {
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
            .ok_or(VkError::new(
                "no host-visible memory for the descriptor buffer",
            ))?;
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

    pub fn alloc_range(&mut self, len: u32) -> Result<HeapRange, VkError> {
        if self.head + len > self.cap {
            return Err(VkError::new("resource heap is full"));
        }
        let base = self.head;
        self.head += len;
        Ok(HeapRange { base, len })
    }

    /// `vkGetDescriptorEXT` is the direct analogue of reading `texture.gpuResourceID()`: it hands
    /// back the opaque descriptor bytes for the caller to place. The difference is that the size
    /// is a runtime property and the destination is a typed array, not one shared `u64` table.
    pub fn write_texture(&self, range: HeapRange, slot: u32, texture: &HeapTexture) {
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
    }

    /// A texture that is written and sampled in the same frame stays in `GENERAL`, so its
    /// sampled descriptor has to record that layout. Metal has no equivalent choice to make.
    pub fn write_texture_in(
        &self,
        range: HeapRange,
        slot: u32,
        texture: &HeapTexture,
        layout: vk::ImageLayout,
    ) {
        debug_assert!(slot < range.len);
        let image = vk::DescriptorImageInfo::default()
            .image_view(texture.view)
            .image_layout(layout);
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
    }

    pub fn write_storage(&self, range: HeapRange, slot: u32, texture: &HeapTexture) {
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
    }

    pub fn write_sampler(&self, range: HeapRange, slot: u32, sampler: &vk::Sampler) {
        debug_assert!(slot < range.len);
        let data = vk::DescriptorDataEXT { p_sampler: sampler };
        self.write(
            vk::DescriptorType::SAMPLER,
            data,
            (range.base + slot) as u64 * self.sizes.sampler as u64,
            self.sizes.sampler,
            self.samplers_cpu,
        );
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
    pub fn bind(
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
            device.swapchain_ext.destroy_swapchain(self.swapchain, None);
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

#[derive(Clone, Copy, PartialEq)]
pub enum BlendMode {
    None,
    Alpha,
    /// Metal spells this as framebuffer fetch — a fragment shader reading `[[color(0)]]`.
    /// Desktop Vulkan has no equivalent, so the formula has to be expressed as fixed
    /// function. `screen` happens to be `src + dst * (1 - src)`, which it can express;
    /// a formula that genuinely needs the previous value cannot be ported at all.
    Screen,
}

pub struct ColorTarget {
    pub format: vk::Format,
    pub blend: BlendMode,
}

impl ColorTarget {
    pub fn new(format: vk::Format) -> Self {
        Self {
            format,
            blend: BlendMode::None,
        }
    }

    pub fn attachment(&self) -> vk::PipelineColorBlendAttachmentState {
        let state = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .color_blend_op(vk::BlendOp::ADD)
            .alpha_blend_op(vk::BlendOp::ADD);
        match self.blend {
            BlendMode::None => state,
            BlendMode::Alpha => state
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
            BlendMode::Screen => state
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_COLOR)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
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

/// `VK_EXT_shader_object` with no `VkPipeline` anywhere — the closest Vulkan gets to the post's
/// "shader IR in, drawable thing out" ideal, and to Metal 4's model where a pipeline is little
/// more than its functions plus target formats.
///
/// The trade is visible in `bind`: everything a `VkGraphicsPipelineCreateInfo` used to bake in
/// must now be set on the command buffer, every time. Creation gets dramatically simpler;
/// recording gets substantially noisier.
pub struct ShaderObjects {
    pub shaders: [vk::ShaderEXT; 2],
    pub layout: vk::PipelineLayout,
    pub topology: vk::PrimitiveTopology,
    pub device: ash::Device,
    pub ext: ash::ext::shader_object::Device,
    pub dynamic3: ash::ext::extended_dynamic_state3::Device,
}

impl ShaderObjects {
    pub fn raster(
        device: &Device,
        spirv: &[u8],
        vertex: &str,
        fragment: &str,
    ) -> Result<Self, VkError> {
        let vertex = std::ffi::CString::new(vertex).unwrap();
        let fragment = std::ffi::CString::new(fragment).unwrap();
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .size(PUSH_SIZE)];

        // Shader objects take the set layouts and push ranges directly rather than a
        // VkPipelineLayout — but `vkCmdPushConstants` still needs one, so it is kept anyway.
        let infos = [
            vk::ShaderCreateInfoEXT::default()
                .flags(vk::ShaderCreateFlagsEXT::LINK_STAGE)
                .stage(vk::ShaderStageFlags::VERTEX)
                .next_stage(vk::ShaderStageFlags::FRAGMENT)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(spirv)
                .name(&vertex)
                .set_layouts(&device.set_layouts)
                .push_constant_ranges(&ranges),
            vk::ShaderCreateInfoEXT::default()
                .flags(vk::ShaderCreateFlagsEXT::LINK_STAGE)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(spirv)
                .name(&fragment)
                .set_layouts(&device.set_layouts)
                .push_constant_ranges(&ranges),
        ];
        let shaders = unsafe { device.shader_object_ext.create_shaders(&infos, None) }
            .map_err(|(_, result)| VkError::from(result))?;

        let layout = unsafe {
            device.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&device.set_layouts)
                    .push_constant_ranges(&ranges),
                None,
            )?
        };

        Ok(Self {
            shaders: [shaders[0], shaders[1]],
            layout,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            device: device.device.clone(),
            ext: device.shader_object_ext.clone(),
            dynamic3: device.dynamic_state3_ext.clone(),
        })
    }

    /// Binds the two stages and then supplies, by hand, every piece of state a graphics
    /// pipeline would have carried. Nothing here is optional: omitting any one of these is
    /// undefined behaviour rather than a default.
    pub fn bind(&self, command_buffer: vk::CommandBuffer, width: u32, height: u32) {
        let stages = [vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT];
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
            self.ext
                .cmd_bind_shaders(command_buffer, &stages, &self.shaders);

            // Vertex input: nothing, because vertices are pulled through a device pointer.
            self.ext.cmd_set_vertex_input(command_buffer, &[], &[]);
            self.ext
                .cmd_set_primitive_topology(command_buffer, self.topology);
            self.ext
                .cmd_set_primitive_restart_enable(command_buffer, false);

            self.ext
                .cmd_set_viewport_with_count(command_buffer, &[viewport]);
            self.ext
                .cmd_set_scissor_with_count(command_buffer, &[scissor]);

            self.ext
                .cmd_set_rasterizer_discard_enable(command_buffer, false);
            self.dynamic3
                .cmd_set_polygon_mode(command_buffer, vk::PolygonMode::FILL);
            self.ext
                .cmd_set_cull_mode(command_buffer, vk::CullModeFlags::NONE);
            self.ext
                .cmd_set_front_face(command_buffer, vk::FrontFace::COUNTER_CLOCKWISE);
            self.ext.cmd_set_depth_bias_enable(command_buffer, false);

            self.ext.cmd_set_depth_test_enable(command_buffer, false);
            self.ext.cmd_set_depth_write_enable(command_buffer, false);
            self.ext.cmd_set_stencil_test_enable(command_buffer, false);

            self.dynamic3
                .cmd_set_rasterization_samples(command_buffer, vk::SampleCountFlags::TYPE_1);
            self.dynamic3
                .cmd_set_sample_mask(command_buffer, vk::SampleCountFlags::TYPE_1, &[!0]);
            self.dynamic3
                .cmd_set_alpha_to_coverage_enable(command_buffer, false);
            self.dynamic3.cmd_set_logic_op_enable(command_buffer, false);
            self.dynamic3
                .cmd_set_color_blend_enable(command_buffer, 0, &[0]);
            self.dynamic3.cmd_set_color_write_mask(
                command_buffer,
                0,
                &[vk::ColorComponentFlags::RGBA],
            );
        }
    }
}

impl Drop for ShaderObjects {
    fn drop(&mut self) {
        unsafe {
            for &shader in &self.shaders {
                self.ext.destroy_shader(shader, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
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
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(desc.topology);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();

        let attachments: Vec<_> = desc.colors.iter().map(ColorTarget::attachment).collect();
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
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
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
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
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

    pub fn depth(mut self, texture: &HeapTexture, clear: Option<f32>, store: bool) -> Self {
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

pub trait IndexFormat: Copy {
    const VK_TYPE: vk::IndexType;
}

impl IndexFormat for u16 {
    const VK_TYPE: vk::IndexType = vk::IndexType::UINT16;
}

impl IndexFormat for u32 {
    const VK_TYPE: vk::IndexType = vk::IndexType::UINT32;
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
            command_buffer,
            layout,
            push: Cell::new(Push::default()),
        };
        raster.viewport(width, height);
        raster.cull(vk::CullModeFlags::NONE, vk::FrontFace::COUNTER_CLOCKWISE);
        raster.depth(&depth_state(vk::CompareOp::ALWAYS, false));
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

    pub fn pipeline(&self, pipeline: vk::Pipeline) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            )
        };
    }

    pub fn heap(&self, heap: &ResourceHeap) {
        heap.bind(self.command_buffer, self.layout, true);
    }

    pub fn shaders(&self, shaders: &ShaderObjects, width: u32, height: u32) {
        shaders.bind(self.command_buffer, width, height);
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

    pub fn draw_indexed<T, I: IndexFormat>(
        &self,
        root: Gpu<'_, T>,
        indices: &HeapBuffer,
        count: u32,
    ) {
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.command_buffer, indices.buffer, 0, I::VK_TYPE);
        }
        self.bind(root);
        unsafe {
            self.device
                .cmd_draw_indexed(self.command_buffer, count, 1, 0, 0, 0)
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

/// `MTLStages`, one for one. Metal's barrier takes producer and consumer stage masks and
/// nothing else; `VkPipelineStageFlags2` is the same idea with a wider vocabulary.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Stages(pub vk::PipelineStageFlags2);

impl Stages {
    pub const DISPATCH: Self = Self(vk::PipelineStageFlags2::COMPUTE_SHADER);
    pub const VERTEX: Self = Self(vk::PipelineStageFlags2::VERTEX_SHADER);
    pub const FRAGMENT: Self = Self(vk::PipelineStageFlags2::FRAGMENT_SHADER);
    pub const BLIT: Self = Self(vk::PipelineStageFlags2::ALL_TRANSFER);
    pub const INDIRECT: Self = Self(vk::PipelineStageFlags2::DRAW_INDIRECT);
}

/// Metal's `MTL4VisibilityOptions`. Vulkan expresses the same thing as access masks, which is
/// why this maps to a pair rather than a single flag.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Device,
    Threadgroup,
}

impl Visibility {
    /// Metal's visibility flag is stage-agnostic: `MTL4VisibilityOptions::DEVICE` means the
    /// same thing whatever stages the barrier joins. Vulkan access masks are not — pairing
    /// `SHADER_WRITE` with `ALL_TRANSFER`, or `SHADER_READ` with `DRAW_INDIRECT`, is invalid.
    /// `MEMORY_READ`/`MEMORY_WRITE` are the only masks legal with every stage, so they are what
    /// keeps `barrier(after, before, hazard)` portable across the whole API.
    pub fn access(self) -> (vk::AccessFlags2, vk::AccessFlags2) {
        match self {
            Visibility::Device => (
                vk::AccessFlags2::MEMORY_WRITE,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            ),
            Visibility::Threadgroup => (
                vk::AccessFlags2::SHADER_WRITE,
                vk::AccessFlags2::SHADER_READ,
            ),
        }
    }
}

pub struct Compute<'a> {
    pub device: &'a ash::Device,
    pub arena: &'a Arena,
    pub command_buffer: vk::CommandBuffer,
    pub layout: vk::PipelineLayout,
    pub push: Cell<Push>,
}

impl<'a> Compute<'a> {
    pub fn new(
        device: &'a ash::Device,
        arena: &'a Arena,
        command_buffer: vk::CommandBuffer,
        layout: vk::PipelineLayout,
    ) -> Self {
        Self {
            device,
            arena,
            command_buffer,
            layout,
            push: Cell::new(Push::default()),
        }
    }

    pub fn pipeline(&self, pipeline: vk::Pipeline) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
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
    pub fn barrier(&self, after: Stages, before: Stages, hazard: Visibility) {
        let (src_access, dst_access) = hazard.access();
        let barriers = [vk::MemoryBarrier2::default()
            .src_stage_mask(after.0)
            .src_access_mask(src_access)
            .dst_stage_mask(before.0)
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

/// Matches Slang's `Atomic<uint>` on the device side.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct AtomicU32(pub u32);

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
        Self::with_layout(device, shaders.layout, arena_size, threads)
    }

    /// Every pipeline layout in `nga-vk` is the same shape, so a `Context` only needs one —
    /// which is what lets the shader-object path reuse all of this unchanged.
    pub fn with_layout(
        device: &Device,
        layout: vk::PipelineLayout,
        arena_size: usize,
        threads: usize,
    ) -> Result<Self, VkError> {
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
                return Err(VkError::new(format!(
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
            arenas.push(Arena::new(device, arena_size, Placement::Bar)?);
        }

        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline = unsafe {
            device.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )?
        };

        Ok(Self {
            device: device.device.clone(),
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
    pub device: &'a ash::Device,
    pub queue: vk::Queue,
    pub arenas: &'a [Arena],
    pub command_buffers: &'a [vk::CommandBuffer],
    pub timeline: vk::Semaphore,
    pub layout: vk::PipelineLayout,
    pub recording: usize,
    pub value: u64,
    pub upload: Option<(vk::Semaphore, u64)>,
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
            arena: &self.arenas[index],
            command_buffer,
            layout: self.layout,
            not_sync: PhantomData,
        }
    }

    /// Make this frame wait on an upload batch on the GPU rather than on the CPU. Pair with
    /// `GpuHeap::flush_async`: the uploads are in flight while the frame is being recorded, and
    /// nothing blocks. `GpuHeap::flush` is the blocking equivalent, for load time.
    pub fn after_uploads(&mut self, value: u64, uploader: &Uploader) {
        self.upload = Some((uploader.timeline, value));
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
    pub device: &'a ash::Device,
    pub arena: &'a Arena,
    pub command_buffer: vk::CommandBuffer,
    pub layout: vk::PipelineLayout,
    pub not_sync: PhantomData<Cell<()>>,
}

impl Recorder<'_> {
    pub fn compute(&mut self) -> Compute<'_> {
        Compute::new(self.device, self.arena, self.command_buffer, self.layout)
    }

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
