#![allow(non_camel_case_types)]

extern crate self as nga;

pub mod abi;
pub mod msl;
pub mod tlsf;

use minmetal::*;
use std::cell::Cell;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

pub use abi::*;
pub use minmetal::FunctionConstantValues;
pub use msl::MSL_PRELUDE;
pub use nga_macros::{Constants, gpu_struct};
pub use tlsf::{Alloc, Tlsf};

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

pub const SHARED_UNTRACKED: ResourceOptions = ResourceOptions::new(
    CpuCacheMode::DefaultCache,
    StorageMode::Shared,
    HazardTrackingMode::Untracked,
);

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

pub struct Arena {
    pub buffer: Buffer,
    pub base: u64,
    pub cpu: *mut u8,
    pub size: usize,
    pub head: AtomicUsize,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new(device: &Device, size: usize) -> Result<Self, MetalError> {
        let buffer = device.new_buffer(size, SHARED_UNTRACKED)?;
        Ok(Self {
            base: buffer.gpu_address()?,
            cpu: buffer.contents() as *mut u8,
            size,
            head: AtomicUsize::new(0),
            buffer,
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
        let (cpu, gpu) = self.alloc_bytes(size_of::<T>(), align_of::<T>().max(16))?;
        unsafe { std::ptr::write(cpu as *mut T, value) };
        Some(GpuBox {
            cpu: unsafe { NonNull::new_unchecked(cpu as *mut T) },
            gpu: Gpu::new(gpu),
        })
    }

    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> Option<GpuBox<'_, [T]>> {
        let (cpu, gpu) = self.alloc_bytes(size_of_val(values), align_of::<T>().max(16))?;
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), cpu as *mut T, values.len()) };
        Some(GpuBox {
            cpu: NonNull::slice_from_raw_parts(
                unsafe { NonNull::new_unchecked(cpu as *mut T) },
                values.len(),
            ),
            gpu: Gpu::new(gpu),
        })
    }

    pub fn alloc_scratch<T>(&self, len: usize) -> Option<GpuBox<'_, [T]>> {
        let bytes = size_of::<T>() * len;
        let (cpu, gpu) = self.alloc_bytes(bytes, align_of::<T>().max(16))?;
        unsafe { std::ptr::write_bytes(cpu, 0, bytes) };
        Some(GpuBox {
            cpu: NonNull::slice_from_raw_parts(
                unsafe { NonNull::new_unchecked(cpu as *mut T) },
                len,
            ),
            gpu: Gpu::new(gpu),
        })
    }

    pub fn reset(&self) {
        self.head.store(0, Ordering::Relaxed);
    }
}

pub struct Pool<'a, T> {
    pub items: NonNull<T>,
    pub base: Gpu<'a, T>,
    pub cap: u32,
    pub bump: u32,
    pub free_head: u32,
    pub live: u32,
}

impl<'a, T> Pool<'a, T> {
    pub fn new(arena: &'a Arena, cap: u32) -> Option<Self> {
        debug_assert!(size_of::<T>() >= 4 && align_of::<T>() >= 4);
        let (cpu, gpu) =
            arena.alloc_bytes(size_of::<T>() * cap as usize, align_of::<T>().max(16))?;
        Some(Self {
            items: unsafe { NonNull::new_unchecked(cpu as *mut T) },
            base: Gpu::new(gpu),
            cap,
            bump: 0,
            free_head: tlsf::NIL,
            live: 0,
        })
    }

    pub fn alloc(&mut self, value: T) -> Option<u32> {
        let index = if self.free_head != tlsf::NIL {
            let index = self.free_head;
            self.free_head = unsafe { self.items.add(index as usize).cast::<u32>().read() };
            index
        } else if self.bump < self.cap {
            self.bump += 1;
            self.bump - 1
        } else {
            return None;
        };
        unsafe { self.items.add(index as usize).write(value) };
        self.live += 1;
        Some(index)
    }

    pub fn free(&mut self, index: u32) {
        debug_assert!(index < self.bump);
        unsafe {
            self.items
                .add(index as usize)
                .cast::<u32>()
                .write(self.free_head)
        };
        self.free_head = index;
        self.live -= 1;
    }

    pub fn gpu(&self, index: u32) -> Gpu<'a, T> {
        Gpu::new(self.base.0.get() + index as u64 * size_of::<T>() as u64)
    }
}

impl<T> Index<u32> for Pool<'_, T> {
    type Output = T;
    fn index(&self, index: u32) -> &T {
        debug_assert!(index < self.bump);
        unsafe { self.items.add(index as usize).as_ref() }
    }
}

impl<T> IndexMut<u32> for Pool<'_, T> {
    fn index_mut(&mut self, index: u32) -> &mut T {
        debug_assert!(index < self.bump);
        unsafe { self.items.add(index as usize).as_mut() }
    }
}

pub struct HeapBuffer {
    pub buffer: Buffer,
    pub gpu: u64,
    pub cpu: *mut u8,
    pub alloc: Alloc,
}

impl HeapBuffer {
    pub fn write<T: Copy>(&mut self, offset: usize, values: &[T]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.cpu.add(offset) as *mut T,
                values.len(),
            )
        };
    }
}

pub struct HeapTexture {
    pub texture: Texture,
    pub alloc: Alloc,
}

impl HeapTexture {
    pub fn upload(&self, mip: usize, bytes: &[u8], stride: usize) {
        let (width, height) = (self.texture.width() >> mip, self.texture.height() >> mip);
        debug_assert_eq!(bytes.len(), stride * height);
        self.texture
            .replace_region(Region::new_2d(0, 0, width, height), mip, bytes, stride);
    }
}

pub struct GpuHeap {
    pub device: Device,
    pub heap: Heap,
    pub tlsf: Tlsf,
}

impl GpuHeap {
    pub fn new(device: &Device, size: u64) -> Result<Self, MetalError> {
        let desc = HeapDescriptor::new();
        desc.set_heap_type(HeapType::Placement);
        desc.set_storage_mode(StorageMode::Shared);
        desc.set_hazard_tracking_mode(HazardTrackingMode::Untracked);
        desc.set_size(size as usize);
        Ok(Self {
            heap: device.new_heap(&desc)?,
            device: device.clone(),
            tlsf: Tlsf::new(size, 256),
        })
    }

    pub fn alloc_buffer(&mut self, len: usize) -> Result<HeapBuffer, MetalError> {
        let need = self
            .device
            .heap_buffer_size_and_align(len, SHARED_UNTRACKED);
        let alloc = self
            .tlsf
            .alloc(need.size as u64, need.align as u64)
            .ok_or(MetalError::new("gpu heap is full"))?;
        let buffer =
            self.heap
                .new_buffer_at_offset(len, SHARED_UNTRACKED, alloc.offset as usize)?;
        Ok(HeapBuffer {
            gpu: buffer.gpu_address()?,
            cpu: buffer.contents() as *mut u8,
            alloc,
            buffer,
        })
    }

    pub fn alloc_texture(&mut self, desc: &TextureDescriptor) -> Result<HeapTexture, MetalError> {
        desc.set_storage_mode(StorageMode::Shared);
        desc.set_resource_options(SHARED_UNTRACKED);
        let need = self.device.heap_texture_size_and_align(desc);
        let alloc = self
            .tlsf
            .alloc(need.size as u64, need.align as u64)
            .ok_or(MetalError::new("gpu heap is full"))?;
        Ok(HeapTexture {
            texture: self
                .heap
                .new_texture_at_offset(desc, alloc.offset as usize)?,
            alloc,
        })
    }

    pub fn free(&mut self, alloc: Alloc) {
        self.tlsf.free(alloc.block);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HeapRange {
    pub base: u32,
    pub len: u32,
    pub block: u32,
}

pub struct ResourceHeap {
    pub storage: HeapBuffer,
    pub ranges: Tlsf,
    pub cap: u32,
    pub pending: VecDeque<(u64, HeapRange)>,
}

impl ResourceHeap {
    pub fn new(heap: &mut GpuHeap, cap: u32) -> Result<Self, MetalError> {
        let storage = heap.alloc_buffer(cap as usize * size_of::<ResourceID>())?;
        unsafe { std::ptr::write_bytes(storage.cpu, 0, cap as usize * size_of::<ResourceID>()) };
        Ok(Self {
            storage,
            ranges: Tlsf::new(cap as u64, 1),
            cap,
            pending: VecDeque::new(),
        })
    }

    pub fn alloc_range(&mut self, len: u32) -> Result<HeapRange, MetalError> {
        let alloc = self
            .ranges
            .alloc(len as u64, 1)
            .ok_or(MetalError::new("resource heap has no free range"))?;
        Ok(HeapRange {
            base: alloc.offset as u32,
            len,
            block: alloc.block,
        })
    }

    pub fn free_range(&mut self, range: HeapRange, frame: u64) {
        debug_assert!(self.pending.back().is_none_or(|&(last, _)| last <= frame));
        self.pending.push_back((frame, range));
    }

    pub fn retire(&mut self, completed: u64) {
        while let Some(&(frame, range)) = self.pending.front() {
            if frame > completed {
                return;
            }
            self.pending.pop_front();
            self.ranges.free(range.block);
        }
    }

    pub fn write(&mut self, range: HeapRange, slot: u32, resource: ResourceID) {
        assert!(slot < range.len);
        unsafe {
            self.storage
                .cpu
                .cast::<ResourceID>()
                .add((range.base + slot) as usize)
                .write(resource)
        };
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct AtomicU32(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Commands(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Pipeline(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Texture2d(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Texture2dW(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Texture2dRW(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Texture2dArray(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Texture3d(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Texture3dW(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Texture3dRW(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct TextureCube(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Depth2d(pub ResourceID);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Sampler(pub ResourceID);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExecRange {
    pub location: u32,
    pub length: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub vertex_start: u32,
    pub base_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrawIndexed {
    pub index_count: u32,
    pub instance_count: u32,
    pub index_start: u32,
    pub base_vertex: i32,
    pub base_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dispatch {
    pub threadgroups: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DispatchThreads {
    pub threads: [u32; 3],
    pub group: [u32; 3],
}

pub struct IndirectCommands {
    pub icb: IndirectCommandBuffer,
    pub handle: Commands,
    pub count: usize,
}

impl IndirectCommands {
    pub fn draws(device: &Device, count: usize, indexed: bool) -> Result<Self, MetalError> {
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(match indexed {
            true => IndirectCommandType::DRAW_INDEXED,
            false => IndirectCommandType::DRAW,
        });
        desc.set_inherit_pipeline_state(true);
        desc.set_inherit_buffers(false);
        desc.set_max_vertex_buffer_bind_count(2);
        desc.set_max_fragment_buffer_bind_count(2);
        Self::new(device, &desc, count)
    }

    pub fn dispatches(device: &Device, count: usize) -> Result<Self, MetalError> {
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
        desc.set_inherit_pipeline_state(true);
        desc.set_inherit_buffers(false);
        desc.set_max_kernel_buffer_bind_count(2);
        Self::new(device, &desc, count)
    }

    pub fn new(
        device: &Device,
        desc: &IndirectCommandBufferDescriptor,
        count: usize,
    ) -> Result<Self, MetalError> {
        let icb =
            device.new_indirect_command_buffer(desc, count, IndirectCommandBufferOptions::NONE)?;
        Ok(Self {
            handle: Commands(icb.gpu_resource_id()?),
            icb,
            count,
        })
    }

    pub fn all(&self) -> minmetal::Range {
        minmetal::Range {
            location: 0,
            length: self.count,
        }
    }
}

pub struct Shaders {
    pub compiler: M4Compiler,
    pub library: Library,
}

impl Shaders {
    pub fn new(device: &Device, source: &str) -> Result<Self, MetalError> {
        let compiler = device.new_m4_compiler_with_descriptor(&M4CompilerDescriptor::new())?;
        let desc = M4LibraryDescriptor::new();
        desc.set_source(&format!("{MSL_PRELUDE}\n{source}"));
        Ok(Self {
            library: compiler.new_library_with_descriptor(&desc)?,
            compiler,
        })
    }

    pub fn function(&self, name: &str, constants: Option<&dyn Constants>) -> M4FunctionDescriptor {
        let function = M4LibraryFunctionDescriptor::new();
        function.set_name(name);
        function.set_library(Some(&self.library));
        let Some(constants) = constants else {
            return function.as_function_descriptor();
        };
        let specialized = M4SpecializedFunctionDescriptor::new();
        specialized.set_function_descriptor(Some(&function.as_function_descriptor()));
        specialized.set_constant_values(Some(&constants.values()));
        specialized.set_specialized_name(name);
        M4FunctionDescriptor {
            raw: retain(specialized.raw),
        }
    }

    pub fn compute(
        &self,
        name: &str,
        constants: Option<&dyn Constants>,
    ) -> Result<ComputePipelineState, MetalError> {
        let pipeline = M4ComputePipelineDescriptor::new();
        pipeline.set_label(name);
        pipeline.set_compute_function_descriptor(Some(&self.function(name, constants)));
        self.compiler.new_compute_pipeline_state(&pipeline, None)
    }

    pub fn raster(&self, desc: &RasterDesc) -> Result<RenderPipelineState, MetalError> {
        let pipeline = M4RenderPipelineDescriptor::new();
        pipeline.set_label(desc.label);
        pipeline.set_vertex_function_descriptor(Some(&self.function(desc.vertex, desc.constants)));
        if !desc.fragment.is_empty() {
            let fragment = self.function(desc.fragment, desc.constants);
            pipeline.set_fragment_function_descriptor(Some(&fragment));
        }
        pipeline.set_support_indirect_command_buffers(match desc.indirect {
            true => M4IndirectCommandBufferSupportState::Enabled,
            false => M4IndirectCommandBufferSupportState::Disabled,
        });
        pipeline.set_rasterization_enabled(!desc.colors.is_empty() || !desc.fragment.is_empty());
        pipeline.set_raster_sample_count(desc.samples);
        pipeline.set_alpha_to_coverage_state(match desc.alpha_to_coverage {
            true => M4AlphaToCoverageState::Enabled,
            false => M4AlphaToCoverageState::Disabled,
        });
        let attachments = pipeline.color_attachments();
        for (index, target) in desc.colors.iter().enumerate() {
            let attachment = attachments.object_at(index);
            attachment.set_pixel_format(target.format);
            attachment.set_write_mask(target.write);
            match target.blend {
                BlendMode::Fixed(blend) => {
                    attachment.set_blending_state(M4BlendState::Enabled);
                    attachment.set_blend_factors(
                        blend.src_rgb,
                        blend.dst_rgb,
                        blend.rgb_op,
                        blend.src_alpha,
                        blend.dst_alpha,
                        blend.alpha_op,
                    );
                }
                _ => attachment.set_blending_state(M4BlendState::Disabled),
            }
        }
        self.compiler
            .new_render_pipeline_state(&pipeline.as_pipeline_descriptor(), None)
    }

    pub fn mesh(&self, desc: &MeshDesc) -> Result<RenderPipelineState, MetalError> {
        let pipeline = M4MeshRenderPipelineDescriptor::new();
        pipeline.set_label(desc.label);
        if !desc.object.is_empty() {
            let object = self.function(desc.object, desc.constants);
            pipeline.set_object_function_descriptor(Some(&object));
            pipeline.set_payload_memory_length(desc.payload);
        }
        pipeline.set_mesh_function_descriptor(Some(&self.function(desc.mesh, desc.constants)));
        if !desc.fragment.is_empty() {
            let fragment = self.function(desc.fragment, desc.constants);
            pipeline.set_fragment_function_descriptor(Some(&fragment));
        }
        pipeline.set_support_indirect_command_buffers(match desc.indirect {
            true => M4IndirectCommandBufferSupportState::Enabled,
            false => M4IndirectCommandBufferSupportState::Disabled,
        });
        pipeline.set_rasterization_enabled(!desc.colors.is_empty() || !desc.fragment.is_empty());
        pipeline.set_raster_sample_count(desc.samples);
        pipeline.set_alpha_to_coverage_state(match desc.alpha_to_coverage {
            true => M4AlphaToCoverageState::Enabled,
            false => M4AlphaToCoverageState::Disabled,
        });
        let attachments = pipeline.color_attachments();
        for (index, target) in desc.colors.iter().enumerate() {
            let attachment = attachments.object_at(index);
            attachment.set_pixel_format(target.format);
            attachment.set_write_mask(target.write);
            match target.blend {
                BlendMode::Fixed(blend) => {
                    attachment.set_blending_state(M4BlendState::Enabled);
                    attachment.set_blend_factors(
                        blend.src_rgb,
                        blend.dst_rgb,
                        blend.rgb_op,
                        blend.src_alpha,
                        blend.dst_alpha,
                        blend.alpha_op,
                    );
                }
                _ => attachment.set_blending_state(M4BlendState::Disabled),
            }
        }
        self.compiler
            .new_render_pipeline_state(&pipeline.as_pipeline_descriptor(), None)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Blend {
    pub src_rgb: BlendFactor,
    pub dst_rgb: BlendFactor,
    pub rgb_op: BlendOperation,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_op: BlendOperation,
}

impl Blend {
    pub const ALPHA: Self = Self {
        src_rgb: BlendFactor::SourceAlpha,
        dst_rgb: BlendFactor::OneMinusSourceAlpha,
        rgb_op: BlendOperation::Add,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::OneMinusSourceAlpha,
        alpha_op: BlendOperation::Add,
    };
    pub const PREMULTIPLIED: Self = Self {
        src_rgb: BlendFactor::One,
        ..Self::ALPHA
    };
    pub const ADDITIVE: Self = Self {
        src_rgb: BlendFactor::One,
        dst_rgb: BlendFactor::One,
        rgb_op: BlendOperation::Add,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::One,
        alpha_op: BlendOperation::Add,
    };
}

#[derive(Clone, Copy, Debug)]
pub enum BlendMode {
    Opaque,
    Fixed(Blend),
    Programmable,
}

#[derive(Clone, Copy, Debug)]
pub struct ColorTarget {
    pub format: PixelFormat,
    pub blend: BlendMode,
    pub write: ColorWriteMask,
}

impl ColorTarget {
    pub fn new(format: PixelFormat) -> Self {
        Self {
            format,
            blend: BlendMode::Opaque,
            write: ColorWriteMask::ALL,
        }
    }
}

pub struct RasterDesc<'a> {
    pub vertex: &'a str,
    pub fragment: &'a str,
    pub colors: &'a [ColorTarget],
    pub samples: usize,
    pub alpha_to_coverage: bool,
    pub label: &'a str,
    pub constants: Option<&'a dyn Constants>,
    pub indirect: bool,
}

impl<'a> RasterDesc<'a> {
    pub fn new(vertex: &'a str, fragment: &'a str, colors: &'a [ColorTarget]) -> Self {
        Self {
            vertex,
            fragment,
            colors,
            samples: 1,
            alpha_to_coverage: false,
            label: vertex,
            constants: None,
            indirect: false,
        }
    }
}

pub struct MeshDesc<'a> {
    pub object: &'a str,
    pub mesh: &'a str,
    pub fragment: &'a str,
    pub colors: &'a [ColorTarget],
    pub payload: usize,
    pub samples: usize,
    pub alpha_to_coverage: bool,
    pub label: &'a str,
    pub constants: Option<&'a dyn Constants>,
    pub indirect: bool,
}

impl<'a> MeshDesc<'a> {
    pub fn new(
        object: &'a str,
        mesh: &'a str,
        fragment: &'a str,
        colors: &'a [ColorTarget],
        payload: usize,
    ) -> Self {
        Self {
            object,
            mesh,
            fragment,
            colors,
            payload,
            samples: 1,
            alpha_to_coverage: false,
            label: mesh,
            constants: None,
            indirect: false,
        }
    }
}

pub fn depth_state(
    device: &Device,
    compare: CompareFunction,
    write: bool,
) -> Result<DepthStencilState, MetalError> {
    let desc = DepthStencilDescriptor::new();
    desc.set_depth_compare_function(compare);
    desc.set_depth_write_enabled(write);
    device.new_depth_stencil_state(&desc)
}

pub struct Pass {
    pub desc: M4RenderPassDescriptor,
}

impl Pass {
    pub fn new() -> Self {
        Self {
            desc: M4RenderPassDescriptor::new(),
        }
    }

    pub fn color(self, index: usize, texture: &Texture, clear: Option<ClearColor>) -> Self {
        let attachment = self
            .desc
            .color_attachments()
            .object_at_indexed_subscript(index);
        attachment.base().set_texture(Some(texture));
        attachment.base().set_store_action(StoreAction::Store);
        match clear {
            Some(color) => {
                attachment.base().set_load_action(LoadAction::Clear);
                attachment.set_clear_color(color);
            }
            None => attachment.base().set_load_action(LoadAction::Load),
        }
        self
    }

    pub fn depth(self, texture: &Texture, clear: Option<f64>) -> Self {
        let attachment = self.desc.depth_attachment();
        attachment.base().set_texture(Some(texture));
        attachment.base().set_store_action(StoreAction::DontCare);
        match clear {
            Some(depth) => {
                attachment.base().set_load_action(LoadAction::Clear);
                attachment.set_clear_depth(depth);
            }
            None => attachment.base().set_load_action(LoadAction::Load),
        }
        self
    }
}

pub trait IndexFormat: Copy {
    const INDEX_TYPE: IndexType;
}

impl IndexFormat for u16 {
    const INDEX_TYPE: IndexType = IndexType::UInt16;
}

impl IndexFormat for u32 {
    const INDEX_TYPE: IndexType = IndexType::UInt32;
}

pub struct Raster<'a> {
    pub encoder: M4RenderCommandEncoder,
    pub table: &'a M4ArgumentTable,
    pub command_buffer: &'a M4CommandBuffer,
}

impl<'a> Raster<'a> {
    pub fn new(
        command_buffer: &'a M4CommandBuffer,
        pass: &Pass,
        table: &'a M4ArgumentTable,
        viewport: Viewport,
    ) -> Option<Self> {
        let encoder = command_buffer.render_command_encoder_with_descriptor(&pass.desc)?;
        encoder.set_argument_table(
            table,
            RenderStages::VERTEX
                | RenderStages::FRAGMENT
                | RenderStages::OBJECT
                | RenderStages::MESH,
        );
        encoder.set_viewport(viewport);
        Some(Self {
            encoder,
            table,
            command_buffer,
        })
    }

    pub fn pipeline(&self, pipeline: &RenderPipelineState) {
        self.encoder.set_render_pipeline_state(pipeline);
    }

    pub fn depth(&self, state: &DepthStencilState) {
        self.encoder.set_depth_stencil_state(state);
    }

    pub fn cull(&self, mode: CullMode, winding: Winding) {
        self.encoder.set_cull_mode(mode);
        self.encoder.set_front_facing_winding(winding);
    }

    pub fn barrier(&self, after: Stages, before: Stages, hazard: M4VisibilityOptions) {
        self.encoder.barrier(after, before, hazard);
    }

    pub fn wait(&self, after_queue: Stages, before: Stages, hazard: M4VisibilityOptions) {
        self.encoder
            .barrier_after_queue_stages(after_queue, before, hazard);
    }

    pub fn signal(&self, after: Stages, before_queue: Stages, hazard: M4VisibilityOptions) {
        self.encoder
            .barrier_before_queue_stages(after, before_queue, hazard);
    }

    pub fn signal_fence(&self, fence: &Fence, after: Stages) {
        self.encoder.update_fence(fence, after);
    }

    pub fn wait_fence(&self, fence: &Fence, before: Stages) {
        self.encoder.wait_for_fence(fence, before);
    }

    pub fn draw<T>(&self, root: Gpu<'_, T>, primitive: PrimitiveType, start: usize, count: usize) {
        self.table.set_address(root.0.get(), 0);
        self.encoder.draw_primitives(primitive, start, count);
    }

    pub fn draw_indexed<T, I: IndexFormat>(
        &self,
        root: Gpu<'_, T>,
        primitive: PrimitiveType,
        indices: Gpu<'_, I>,
        count: usize,
    ) {
        self.table.set_address(root.0.get(), 0);
        self.encoder.draw_indexed_primitives(
            primitive,
            count,
            I::INDEX_TYPE,
            indices.0.get(),
            count * size_of::<I>(),
        );
    }

    pub fn draw_mesh<T>(
        &self,
        root: Gpu<'_, T>,
        groups: usize,
        object_group: usize,
        mesh_group: usize,
    ) {
        self.table.set_address(root.0.get(), 0);
        self.encoder.draw_mesh_threadgroups(
            Size::new(groups, 1, 1),
            Size::new(object_group, 1, 1),
            Size::new(mesh_group, 1, 1),
        );
    }

    pub fn draw_indirect<T>(&self, root: Gpu<'_, T>, primitive: PrimitiveType, args: Gpu<'_, Draw>) {
        self.table.set_address(root.0.get(), 0);
        self.encoder
            .draw_primitives_with_indirect_buffer(primitive, args.0.get());
    }

    pub fn draw_indexed_indirect<T, I: IndexFormat>(
        &self,
        root: Gpu<'_, T>,
        primitive: PrimitiveType,
        indices: Gpu<'_, I>,
        index_count: usize,
        args: Gpu<'_, DrawIndexed>,
    ) {
        self.table.set_address(root.0.get(), 0);
        self.encoder.draw_indexed_primitives_with_indirect_buffer(
            primitive,
            I::INDEX_TYPE,
            indices.0.get(),
            index_count * size_of::<I>(),
            args.0.get(),
        );
    }

    pub fn execute(&self, commands: &IndirectCommands, range: minmetal::Range) {
        self.encoder.execute_commands_in_buffer(&commands.icb, range);
    }

    pub fn execute_indirect(&self, commands: &IndirectCommands, range: Gpu<'_, ExecRange>) {
        self.encoder
            .execute_commands_in_buffer_with_indirect_buffer(&commands.icb, range.0.get());
    }
}

impl Drop for Raster<'_> {
    fn drop(&mut self) {
        self.encoder.end_encoding();
    }
}

pub struct Compute<'a> {
    pub encoder: M4ComputeCommandEncoder,
    pub table: &'a M4ArgumentTable,
    pub command_buffer: &'a M4CommandBuffer,
}

impl<'a> Compute<'a> {
    pub fn new(command_buffer: &'a M4CommandBuffer, table: &'a M4ArgumentTable) -> Option<Self> {
        let encoder = command_buffer.compute_command_encoder()?;
        encoder.set_argument_table(Some(table));
        Some(Self {
            encoder,
            table,
            command_buffer,
        })
    }

    pub fn pipeline(&self, pipeline: &ComputePipelineState) {
        self.encoder.set_compute_pipeline_state(pipeline);
    }

    pub fn dispatch<T>(&self, root: Gpu<'_, T>, threads: usize, group: usize) {
        self.table.set_address(root.0.get(), 0);
        self.encoder
            .dispatch_threads(Size::new(threads, 1, 1), Size::new(group, 1, 1));
    }

    pub fn dispatch_indirect<T>(&self, root: Gpu<'_, T>, args: Gpu<'_, Dispatch>, group: usize) {
        self.table.set_address(root.0.get(), 0);
        self.encoder
            .dispatch_threadgroups_with_indirect_buffer(args.0.get(), Size::new(group, 1, 1));
    }

    pub fn dispatch_threads_indirect<T>(&self, root: Gpu<'_, T>, args: Gpu<'_, DispatchThreads>) {
        self.table.set_address(root.0.get(), 0);
        self.encoder
            .dispatch_threads_with_indirect_buffer(args.0.get());
    }

    pub fn execute(&self, commands: &IndirectCommands, range: minmetal::Range) {
        self.encoder.execute_commands_in_buffer(&commands.icb, range);
    }

    pub fn execute_indirect(&self, commands: &IndirectCommands, range: Gpu<'_, ExecRange>) {
        self.encoder
            .execute_commands_in_buffer_with_indirect_buffer(&commands.icb, range.0.get());
    }

    pub fn reset(&self, commands: &IndirectCommands, range: minmetal::Range) {
        self.encoder.reset_commands_in_buffer(&commands.icb, range);
    }

    pub fn barrier(&self, after: Stages, before: Stages, hazard: M4VisibilityOptions) {
        self.encoder.barrier(after, before, hazard);
    }

    pub fn wait(&self, after_queue: Stages, before: Stages, hazard: M4VisibilityOptions) {
        self.encoder
            .barrier_after_queue_stages(after_queue, before, hazard);
    }

    pub fn signal(&self, after: Stages, before_queue: Stages, hazard: M4VisibilityOptions) {
        self.encoder
            .barrier_before_queue_stages(after, before_queue, hazard);
    }

    pub fn signal_fence(&self, fence: &Fence, after: Stages) {
        self.encoder.update_fence(fence, after);
    }

    pub fn wait_fence(&self, fence: &Fence, before: Stages) {
        self.encoder.wait_for_fence(fence, before);
    }
}

impl Drop for Compute<'_> {
    fn drop(&mut self) {
        self.encoder.end_encoding();
    }
}

pub const FRAMES_IN_FLIGHT: u64 = 2;

pub enum Retired {
    Buffer(HeapBuffer),
    Texture(HeapTexture),
}

pub struct Context {
    pub device: Device,
    pub queue: M4CommandQueue,
    pub command_buffers: Vec<M4CommandBuffer>,
    pub tables: Vec<M4ArgumentTable>,
    pub event: SharedEvent,
    pub residency: ResidencySet,
    pub heap: GpuHeap,
    pub resources: ResourceHeap,
    pub arenas: Vec<Arena>,
    pub allocators: Vec<M4CommandAllocator>,
    pub threads: usize,
    pub retired: VecDeque<(u64, Retired)>,
    pub frame: u64,
}

impl Context {
    pub fn new(
        device: &Device,
        arena_size: usize,
        heap_size: u64,
        resource_cap: u32,
        threads: usize,
    ) -> Result<Self, MetalError> {
        assert!(threads > 0);
        let queue = device.new_m4_command_queue().ok_or(MetalError::new(
            "MTL4CommandQueue is unavailable — Metal 4 requires macOS 26",
        ))?;

        let mut arenas = Vec::new();
        let mut allocators = Vec::new();
        let mut command_buffers = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT as usize * threads {
            arenas.push(Arena::new(device, arena_size)?);
            allocators.push(
                device
                    .new_m4_command_allocator()
                    .ok_or(MetalError::new("newCommandAllocator returned nil"))?,
            );
            command_buffers.push(
                device
                    .new_m4_command_buffer()
                    .ok_or(MetalError::new("newCommandBuffer returned nil"))?,
            );
        }

        let mut heap = GpuHeap::new(device, heap_size)?;
        let resources = ResourceHeap::new(&mut heap, resource_cap)?;
        let residency = device.new_residency_set(&ResidencySetDescriptor::new()?)?;
        for arena in &arenas {
            residency.add_allocation(&Allocation::from_buffer(&arena.buffer))?;
        }
        residency.add_allocation(&Allocation::from_heap(&heap.heap))?;
        residency.commit()?;
        queue.add_residency_set(&residency);

        let table_desc = M4ArgumentTableDescriptor::new();
        table_desc.set_max_buffer_bind_count(2);
        let mut tables = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT as usize * threads {
            let table = device.new_m4_argument_table_with_descriptor(&table_desc)?;
            table.set_address(resources.storage.gpu, 1);
            tables.push(table);
        }

        Ok(Self {
            tables,
            event: device.new_shared_event()?,
            device: device.clone(),
            queue,
            command_buffers,
            residency,
            heap,
            resources,
            arenas,
            allocators,
            threads,
            retired: VecDeque::new(),
            frame: 0,
        })
    }

    pub fn resident(&self, allocation: &Allocation) -> Result<(), MetalError> {
        self.residency.add_allocation(allocation)?;
        self.residency.commit()
    }

    pub fn free_later(&mut self, retired: Retired) {
        self.retired.push_back((self.frame, retired));
    }

    pub fn begin_frame(&mut self) -> Frame<'_> {
        let index = self.frame;
        self.frame += 1;
        if index >= FRAMES_IN_FLIGHT {
            self.event
                .wait_until_signaled_value(index + 1 - FRAMES_IN_FLIGHT, 1000);
        }

        let completed = self.event.signaled_value();
        self.resources.retire(completed);
        while self.retired.front().is_some_and(|&(frame, _)| frame <= completed) {
            match self.retired.pop_front().unwrap().1 {
                Retired::Buffer(buffer) => self.heap.free(buffer.alloc),
                Retired::Texture(texture) => self.heap.free(texture.alloc),
            }
        }

        let base = (index % FRAMES_IN_FLIGHT) as usize * self.threads;
        let slots = base..base + self.threads;
        for slot in slots.clone() {
            self.allocators[slot].reset();
            self.arenas[slot].reset();
        }

        Frame {
            arenas: &self.arenas[slots.clone()],
            allocators: &self.allocators[slots.clone()],
            command_buffers: &self.command_buffers[slots.clone()],
            tables: &self.tables[slots],
            queue: &self.queue,
            event: &self.event,
            value: index + 1,
            used: 0,
        }
    }
}

pub struct Frame<'a> {
    pub arenas: &'a [Arena],
    pub allocators: &'a [M4CommandAllocator],
    pub command_buffers: &'a [M4CommandBuffer],
    pub tables: &'a [M4ArgumentTable],
    pub queue: &'a M4CommandQueue,
    pub event: &'a SharedEvent,
    pub value: u64,
    pub used: usize,
}

impl Frame<'_> {
    pub fn recorders(&mut self, count: usize) -> Vec<Recorder<'_>> {
        assert!(self.used == 0, "recorders already taken for this frame");
        assert!(
            count <= self.command_buffers.len(),
            "{count} recorders requested, Context::new was given {}",
            self.command_buffers.len()
        );
        self.used = count;
        (0..count)
            .map(|i| {
                self.command_buffers[i].begin_command_buffer_with_allocator(&self.allocators[i]);
                Recorder {
                    arena: &self.arenas[i],
                    command_buffer: &self.command_buffers[i],
                    table: &self.tables[i],
                    not_sync: PhantomData,
                }
            })
            .collect()
    }

    pub fn recorder(&mut self) -> Recorder<'_> {
        self.recorders(1).pop().unwrap()
    }

    pub fn submit(self, drawable: Option<&Drawable>) {
        if let Some(drawable) = drawable {
            self.queue.wait_for_drawable(drawable);
        }
        self.queue.commit(&self.command_buffers[..self.used]);
        self.queue.signal_event(&self.event.as_event(), self.value);
        if let Some(drawable) = drawable {
            self.queue.signal_drawable(drawable);
            drawable.present();
        }
    }
}

pub struct Recorder<'a> {
    pub arena: &'a Arena,
    pub command_buffer: &'a M4CommandBuffer,
    pub table: &'a M4ArgumentTable,
    pub not_sync: PhantomData<Cell<()>>,
}

impl Recorder<'_> {
    pub fn raster(&mut self, pass: &Pass, viewport: Viewport) -> Option<Raster<'_>> {
        Raster::new(self.command_buffer, pass, self.table, viewport)
    }

    pub fn compute(&mut self) -> Option<Compute<'_>> {
        Compute::new(self.command_buffer, self.table)
    }
}

impl Drop for Recorder<'_> {
    fn drop(&mut self) {
        self.command_buffer.end_command_buffer();
    }
}
