#![allow(non_camel_case_types)]

extern crate self as nga;

pub mod abi;
pub mod msl;
pub mod tlsf;

use minmetal::*;
use std::cell::Cell;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;

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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuSlice<'a, T> {
    pub ptr: Gpu<'a, T>,
    pub len: u32,
}

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
    pub head: Cell<usize>,
}

impl Arena {
    pub fn new(device: &Device, size: usize) -> Result<Self, MetalError> {
        let buffer = device.new_buffer(size, SHARED_UNTRACKED)?;
        Ok(Self {
            base: buffer.gpu_address()?,
            cpu: buffer.contents() as *mut u8,
            size,
            head: Cell::new(0),
            buffer,
        })
    }

    pub fn alloc_bytes(&self, size: usize, align: usize) -> Option<(*mut u8, u64)> {
        let offset = (self.head.get() + align - 1) & !(align - 1);
        if offset + size > self.size {
            return None;
        }
        self.head.set(offset + size);
        unsafe { Some((self.cpu.add(offset), self.base + offset as u64)) }
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

    pub fn reset(&self) {
        self.head.set(0);
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
            tlsf: Tlsf::new(size),
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
        self.tlsf.free(alloc);
    }
}

pub struct RangeAllocator {
    pub free: Vec<(u32, u32)>,
}

impl RangeAllocator {
    pub fn new(cap: u32) -> Self {
        Self {
            free: vec![(0, cap)],
        }
    }

    pub fn alloc(&mut self, len: u32) -> Option<u32> {
        let index = self.free.iter().position(|&(_, free)| free >= len)?;
        let (base, free) = self.free[index];
        if free == len {
            self.free.remove(index);
        } else {
            self.free[index] = (base + len, free - len);
        }
        Some(base)
    }

    pub fn free(&mut self, base: u32, len: u32) {
        let index = self.free.partition_point(|&(start, _)| start < base);
        debug_assert!(index == self.free.len() || base + len <= self.free[index].0);
        debug_assert!(index == 0 || self.free[index - 1].0 + self.free[index - 1].1 <= base);
        self.free.insert(index, (base, len));
        if index + 1 < self.free.len() && base + len == self.free[index + 1].0 {
            self.free[index].1 += self.free[index + 1].1;
            self.free.remove(index + 1);
        }
        if index > 0 && self.free[index - 1].0 + self.free[index - 1].1 == base {
            self.free[index - 1].1 += self.free[index].1;
            self.free.remove(index);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HeapRange {
    pub base: u32,
    pub len: u32,
}

pub struct ResourceHeap {
    pub storage: HeapBuffer,
    pub ranges: RangeAllocator,
    pub cap: u32,
}

impl ResourceHeap {
    pub fn new(heap: &mut GpuHeap, cap: u32) -> Result<Self, MetalError> {
        let storage = heap.alloc_buffer(cap as usize * size_of::<ResourceID>())?;
        unsafe { std::ptr::write_bytes(storage.cpu, 0, cap as usize * size_of::<ResourceID>()) };
        Ok(Self {
            storage,
            ranges: RangeAllocator::new(cap),
            cap,
        })
    }

    pub fn alloc_range(&mut self, len: u32) -> Result<HeapRange, MetalError> {
        let base = self
            .ranges
            .alloc(len)
            .ok_or(MetalError::new("resource heap has no free range"))?;
        Ok(HeapRange { base, len })
    }

    pub fn free_range(&mut self, range: HeapRange) {
        self.ranges.free(range.base, range.len);
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

    pub fn function(
        &self,
        name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> M4FunctionDescriptor {
        let function = M4LibraryFunctionDescriptor::new();
        function.set_name(name);
        function.set_library(Some(&self.library));
        let Some(constants) = constants else {
            return function.as_function_descriptor();
        };
        let specialized = M4SpecializedFunctionDescriptor::new();
        specialized.set_function_descriptor(Some(&function.as_function_descriptor()));
        specialized.set_constant_values(Some(constants));
        specialized.set_specialized_name(name);
        M4FunctionDescriptor {
            raw: retain(specialized.raw),
        }
    }

    pub fn compute(
        &self,
        name: &str,
        constants: Option<&FunctionConstantValues>,
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
    pub constants: Option<&'a FunctionConstantValues>,
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
        encoder.set_argument_table(table, RenderStages::VERTEX | RenderStages::FRAGMENT);
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
    pub command_buffer: M4CommandBuffer,
    pub table: M4ArgumentTable,
    pub event: SharedEvent,
    pub residency: ResidencySet,
    pub heap: GpuHeap,
    pub arenas: Vec<Arena>,
    pub allocators: Vec<M4CommandAllocator>,
    pub retired: Vec<(u64, Retired)>,
    pub frame: u64,
}

impl Context {
    pub fn new(device: &Device, arena_size: usize, heap_size: u64) -> Result<Self, MetalError> {
        let queue = device.new_m4_command_queue().ok_or(MetalError::new(
            "MTL4CommandQueue is unavailable — Metal 4 requires macOS 26",
        ))?;
        let command_buffer = device
            .new_m4_command_buffer()
            .ok_or(MetalError::new("newCommandBuffer returned nil"))?;

        let mut arenas = Vec::new();
        let mut allocators = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            arenas.push(Arena::new(device, arena_size)?);
            allocators.push(
                device
                    .new_m4_command_allocator()
                    .ok_or(MetalError::new("newCommandAllocator returned nil"))?,
            );
        }

        let heap = GpuHeap::new(device, heap_size)?;
        let residency = device.new_residency_set(&ResidencySetDescriptor::new()?)?;
        for arena in &arenas {
            residency.add_allocation(&Allocation::from_buffer(&arena.buffer))?;
        }
        residency.add_allocation(&Allocation::from_heap(&heap.heap))?;
        residency.commit()?;
        queue.add_residency_set(&residency);

        let table_desc = M4ArgumentTableDescriptor::new();
        table_desc.set_max_buffer_bind_count(1);

        Ok(Self {
            table: device.new_m4_argument_table_with_descriptor(&table_desc)?,
            event: device.new_shared_event()?,
            device: device.clone(),
            queue,
            command_buffer,
            residency,
            heap,
            arenas,
            allocators,
            retired: Vec::new(),
            frame: 0,
        })
    }

    pub fn resident(&self, allocation: &Allocation) -> Result<(), MetalError> {
        self.residency.add_allocation(allocation)?;
        self.residency.commit()
    }

    pub fn free_later(&mut self, retired: Retired) {
        self.retired.push((self.frame, retired));
    }

    pub fn begin_frame(&mut self) -> Frame<'_> {
        let index = self.frame;
        self.frame += 1;
        if index >= FRAMES_IN_FLIGHT {
            self.event
                .wait_until_signaled_value(index + 1 - FRAMES_IN_FLIGHT, 1000);
        }

        let completed = self.event.signaled_value();
        let heap = &mut self.heap;
        let mut i = 0;
        while i < self.retired.len() {
            if self.retired[i].0 > completed {
                i += 1;
                continue;
            }
            match self.retired.swap_remove(i).1 {
                Retired::Buffer(buffer) => heap.free(buffer.alloc),
                Retired::Texture(texture) => heap.free(texture.alloc),
            }
        }

        let slot = (index % FRAMES_IN_FLIGHT) as usize;
        self.allocators[slot].reset();
        self.arenas[slot].reset();
        self.command_buffer
            .begin_command_buffer_with_allocator(&self.allocators[slot]);

        Frame {
            arena: &self.arenas[slot],
            command_buffer: &self.command_buffer,
            table: &self.table,
            queue: &self.queue,
            event: &self.event,
            value: index + 1,
        }
    }
}

pub struct Frame<'a> {
    pub arena: &'a Arena,
    pub command_buffer: &'a M4CommandBuffer,
    pub table: &'a M4ArgumentTable,
    pub queue: &'a M4CommandQueue,
    pub event: &'a SharedEvent,
    pub value: u64,
}

impl Frame<'_> {
    pub fn raster(&mut self, pass: &Pass, viewport: Viewport) -> Option<Raster<'_>> {
        Raster::new(self.command_buffer, pass, self.table, viewport)
    }

    pub fn compute(&mut self) -> Option<Compute<'_>> {
        Compute::new(self.command_buffer, self.table)
    }

    pub fn submit(self, drawable: Option<&Drawable>) {
        self.command_buffer.end_command_buffer();
        if let Some(drawable) = drawable {
            self.queue.wait_for_drawable(drawable);
        }
        self.queue
            .commit(std::slice::from_ref(self.command_buffer));
        self.queue.signal_event(&self.event.as_event(), self.value);
        if let Some(drawable) = drawable {
            self.queue.signal_drawable(drawable);
            drawable.present();
        }
    }
}
