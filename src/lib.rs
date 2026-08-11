#![allow(non_camel_case_types)]

pub mod tlsf;

use minmetal::*;
use std::cell::Cell;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;
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

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct f32x4x4(pub [[f32; 4]; 4]);

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
