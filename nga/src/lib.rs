//! One API over Metal 4 and Vulkan 1.4, following `target/No Graphics API.md`.
//!
//! Everything in this file is backend-independent and written once. The two backend files
//! implement the same set of names against `ash` and `minmetal`, neither of which appears in
//! any public signature. See the contract comment at the top of `vulkan.rs` and `metal.rs`.

use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;

pub mod ir;
pub mod msl;
pub mod spirv;
pub mod tlsf;
pub mod vector;

pub use nga_gpu::{Constants, gpu_mod};
pub use tlsf::*;
pub use vector::*;

#[cfg(not(target_vendor = "apple"))]
mod vulkan;

#[cfg(not(target_vendor = "apple"))]
use vulkan as backend;

#[cfg(target_vendor = "apple")]
mod metal;
#[cfg(target_vendor = "apple")]
use metal as backend;

pub use backend::*;

/// Writes both targets to `dir` as `name.spv` and `name.metal`, and answers with the paths.
///
/// This is what an ahead-of-time build runs: a bin target in the workspace links the crate its
/// shaders live in, calls this, and an Apple build can compile the generated `.metal` to a
/// `.metallib`. Both files are produced on every host, which also makes the Metal output
/// checkable from a machine that cannot build for Apple.
pub fn dump(
    dir: impl AsRef<std::path::Path>,
    name: &str,
    entries: &[&'static ir::GpuFn],
) -> std::io::Result<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let spirv = dir.join(format!("{name}.spv"));
    let metal = dir.join(format!("{name}.metal"));
    std::fs::write(&spirv, spirv::module(entries))?;
    std::fs::write(&metal, msl::module(entries))?;
    Ok((spirv, metal))
}

/// One error type for both backends. `VkResult` and `NSError` are stringified at the boundary
/// rather than escaping, because neither is meaningful to a caller that cannot name it.
#[derive(Debug)]
pub struct Error(pub String);

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

// -- device pointers ---------------------------------------------------------------------

/// A device address. Eight bytes, `Copy`, and a legal field of a struct a shader shares.
///
/// `Gpu<T>` names the pointee for the shader ABI. [`HeapBuffer::gpu`] returns one, and
/// [`GpuHeap::free_buffer`] can reclaim that allocation on a later line. The host object owns
/// the memory. [`GpuHeap::free_later`] keeps it alive until the GPU retires the frame.
#[repr(transparent)]
#[derive(Debug)]
pub struct Gpu<T: ?Sized>(pub(crate) NonZeroU64, PhantomData<fn() -> *const T>);

pub(crate) fn device_ptr<T: ?Sized>(address: u64) -> Gpu<T> {
    unsafe { Gpu::new(address) }
}

pub(crate) fn device_ptr_rw<T: ?Sized>(address: u64) -> GpuRw<T> {
    unsafe { GpuRw::new(address) }
}

impl<T: ?Sized> Gpu<T> {
    /// Wrap a device address.
    ///
    /// # Safety
    ///
    /// `address` must be a live device pointer of type `T`, or used only as an opaque token.
    /// Host code must not dereference it: [`Deref`] panics rather than manufacture a reference.
    pub unsafe fn new(address: u64) -> Self {
        Self(NonZeroU64::new(address).unwrap(), PhantomData)
    }

    pub fn get(self) -> u64 {
        self.0.get()
    }

    /// The same address, marked for GPU writes. See [`GpuRw`].
    pub fn as_rw(self) -> GpuRw<T> {
        GpuRw(self.0, PhantomData)
    }
}

impl<T> Gpu<[T]> {
    pub fn elements(self) -> Gpu<T> {
        Gpu(self.0, PhantomData)
    }
}

impl<T: ?Sized> Clone for Gpu<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Gpu<T> {}

/// The same device address as [`Gpu`], for a shader that writes through it.
///
/// The value is `Copy`. Two `GpuRw<T>` can name one allocation, including from different
/// threads. Barriers on the command recorder order those writes.
#[repr(transparent)]
#[derive(Debug)]
pub struct GpuRw<T: ?Sized>(pub(crate) NonZeroU64, PhantomData<fn(*mut T)>);

impl<T: ?Sized> GpuRw<T> {
    /// Wrap a device address for shader writes.
    ///
    /// # Safety
    ///
    /// Same contract as [`Gpu::new`]. [`DerefMut`] panics on the host, so copying a `GpuRw`
    /// cannot produce aliased `&mut T` values.
    pub unsafe fn new(address: u64) -> Self {
        Self(NonZeroU64::new(address).unwrap(), PhantomData)
    }

    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl<T> GpuRw<[T]> {
    pub fn elements(self) -> GpuRw<T> {
        GpuRw(self.0, PhantomData)
    }
}

impl<T: ?Sized> Clone for GpuRw<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for GpuRw<T> {}

// A shader body reads and writes through these exactly as it would through a reference. The
// host implementations only make that ordinary Rust type-check. They must not manufacture a
// reference from a device address: shader functions are still callable host functions, and a
// safe call must panic rather than invoke undefined behaviour.

#[cold]
#[track_caller]
fn shader_pointer_on_host() -> ! {
    panic!("a GPU device pointer cannot be dereferenced by host code")
}

impl<T> Deref for Gpu<T> {
    type Target = T;
    fn deref(&self) -> &T {
        shader_pointer_on_host()
    }
}

impl<T> Deref for GpuRw<T> {
    type Target = T;
    fn deref(&self) -> &T {
        shader_pointer_on_host()
    }
}

impl<T> DerefMut for GpuRw<T> {
    fn deref_mut(&mut self) -> &mut T {
        shader_pointer_on_host()
    }
}

impl<T> Index<u32> for Gpu<[T]> {
    type Output = T;
    fn index(&self, index: u32) -> &T {
        let _ = index;
        shader_pointer_on_host()
    }
}

impl<T> Index<u32> for GpuRw<[T]> {
    type Output = T;
    fn index(&self, index: u32) -> &T {
        let _ = index;
        shader_pointer_on_host()
    }
}

impl<T> IndexMut<u32> for GpuRw<[T]> {
    fn index_mut(&mut self, index: u32) -> &mut T {
        let _ = index;
        shader_pointer_on_host()
    }
}

impl AtomicU32 {
    pub fn atomic_add(&mut self, value: u32) -> u32 {
        let previous = self.0;
        self.0 = previous.wrapping_add(value);
        previous
    }

    pub fn atomic_min(&mut self, value: u32) -> u32 {
        let previous = self.0;
        self.0 = previous.min(value);
        previous
    }

    pub fn atomic_max(&mut self, value: u32) -> u32 {
        let previous = self.0;
        self.0 = previous.max(value);
        previous
    }

    pub fn atomic_exchange(&mut self, value: u32) -> u32 {
        std::mem::replace(&mut self.0, value)
    }
}

impl ThreadgroupAtomicU32 {
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    pub fn atomic_add(&mut self, value: u32) -> u32 {
        let previous = self.0;
        self.0 = previous.wrapping_add(value);
        previous
    }

    pub fn atomic_exchange(&mut self, value: u32) -> u32 {
        std::mem::replace(&mut self.0, value)
    }
}

// Stage-specific shader operations. Their host bodies only make shader functions type-check;
// `msl` replaces every call with the corresponding Metal operation.
pub fn dfdx<T: Copy>(value: T) -> T {
    value
}

pub fn dfdy<T: Copy>(value: T) -> T {
    value
}

pub fn mesh_set_threadgroups(_groups: u32x3) {}

pub fn mesh_set_primitive_count(_count: u32) {}

pub fn mesh_set_vertex<T: Copy>(_index: u32, _vertex: T) {}

pub fn mesh_set_index(_index: u32, _vertex: u32) {}

pub fn threadgroup_barrier() {}

pub fn tile_read<T: Copy + Default>(_at: u32x2) -> T {
    T::default()
}

pub fn tile_write<T: Copy>(_value: T, _at: u32x2) {}

pub fn command_set_pipeline(_commands: Commands, _slot: u32, _pipeline: PipelineHandle) {}

pub fn command_set_vertex_buffer<T: Copy>(_commands: Commands, _slot: u32, _value: T, _index: u32) {
}

pub fn command_draw(
    _commands: Commands,
    _slot: u32,
    _lines: bool,
    _vertex_start: u32,
    _vertex_count: u32,
    _instance_count: u32,
    _base_instance: u32,
) {
}

/// A host-mapped allocation and its device address. Deref writes go straight into mapped
/// memory; there is no staging step and no flush on either backend.
///
/// `'a` is the host mapping. [`GpuBox::gpu`] copies out as a bare [`Gpu`].
pub struct GpuBox<'a, T: ?Sized> {
    pub cpu: NonNull<T>,
    pub gpu: Gpu<T>,
    _mapped: PhantomData<&'a mut T>,
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

// -- bindless handles --------------------------------------------------------------------

/// A slot in the resource heap, as written into a struct a shader shares.
///
/// `DescriptorHandle<T>` lowers to a `uint2` on SPIR-V, where only the low word is the
/// descriptor index, and to a bare argument-buffer member holding an `MTLResourceID` on
/// Metal. Both are eight bytes at eight-byte alignment, so this one type covers both.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Handle(pub u64);

macro_rules! handle {
    ($($name:ident,)*) => {
        $(
            /// A typed [`Handle`]. The type only picks which resource the emitters declare;
            /// the bytes are identical.
            #[repr(transparent)]
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
            pub struct $name(pub Handle);
        )*
    };
}

handle! {
    Texture2d,
    Texture2dW,
    Texture2dRW,
    Texture2dArray,
    Texture3d,
    Texture3dW,
    Texture3dRW,
    TextureCube,
    Depth2d,
    Sampler,
    // The last two are Metal-only: an `IndirectCommands` buffer, and a `Pipeline` built with
    // `indirect`, both named by a shader that encodes commands. Zero on Vulkan.
    Commands,
    PipelineHandle,
}

/// `atomic_uint` on Metal, `uint` under a SPIR-V atomic on Vulkan. Shaders atomically update
/// it through a device pointer; the host only ever reads or clears it.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AtomicU32(pub u32);

/// An atomic local shared by the invocations in one shader threadgroup.
///
/// This type is valid only for a local variable in a `#[gpu_mod]` function. The MSL emitter
/// declares it in the `threadgroup` address space and ignores its Rust initializer; initialize
/// it from one lane and call [`threadgroup_barrier`] before other lanes use it.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ThreadgroupAtomicU32(pub u32);

// -- indirect argument structs -----------------------------------------------------------

/// Matches `nga_draw` in the prelude and `VkDrawIndirectCommand` / `MTLDrawPrimitivesIndirectArguments`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Draw {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub vertex_start: u32,
    pub base_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DrawIndexed {
    pub index_count: u32,
    pub instance_count: u32,
    pub index_start: u32,
    pub base_vertex: i32,
    pub base_instance: u32,
}

/// Indirect dispatch: three group counts, matching `VkDispatchIndirectCommand`.
///
/// A direct launch is a [`DispatchThreads`], built with [`Dispatch::x`] or [`Dispatch::xy`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Dispatch {
    pub threadgroups: [u32; 3],
}

impl Dispatch {
    /// `threads` invocations along X, in groups of `group`.
    pub fn x(threads: usize, group: usize) -> DispatchThreads {
        debug_assert!(
            group > 0 && threads <= u32::MAX as usize && group <= u32::MAX as usize,
            "dispatch dimensions do not fit in u32"
        );
        DispatchThreads {
            threads: [threads as u32, 1, 1],
            group: [group as u32, 1, 1],
        }
    }

    /// `width` by `height` invocations, in square groups of `group`.
    pub fn xy(width: u32, height: u32, group: u32) -> DispatchThreads {
        debug_assert!(group > 0, "a dispatch group has size zero");
        DispatchThreads {
            threads: [width, height, 1],
            group: [group, group, 1],
        }
    }
}

/// How many threads a compute launch runs, and the size of each group.
///
/// [`Compute::dispatch`] takes one. A shader that writes an indirect dispatch-threads command
/// uses the same layout. [`Dispatch`] is the other indirect layout: group counts, no group size.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchThreads {
    pub threads: [u32; 3],
    pub group: [u32; 3],
}

impl DispatchThreads {
    fn grid(self) -> ([u32; 3], [u32; 3]) {
        debug_assert!(
            self.group[0] > 0 && self.group[1] > 0 && self.group[2] > 0,
            "a dispatch group has size zero"
        );
        (self.threads, self.group)
    }
}

// -- shared ABI --------------------------------------------------------------------------

/// Recorded by `#[gpu_mod]`.
#[derive(PartialEq, Debug)]
pub struct TypeAbi {
    pub name: &'static str,
    pub size: usize,
    pub align: usize,
    pub fields: &'static [FieldAbi],
}

/// `size` is here because the MSL emitter needs it to work out how much padding follows a
/// field; `ty` is here because both emitters need to know what the field is.
#[derive(PartialEq, Debug)]
pub struct FieldAbi {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
    pub ty: ir::Ty,
}

pub trait DeviceAbi: Copy {
    const ABI: TypeAbi;
}

/// The scalar kinds a specialization constant can take. Vulkan needs the byte width (`bool`
/// is four bytes in a SPIR-V specialization block, not one); Metal needs an `MTLDataType`.
/// Neither is spellable here, so both backends match on this instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    Bool,
    F16,
    F32,
    U32,
    I32,
    U64,
    F16x2,
    F16x3,
    F16x4,
    F32x2,
    F32x3,
    F32x4,
    U32x2,
    U32x3,
    U32x4,
    I32x2,
    I32x3,
    I32x4,
}

/// One specialization constant, ready for either backend. Sixteen bytes covers every
/// [`Scalar`], so this stays `Copy` and the payload never allocates.
#[derive(Clone, Copy, Debug)]
pub struct Constant {
    pub index: u32,
    pub scalar: Scalar,
    pub bytes: [u8; 16],
    pub len: u8,
}

impl Constant {
    pub fn push<T: ConstantValue>(out: &mut Vec<Constant>, index: u32, value: T) {
        let mut bytes = [0u8; 16];
        let source =
            unsafe { std::slice::from_raw_parts(&value as *const T as *const u8, size_of::<T>()) };
        bytes[..source.len()].copy_from_slice(source);
        out.push(Constant {
            index,
            scalar: T::SCALAR,
            bytes,
            len: source.len() as u8,
        });
    }
}

pub trait ConstantValue: Copy {
    const SCALAR: Scalar;
}

pub trait Constants {
    fn values(&self) -> Vec<Constant>;
}

impl ConstantValue for bool {
    const SCALAR: Scalar = Scalar::Bool;
}

impl ConstantValue for f32 {
    const SCALAR: Scalar = Scalar::F32;
}

impl ConstantValue for u32 {
    const SCALAR: Scalar = Scalar::U32;
}

impl ConstantValue for i32 {
    const SCALAR: Scalar = Scalar::I32;
}

impl ConstantValue for u64 {
    const SCALAR: Scalar = Scalar::U64;
}

impl<T: ?Sized> ConstantValue for Gpu<T> {
    const SCALAR: Scalar = Scalar::U64;
}

impl<T: ?Sized> ConstantValue for GpuRw<T> {
    const SCALAR: Scalar = Scalar::U64;
}

// -- enums -------------------------------------------------------------------------------
//
// Every variant here exists on both backends. Growth is on demand: adding one means adding a
// row to the `match` in each backend file, and the compiler points at both.

/// Where an allocation lives. Vulkan picks a memory type; Metal picks a storage mode, which on
/// unified memory collapses to `Shared` for everything but `Device`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {
    /// Fast for the GPU, not host-visible on a discrete Vulkan device. Writes go through the
    /// heap's staging path.
    Device,
    /// Host-visible and device-readable. Root data, indirect arguments, staging.
    Upload,
    /// Host-visible and cached for reading back what the GPU wrote.
    Readback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgba16Float,
    Rgba32Float,
    Rg11B10Float,
    Rgb10A2Unorm,
    Rg8Unorm,
    Rg16Float,
    R8Unorm,
    R16Float,
    R32Float,
    R32Uint,
    Rg32Uint,
    Rgba32Uint,
    D32Float,
    D32FloatS8,
    D16Unorm,
}

impl Format {
    /// Bytes per texel. Block-compressed formats are not in the enum yet, so this is exact.
    pub fn size(self) -> usize {
        match self {
            Format::R8Unorm => 1,
            Format::Rg8Unorm | Format::R16Float | Format::D16Unorm => 2,
            Format::Rgba8Unorm
            | Format::Rgba8UnormSrgb
            | Format::Bgra8Unorm
            | Format::Bgra8UnormSrgb
            | Format::Rg11B10Float
            | Format::Rgb10A2Unorm
            | Format::Rg16Float
            | Format::R32Float
            | Format::R32Uint
            | Format::D32Float => 4,
            Format::D32FloatS8 => 5,
            Format::Rgba16Float | Format::Rg32Uint => 8,
            Format::Rgba32Float | Format::Rgba32Uint => 16,
        }
    }

    pub fn is_depth(self) -> bool {
        matches!(
            self,
            Format::D32Float | Format::D32FloatS8 | Format::D16Unorm
        )
    }

    pub fn has_stencil(self) -> bool {
        matches!(self, Format::D32FloatS8)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage(pub u32);

impl Usage {
    pub const SAMPLED: Usage = Usage(0x1);
    pub const STORAGE: Usage = Usage(0x2);
    pub const COLOR: Usage = Usage(0x4);
    pub const DEPTH_STENCIL: Usage = Usage(0x8);
    pub const COPY_SRC: Usage = Usage(0x10);
    pub const COPY_DST: Usage = Usage(0x20);

    pub fn has(self, other: Usage) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for Usage {
    type Output = Usage;
    fn bitor(self, rhs: Usage) -> Usage {
        Usage(self.0 | rhs.0)
    }
}

/// Capabilities that exist on one backend and not necessarily the other.
///
/// The shared API is the intersection; anything here is outside it and has to be asked
/// about before use. `Device::supports` is the only honest way to spell that. A cargo feature
/// would be a compile-time answer to a runtime question, and the answer varies by adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    /// `VK_EXT_mesh_shader` / `MTLRenderPipelineDescriptor` mesh functions.
    MeshShaders,
    /// Metal tile shaders and imageblocks. There is no Vulkan equivalent and there will not
    /// be one, because this is the TBDR hardware showing through, not an API gap.
    TileShaders,
    /// Metal indirect command buffers. Vulkan reaches most of the same ground through
    /// `vkCmdDrawIndirectCount` and `VK_EXT_multi_draw`, which are not the same shape.
    IndirectCommands,
    /// Timestamp queries around a pass.
    Timestamps,
    /// Occlusion queries.
    Occlusion,
    /// Runs of draws issued in one call: `VK_EXT_multi_draw`.
    MultiDraw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

/// Everything a sampler is. Metal bakes the equivalent into an `MTLSamplerDescriptor` and
/// Vulkan into a `VkSampler`; both are heap resources addressed by a [`Sampler`] handle.
#[derive(Clone, Copy, Debug)]
pub struct SamplerDesc {
    pub min: Filter,
    pub mag: Filter,
    pub mip: Filter,
    pub address: AddressMode,
    pub anisotropy: u32,
    /// Set for a shadow sampler; `None` for an ordinary one.
    pub compare: Option<CompareOp>,
}

impl SamplerDesc {
    pub fn linear() -> Self {
        Self {
            min: Filter::Linear,
            mag: Filter::Linear,
            mip: Filter::Linear,
            address: AddressMode::ClampToEdge,
            anisotropy: 1,
            compare: None,
        }
    }

    pub fn nearest() -> Self {
        Self {
            min: Filter::Nearest,
            mag: Filter::Nearest,
            mip: Filter::Nearest,
            address: AddressMode::ClampToEdge,
            anisotropy: 1,
            compare: None,
        }
    }

    pub fn address(mut self, address: AddressMode) -> Self {
        self.address = address;
        self
    }

    pub fn anisotropy(mut self, samples: u32) -> Self {
        self.anisotropy = samples;
        self
    }

    pub fn compare(mut self, op: CompareOp) -> Self {
        self.compare = Some(op);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    TriangleList,
    TriangleStrip,
    LineList,
    LineStrip,
    PointList,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cull {
    None,
    Front,
    Back,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Winding {
    Clockwise,
    CounterClockwise,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstColor,
    OneMinusDstColor,
    DstAlpha,
    OneMinusDstAlpha,
}

/// Pipeline stages, as a mask. The names follow the blog post rather than either backend:
/// these are the points a barrier can sit between, not a full enumeration of hardware stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage(pub u32);

impl Stage {
    pub const NONE: Stage = Stage(0);
    pub const TRANSFER: Stage = Stage(0x1);
    pub const COMPUTE: Stage = Stage(0x2);
    pub const VERTEX: Stage = Stage(0x4);
    pub const PIXEL: Stage = Stage(0x8);
    pub const COLOR_OUT: Stage = Stage(0x10);
    pub const DEPTH_STENCIL: Stage = Stage(0x20);
    pub const INDIRECT: Stage = Stage(0x40);
    pub const ALL: Stage = Stage(0x7f);

    pub fn has(self, other: Stage) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for Stage {
    type Output = Stage;
    fn bitor(self, rhs: Stage) -> Stage {
        Stage(self.0 | rhs.0)
    }
}

/// What a barrier has to make visible. Empty means execution ordering only, with no cache
/// flush. That is the cheap case, and the one worth spelling explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hazard(pub u32);

impl Hazard {
    pub const NONE: Hazard = Hazard(0);
    pub const BUFFERS: Hazard = Hazard(0x1);
    pub const TEXTURES: Hazard = Hazard(0x2);
    pub const DRAW_ARGUMENTS: Hazard = Hazard(0x4);
    pub const DESCRIPTORS: Hazard = Hazard(0x8);
    pub const DEPTH_STENCIL: Hazard = Hazard(0x10);
    pub const ALL: Hazard = Hazard(0x1f);

    pub fn has(self, other: Hazard) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for Hazard {
    type Output = Hazard;
    fn bitor(self, rhs: Hazard) -> Hazard {
        Hazard(self.0 | rhs.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Viewport {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Scissor {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Index buffers are `u16` or `u32` and nothing else on either backend.
pub trait IndexFormat: Copy {
    const WIDE: bool;
}

impl IndexFormat for u16 {
    const WIDE: bool = false;
}

impl IndexFormat for u32 {
    const WIDE: bool = true;
}

// -- pipeline description ----------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    Opaque,
    Alpha,
    Premultiplied,
    Additive,
    Screen,
}

#[derive(Clone, Copy, Debug)]
pub struct ColorTarget {
    pub format: Format,
    pub blend: BlendMode,
    pub write_mask: u8,
}

impl ColorTarget {
    pub fn new(format: Format) -> Self {
        Self {
            format,
            blend: BlendMode::Opaque,
            write_mask: 0xf,
        }
    }

    pub fn blend(mut self, blend: BlendMode) -> Self {
        self.blend = blend;
        self
    }
}

/// Everything a raster pipeline needs that is not the shader itself. The depth, stencil and
/// cull state is deliberately absent: both backends set those dynamically per draw.
#[derive(Clone, Copy)]
pub struct RasterDesc<'a> {
    pub vertex: &'a str,
    pub fragment: &'a str,
    pub colors: &'a [ColorTarget],
    pub depth: Option<Format>,
    pub topology: Topology,
    pub samples: u32,
    pub constants: Option<&'a dyn Constants>,
    /// Whether a shader may encode this pipeline into an [`IndirectCommands`] buffer. Costs
    /// nothing to leave off and is refused by Metal if a shader tries it anyway; ignored on
    /// Vulkan, which has no such buffer.
    pub indirect: bool,
}

impl<'a> RasterDesc<'a> {
    pub fn new(vertex: &'a str, fragment: &'a str, colors: &'a [ColorTarget]) -> Self {
        Self {
            vertex,
            fragment,
            colors,
            depth: None,
            topology: Topology::TriangleList,
            samples: 1,
            constants: None,
            indirect: false,
        }
    }

    pub fn indirect(mut self) -> Self {
        self.indirect = true;
        self
    }

    pub fn depth(mut self, format: Format) -> Self {
        self.depth = Some(format);
        self
    }

    pub fn topology(mut self, topology: Topology) -> Self {
        self.topology = topology;
        self
    }

    /// Rasterisation sample count. Every attachment the pass binds has to have been built with
    /// the same count, see [`TextureDesc::samples`], and the resolve target with one.
    pub fn samples(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }
}

#[derive(Clone, Copy)]
pub struct VertexShader(pub &'static ir::GpuFn);

#[derive(Clone, Copy)]
pub struct FragmentShader(pub &'static ir::GpuFn);

#[derive(Clone, Copy)]
pub struct ComputeShader(pub &'static ir::GpuFn);

#[derive(Clone, Copy)]
pub struct ObjectShader(pub &'static ir::GpuFn);

#[derive(Clone, Copy)]
pub struct MeshShader(pub &'static ir::GpuFn);

#[derive(Clone, Copy)]
pub struct TileShader(pub &'static ir::GpuFn);

pub struct RasterBuilder<'a> {
    pub(crate) shaders: &'a mut Shaders,
    pub vertex: &'a str,
    pub fragment: &'a str,
    pub colors: [ColorTarget; 4],
    pub color_count: usize,
    pub depth: Option<Format>,
    pub topology: Topology,
    pub samples: u32,
    pub constants: Option<&'a dyn Constants>,
    pub indirect: bool,
}

impl<'a> RasterBuilder<'a> {
    pub fn color(mut self, format: Format) -> Self {
        self.colors[self.color_count] = ColorTarget::new(format);
        self.color_count += 1;
        self
    }

    pub fn color_target(mut self, target: ColorTarget) -> Self {
        self.colors[self.color_count] = target;
        self.color_count += 1;
        self
    }

    pub fn blend(mut self, blend: BlendMode) -> Self {
        self.colors[self.color_count - 1].blend = blend;
        self
    }

    pub fn depth(mut self, format: Format) -> Self {
        self.depth = Some(format);
        self
    }

    pub fn topology(mut self, topology: Topology) -> Self {
        self.topology = topology;
        self
    }

    pub fn samples(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }

    pub fn constants(mut self, constants: &'a dyn Constants) -> Self {
        self.constants = Some(constants);
        self
    }

    pub fn indirect(mut self) -> Self {
        self.indirect = true;
        self
    }

    pub fn build(self) -> Result<Pipeline, Error> {
        let shaders = self.shaders;
        shaders.build_raster(&RasterDesc {
            vertex: self.vertex,
            fragment: self.fragment,
            colors: &self.colors[..self.color_count],
            depth: self.depth,
            topology: self.topology,
            samples: self.samples,
            constants: self.constants,
            indirect: self.indirect,
        })
    }
}

/// How a multisample depth attachment collapses to its single-sample resolve target.
///
/// Colour has no equivalent knob: both APIs average, and Vulkan only guarantees `AVERAGE` for
/// colour anyway. Depth is the opposite. Averaging a depth buffer is meaningless, so this is
/// the one place a mode has to be named. `SampleZero` is the only mode both APIs always
/// support. `Min`/`Max` are optional in `depthResolveModes`, and validation is what catches a
/// driver that lacks them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveMode {
    SampleZero,
    Min,
    Max,
}

/// A mesh pipeline: an optional object (amplification) stage, a mesh stage, and a fragment
/// stage. There are no vertices and no index buffer; the mesh stage emits primitives directly.
///
/// Metal-only today; ask [`Device::supports`] with [`Feature::MeshShaders`] before building one.
#[derive(Clone, Copy)]
pub struct MeshDesc<'a> {
    /// The object stage, or `""` for a mesh pipeline with no amplification.
    pub object: &'a str,
    pub mesh: &'a str,
    pub fragment: &'a str,
    pub colors: &'a [ColorTarget],
    /// Bytes of payload the object stage hands to the mesh stage. Ignored without an object
    /// stage.
    pub payload: usize,
    pub samples: u32,
    pub constants: Option<&'a dyn Constants>,
    /// See [`RasterDesc::indirect`].
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
            constants: None,
            indirect: false,
        }
    }

    pub fn indirect(mut self) -> Self {
        self.indirect = true;
        self
    }
}

/// A tile pipeline: a compute function dispatched inside a render pass, reading and writing the
/// imageblock the pass is rendering into without a round trip to memory.
///
/// This one is Apple-only for good. It is the tile-based deferred hardware showing through,
/// and there is no Vulkan equivalent to grow into.
#[derive(Clone, Copy)]
pub struct TileDesc<'a> {
    pub function: &'a str,
    pub colors: &'a [Format],
    pub samples: u32,
    pub constants: Option<&'a dyn Constants>,
}

impl<'a> TileDesc<'a> {
    pub fn new(function: &'a str, colors: &'a [Format]) -> Self {
        Self {
            function,
            colors,
            samples: 1,
            constants: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFace {
    pub compare: CompareOp,
    pub fail: StencilOp,
    pub depth_fail: StencilOp,
    pub pass: StencilOp,
    pub read_mask: u32,
    pub write_mask: u32,
}

impl StencilFace {
    pub fn new(compare: CompareOp) -> Self {
        Self {
            compare,
            fail: StencilOp::Keep,
            depth_fail: StencilOp::Keep,
            pass: StencilOp::Keep,
            read_mask: u32::MAX,
            write_mask: u32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilState {
    pub front: StencilFace,
    pub back: StencilFace,
}

impl StencilState {
    pub fn new(front: StencilFace, back: StencilFace) -> Self {
        Self { front, back }
    }

    pub fn symmetric(face: StencilFace) -> Self {
        Self {
            front: face,
            back: face,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthState {
    pub compare: CompareOp,
    pub write: bool,
    pub stencil: Option<StencilState>,
}

impl DepthState {
    pub fn new(compare: CompareOp, write: bool) -> Self {
        Self {
            compare,
            write,
            stencil: None,
        }
    }

    pub fn stencil(mut self, stencil: StencilState) -> Self {
        self.stencil = Some(stencil);
        self
    }

    /// Metal folds "no test" into `CompareOp::Always`; Vulkan has a separate enable bit, and
    /// leaving it on with `ALWAYS` still costs a depth write when `write` is set.
    pub fn test(&self) -> bool {
        self.compare != CompareOp::Always || self.write
    }
}

pub const FRAMES_IN_FLIGHT: u64 = 2;
