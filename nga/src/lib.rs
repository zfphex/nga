//! One API over Metal 4 and Vulkan 1.4, following `target/No Graphics API.md`.
//!
//! Everything in this file is backend-independent and written once. The two backend files
//! implement the same set of names against `ash` and `minmetal`, neither of which appears in
//! any public signature — see the contract comment at the top of `vulkan.rs` and `metal.rs`.

use std::cell::Cell;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;

pub mod slang;
pub mod tlsf;

pub use nga_slang::{Constants, gpu_struct};
pub use slang::SLANG_PRELUDE;
pub use tlsf::{Alloc, Tlsf};

#[cfg(not(target_vendor = "apple"))]
mod vulkan;
#[cfg(not(target_vendor = "apple"))]
use vulkan as backend;

#[cfg(target_vendor = "apple")]
mod metal;
#[cfg(target_vendor = "apple")]
use metal as backend;

pub use backend::*;

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

/// A device address with a borrow attached. Eight bytes, `Copy`, embeddable in a struct shared
/// with Slang. Covariant, so a `Gpu<'static, T>` coerces into any shorter frame lifetime; the
/// reverse — storing a frame-arena pointer in a persistent structure — is a compile error.
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

/// The invariant counterpart of [`Gpu`]. `Cell` in the marker is what kills the covariance:
/// a shader writing through this pointer must not be handed a longer-lived alias.
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

/// A host-mapped allocation and its device address. Deref writes go straight into mapped
/// memory; there is no staging step and no flush on either backend.
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

// -- bindless handles --------------------------------------------------------------------

/// A slot in the resource heap, as written into a struct shared with Slang.
///
/// `DescriptorHandle<T>` lowers to a `uint2` on SPIR-V — only the low word is the descriptor
/// index — and to a bare argument-buffer member holding an `MTLResourceID` on Metal. Both are
/// eight bytes at eight-byte alignment, so this one type covers both.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Handle(pub u64);

macro_rules! handle {
    ($($name:ident,)*) => {
        $(
            /// A typed [`Handle`]. The type only picks the Slang spelling the macro emits;
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Dispatch {
    pub threadgroups: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchThreads {
    pub threads: [u32; 3],
    pub group: [u32; 3],
}

// -- shared ABI --------------------------------------------------------------------------

/// Recorded by `#[gpu_struct]`. `fields` carries name, offset and size because the Slang
/// emitter needs the size to work out how much padding to insert after each field.
pub struct TypeAbi {
    pub name: &'static str,
    pub decl: &'static str,
    pub size: usize,
    pub align: usize,
    pub fields: &'static [(&'static str, usize, usize)],
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
    const SLANG_TYPE: &'static str;
}

pub trait Constants {
    fn slang() -> String
    where
        Self: Sized;

    fn values(&self) -> Vec<Constant>;
}

macro_rules! vector {
    ($($name:ident: [$element:ty; $count:literal] align $align:tt = $slang:literal @ $scalar:ident,)*) => {
        $(
            #[repr(C, align($align))]
            #[derive(Clone, Copy, Debug, PartialEq)]
            pub struct $name(pub [$element; $count]);

            impl ConstantValue for $name {
                const SCALAR: Scalar = Scalar::$scalar;
                const SLANG_TYPE: &'static str = $slang;
            }
        )*
    };
}

vector! {
    f32x2: [f32; 2] align 8 = "float2" @ F32x2,
    f32x3: [f32; 3] align 16 = "float3" @ F32x3,
    f32x4: [f32; 4] align 16 = "float4" @ F32x4,
    u32x2: [u32; 2] align 8 = "uint2" @ U32x2,
    u32x3: [u32; 3] align 16 = "uint3" @ U32x3,
    u32x4: [u32; 4] align 16 = "uint4" @ U32x4,
    i32x2: [i32; 2] align 8 = "int2" @ I32x2,
    i32x3: [i32; 3] align 16 = "int3" @ I32x3,
    i32x4: [i32; 4] align 16 = "int4" @ I32x4,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct f32x3x3(pub [f32x3; 3]);

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct f32x4x4(pub [[f32; 4]; 4]);

impl ConstantValue for bool {
    const SCALAR: Scalar = Scalar::Bool;
    const SLANG_TYPE: &'static str = "bool";
}

impl ConstantValue for f32 {
    const SCALAR: Scalar = Scalar::F32;
    const SLANG_TYPE: &'static str = "float";
}

impl ConstantValue for u32 {
    const SCALAR: Scalar = Scalar::U32;
    const SLANG_TYPE: &'static str = "uint";
}

impl ConstantValue for i32 {
    const SCALAR: Scalar = Scalar::I32;
    const SLANG_TYPE: &'static str = "int";
}

impl ConstantValue for u64 {
    const SCALAR: Scalar = Scalar::U64;
    const SLANG_TYPE: &'static str = "uint64_t";
}

impl<T: ?Sized> ConstantValue for Gpu<'static, T> {
    const SCALAR: Scalar = Scalar::U64;
    const SLANG_TYPE: &'static str = "uint64_t";
}

impl<T: ?Sized> ConstantValue for GpuMut<'static, T> {
    const SCALAR: Scalar = Scalar::U64;
    const SLANG_TYPE: &'static str = "uint64_t";
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
/// The shared surface is the intersection; anything here is outside it and has to be asked
/// about before use. `Device::supports` is the only honest way to spell that — a cargo feature
/// would be a compile-time answer to a runtime question, and the answer varies by adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    /// `VK_EXT_mesh_shader` / `MTLRenderPipelineDescriptor` mesh functions.
    MeshShaders,
    /// Metal tile shaders and imageblocks. There is no Vulkan equivalent and there will not
    /// be one — this is the TBDR hardware showing through, not an API gap.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendFactor {
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
/// flush — the cheap case, and the one worth spelling explicitly.
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

/// Everything a raster pipeline needs that is not the shader itself. The depth and cull state
/// is deliberately absent: both backends set those dynamically per draw.
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
    /// the same count — see [`TextureDesc::samples`] — and the resolve target with one.
    pub fn samples(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }
}

/// How a multisample depth attachment collapses to its single-sample resolve target.
///
/// Colour has no equivalent knob: both APIs average, and Vulkan only guarantees `AVERAGE` for
/// colour anyway. Depth is the opposite — averaging a depth buffer is meaningless, so this is
/// the one place a mode has to be named. `SampleZero` is the only mode both APIs always
/// support — `Min`/`Max` are optional in `depthResolveModes` and validation is what catches a
/// driver that lacks them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveMode {
    SampleZero,
    Min,
    Max,
}

/// A mesh pipeline: an optional object (amplification) stage, a mesh stage, and a fragment
/// stage. There are no vertices and no index buffer — the mesh stage emits primitives directly.
///
/// Metal-only today; ask [`Device::supports`] with [`Feature::MeshShaders`] before building one.
/// The shader side is portable Slang either way (`[shader("amplification")]` / `[shader("mesh")]`),
/// so only the pipeline is backend-specific.
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
/// This one is Apple-only for good — it is the tile-based deferred hardware showing through,
/// and there is no Vulkan equivalent to grow into. Slang has no tile stage either, so the
/// function has to come from an MSL source; see `shaders/tbdr.metal`.
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

#[derive(Clone, Copy, Debug)]
pub struct DepthState {
    pub compare: CompareOp,
    pub write: bool,
}

impl DepthState {
    pub fn new(compare: CompareOp, write: bool) -> Self {
        Self { compare, write }
    }

    /// Metal folds "no test" into `CompareOp::Always`; Vulkan has a separate enable bit, and
    /// leaving it on with `ALWAYS` still costs a depth write when `write` is set.
    pub fn test(&self) -> bool {
        self.compare != CompareOp::Always || self.write
    }
}

// -- pool --------------------------------------------------------------------------------

/// A fixed-capacity slab inside an [`Arena`], with a free list threaded through the unused
/// slots. Indices are stable, so a shader can hold one across frames; the arena is not reset
/// while a pool lives in it.
pub struct Pool<'a, T> {
    slots: NonNull<T>,
    base: u64,
    cap: u32,
    len: u32,
    free: Vec<u32>,
    marker: PhantomData<&'a mut [T]>,
}

impl<'a, T> Pool<'a, T> {
    pub fn new(arena: &'a Arena, cap: u32) -> Option<Self> {
        let (cpu, gpu) = arena.alloc_bytes(size_of::<T>() * cap as usize, align_of::<T>())?;
        Some(Self {
            slots: NonNull::new(cpu as *mut T)?,
            base: gpu,
            cap,
            len: 0,
            free: Vec::new(),
            marker: PhantomData,
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
        unsafe { self.slots.add(index as usize).write(value) };
        Some(index)
    }

    pub fn free(&mut self, index: u32) {
        debug_assert!(index < self.len, "freeing a slot that was never allocated");
        self.free.push(index);
    }

    pub fn gpu(&self, index: u32) -> Gpu<'a, T> {
        debug_assert!(index < self.len, "slot {index} was never allocated");
        Gpu::new(self.base + index as u64 * size_of::<T>() as u64)
    }
}

impl<T> Index<u32> for Pool<'_, T> {
    type Output = T;
    fn index(&self, index: u32) -> &T {
        debug_assert!(index < self.len, "slot {index} was never allocated");
        unsafe { self.slots.add(index as usize).as_ref() }
    }
}

impl<T> IndexMut<u32> for Pool<'_, T> {
    fn index_mut(&mut self, index: u32) -> &mut T {
        debug_assert!(index < self.len, "slot {index} was never allocated");
        unsafe { self.slots.add(index as usize).as_mut() }
    }
}

/// The compiled shader for `name`, as bytes. SPIR-V on the Vulkan backend, MSL source on the
/// Metal one — both land in `OUT_DIR` from the same `.slang` file, and this is the only place
/// the extension is spelled.
#[cfg(not(target_vendor = "apple"))]
#[macro_export]
macro_rules! shader {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")) as &[u8]
    };
}

#[cfg(target_vendor = "apple")]
#[macro_export]
macro_rules! shader {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metal")) as &[u8]
    };
}

pub const FRAMES_IN_FLIGHT: u64 = 2;
