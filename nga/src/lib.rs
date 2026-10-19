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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapUsage {
    pub capacity: u64,
    pub allocated: u64,
    pub peak: u64,
    pub retired: u64,
    pub allocator_metadata: usize,
}

impl Timers {
    pub fn checked_ns(&self, start: usize, end: usize) -> Option<f64> {
        let (&start_tick, &end_tick) = (self.ticks.get(start)?, self.ticks.get(end)?);
        if !self.written(start) || !self.written(end) || end_tick < start_tick {
            return None;
        }
        Some((end_tick - start_tick) as f64 * self.ns_per_tick)
    }

    pub fn checked_ms(&self, start: usize, end: usize) -> Option<f64> {
        Some(self.checked_ns(start, end)? / 1e6)
    }
}

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

    pub unsafe fn new(address: u64) -> Self {
        Self(NonZeroU64::new(address).unwrap(), PhantomData)
    }

    pub fn get(self) -> u64 {
        self.0.get()
    }

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

#[repr(transparent)]
#[derive(Debug)]
pub struct GpuRw<T: ?Sized>(pub(crate) NonZeroU64, PhantomData<fn(*mut T)>);

impl<T: ?Sized> GpuRw<T> {

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

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Handle(pub u64);

macro_rules! handle {
    ($($name:ident,)*) => {
        $(

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
    TextureCubeRW,
    Depth2d,
    Sampler,

    Commands,
    PipelineHandle,
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AtomicU32(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ThreadgroupAtomicU32(pub u32);

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

impl Dispatch {
    pub fn x(threads: usize) -> DispatchThreads {
        assert!(
            threads <= u32::MAX as usize,
            "dispatch dimensions do not fit in u32"
        );
        DispatchThreads {
            threads: [threads as u32, 1, 1],
        }
    }

    pub fn xy(width: u32, height: u32) -> DispatchThreads {
        DispatchThreads {
            threads: [width, height, 1],
        }
    }

    pub fn xyz(x: u32, y: u32, z: u32) -> DispatchThreads {
        DispatchThreads {
            threads: [x, y, z],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchThreads {
    pub threads: [u32; 3],
}

#[derive(PartialEq, Debug)]
pub struct TypeAbi {
    pub name: &'static str,
    pub size: usize,
    pub align: usize,
    pub fields: &'static [FieldAbi],
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {

    Device,

    Upload,

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
    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc4RSnorm,
    Bc5RgUnorm,
    Bc5RgSnorm,
    Bc6hRgbFloat,
    Bc6hRgbUfloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,
    Astc4x4Unorm,
    Astc4x4UnormSrgb,
}

impl Format {
    pub fn size(self) -> usize {
        assert!(
            !self.is_compressed(),
            "compressed formats have no whole-byte texel size"
        );
        self.block_size()
    }

    pub fn block_extent(self) -> (u32, u32) {
        match self {
            Format::Bc1RgbaUnorm
            | Format::Bc1RgbaUnormSrgb
            | Format::Bc2RgbaUnorm
            | Format::Bc2RgbaUnormSrgb
            | Format::Bc3RgbaUnorm
            | Format::Bc3RgbaUnormSrgb
            | Format::Bc4RUnorm
            | Format::Bc4RSnorm
            | Format::Bc5RgUnorm
            | Format::Bc5RgSnorm
            | Format::Bc6hRgbFloat
            | Format::Bc6hRgbUfloat
            | Format::Bc7RgbaUnorm
            | Format::Bc7RgbaUnormSrgb
            | Format::Astc4x4Unorm
            | Format::Astc4x4UnormSrgb => (4, 4),
            _ => (1, 1),
        }
    }

    pub fn is_compressed(self) -> bool {
        self.block_extent() != (1, 1)
    }

    pub fn row_size(self, width: u32) -> usize {
        width.div_ceil(self.block_extent().0) as usize * self.block_size()
    }

    pub fn image_size(self, width: u32, height: u32) -> usize {
        self.row_size(width) * height.div_ceil(self.block_extent().1) as usize
    }

    pub fn block_size(self) -> usize {
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
            Format::Bc1RgbaUnorm
            | Format::Bc1RgbaUnormSrgb
            | Format::Bc4RUnorm
            | Format::Bc4RSnorm => 8,
            Format::Bc2RgbaUnorm
            | Format::Bc2RgbaUnormSrgb
            | Format::Bc3RgbaUnorm
            | Format::Bc3RgbaUnormSrgb
            | Format::Bc5RgUnorm
            | Format::Bc5RgSnorm
            | Format::Bc6hRgbFloat
            | Format::Bc6hRgbUfloat
            | Format::Bc7RgbaUnorm
            | Format::Bc7RgbaUnormSrgb
            | Format::Astc4x4Unorm
            | Format::Astc4x4UnormSrgb => 16,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {

    MeshShaders,

    TileShaders,

    IndirectCommands,

    Timestamps,

    Occlusion,

    MultiDraw,
    TextureCompressionBc,
    TextureCompressionAstc,
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

#[derive(Clone, Copy, Debug)]
pub struct SamplerDesc {
    pub min: Filter,
    pub mag: Filter,
    pub mip: Filter,
    pub address: AddressMode,
    pub anisotropy: u32,

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

pub trait IndexFormat: Copy {
    const WIDE: bool;
}

impl IndexFormat for u16 {
    const WIDE: bool = false;
}

impl IndexFormat for u32 {
    const WIDE: bool = true;
}

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

#[derive(Clone, Copy)]
pub struct RasterDesc<'a> {
    pub vertex: &'a str,
    pub fragment: &'a str,
    pub colors: &'a [ColorTarget],
    pub depth: Option<Format>,
    pub topology: Topology,
    pub samples: u32,
    pub constants: Option<&'a dyn Constants>,

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveMode {
    SampleZero,
    Min,
    Max,
}

#[derive(Clone, Copy)]
pub struct MeshDesc<'a> {

    pub object: &'a str,
    pub mesh: &'a str,
    pub fragment: &'a str,
    pub colors: &'a [ColorTarget],

    pub payload: usize,
    pub samples: u32,
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
            constants: None,
            indirect: false,
        }
    }

    pub fn indirect(mut self) -> Self {
        self.indirect = true;
        self
    }
}

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

    pub fn test(&self) -> bool {
        self.compare != CompareOp::Always || self.write
    }
}

pub const FRAMES_IN_FLIGHT: u64 = 2;
