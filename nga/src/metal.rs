//! The Metal 4 backend.
//!
//! ## Backend contract
//!
//! `lib.rs` ends with `pub use backend::*`, so this file and `vulkan.rs` must define the same
//! names with the same signatures. Neither is compiled on the other's platform, so the
//! compiler cannot check that — this list is the checklist, and it is kept in the same order
//! in both files:
//!
//! `Device`, `Arena`, `GpuHeap`, `HeapBuffer`, `HeapTexture`, `TextureDesc`, `ResourceHeap`,
//! `HeapRange`, `Shaders`, `Pipeline`, `Surface`, `Drawable`, `Pass`, `Raster`, `Compute`,
//! `Context`, `Frame`, `Recorder`, `Fence`.
//!
//! Nothing from `minmetal` appears in a public signature. Every handle that crosses the
//! boundary is a newtype declared here with private fields, so `pub use backend::*` cannot
//! leak one.
//!
//! ## What Metal does not need
//!
//! Three things `vulkan.rs` spends hundreds of lines on are absent here, and their methods are
//! kept as no-ops rather than deleted so the two surfaces still match:
//!
//! - **Staging.** Every heap is `StorageMode::Shared` under unified memory, so `GpuHeap::write`
//!   is a memcpy and `flush` has nothing to submit.
//! - **Image layouts.** There is no layout to transition into, so `transition` does nothing.
//! - **Descriptor tables.** A handle *is* an `MTLResourceID`, stored inline in the root struct
//!   the shader dereferences, so `ResourceHeap` allocates no descriptor memory at all.

use crate::{
    AddressMode, Alloc, BlendFactor, BlendMode, CompareOp, Constant, Constants, Cull,
    DepthState, Error, FRAMES_IN_FLIGHT, Feature, Filter, Format, Gpu, GpuBox, GpuMut, Handle,
    Hazard, IndexFormat, Memory, RasterDesc, Sampler, SamplerDesc, Scalar, Scissor, Stage,
    Texture2d, Texture2dRW, Tlsf, Topology, Usage, Viewport, Winding,
};
use minmetal::{
    Allocation, Buffer, ColorWriteMask, CompareFunction, CpuCacheMode, CullMode,
    DepthStencilDescriptor, DepthStencilState, FunctionConstantValues, HazardTrackingMode, Heap,
    HeapDescriptor, HeapType, IndexType, M4AlphaToCoverageState, M4ArgumentTable,
    M4ArgumentTableDescriptor, M4BlendState, M4CommandAllocator, M4CommandBuffer, M4CommandQueue,
    M4Compiler, M4CompilerDescriptor, M4ComputeCommandEncoder, M4ComputePipelineDescriptor,
    M4FunctionDescriptor, M4IndirectCommandBufferSupportState, M4LibraryDescriptor,
    M4LibraryFunctionDescriptor, M4RenderCommandEncoder, M4RenderPassDescriptor,
    M4RenderPipelineDescriptor, M4SpecializedFunctionDescriptor, M4VisibilityOptions, MetalError,
    MetalLayer, Origin, PixelFormat, PrimitiveTopologyClass, PrimitiveType, Region, RenderStages,
    ResidencySet, ResidencySetDescriptor, ResourceOptions, SamplerAddressMode, SamplerDescriptor,
    SamplerMinMagFilter, SamplerMipFilter, SharedEvent, Size, Stages, StorageMode, Texture,
    TextureDescriptor, TextureUsage, id, retain,
};
use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// A compiled pipeline. The object itself is owned by the [`Shaders`] that built it; this is a
/// borrowed pointer, which is what keeps it `Copy` the way `vk::Pipeline` is. `id` is `Send`
/// and `Sync`, so a pipeline still crosses threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pipeline {
    raw: id,
    /// Metal picks the primitive at draw time, Vulkan at pipeline creation. Carrying it here is
    /// what lets `RasterDesc::topology` mean the same thing on both.
    primitive: PrimitiveType,
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

/// A sampler *object*. `Sampler` in `lib.rs` is the bindless handle that points at one of
/// these, which is why this needs a different name.
pub struct SamplerState(minmetal::SamplerState);

impl From<MetalError> for Error {
    fn from(error: MetalError) -> Self {
        Error::new(error.to_string())
    }
}

// -- enum translation --------------------------------------------------------------------
//
// One `fn` per public enum. Adding a variant in `lib.rs` breaks these, which is the point:
// the compiler points at every backend that has not caught up.

pub(crate) fn format(format: Format) -> PixelFormat {
    match format {
        Format::Rgba8Unorm => PixelFormat::Rgba8Unorm,
        Format::Rgba8UnormSrgb => PixelFormat::Rgba8UnormSrgb,
        Format::Bgra8Unorm => PixelFormat::Bgra8Unorm,
        Format::Bgra8UnormSrgb => PixelFormat::Bgra8UnormSrgb,
        Format::Rgba16Float => PixelFormat::Rgba16Float,
        Format::Rgba32Float => PixelFormat::Rgba32Float,
        Format::Rg11B10Float => PixelFormat::Rg11B10Float,
        Format::Rgb10A2Unorm => PixelFormat::Rgb10A2Unorm,
        Format::Rg8Unorm => PixelFormat::Rg8Unorm,
        Format::Rg16Float => PixelFormat::Rg16Float,
        Format::R8Unorm => PixelFormat::R8Unorm,
        Format::R16Float => PixelFormat::R16Float,
        Format::R32Float => PixelFormat::R32Float,
        Format::R32Uint => PixelFormat::R32Uint,
        Format::Rg32Uint => PixelFormat::Rg32Uint,
        Format::Rgba32Uint => PixelFormat::Rgba32Uint,
        Format::D32Float => PixelFormat::Depth32Float,
        Format::D32FloatS8 => PixelFormat::Depth32FloatStencil8,
        Format::D16Unorm => PixelFormat::Depth16Unorm,
    }
}

pub(crate) fn from_metal_format(raw: PixelFormat) -> Option<Format> {
    Some(match raw {
        PixelFormat::Rgba8Unorm => Format::Rgba8Unorm,
        PixelFormat::Rgba8UnormSrgb => Format::Rgba8UnormSrgb,
        PixelFormat::Bgra8Unorm => Format::Bgra8Unorm,
        PixelFormat::Bgra8UnormSrgb => Format::Bgra8UnormSrgb,
        PixelFormat::Rgba16Float => Format::Rgba16Float,
        PixelFormat::Rgba32Float => Format::Rgba32Float,
        PixelFormat::Rg11B10Float => Format::Rg11B10Float,
        PixelFormat::Rgb10A2Unorm => Format::Rgb10A2Unorm,
        PixelFormat::Rg8Unorm => Format::Rg8Unorm,
        PixelFormat::Rg16Float => Format::Rg16Float,
        PixelFormat::R8Unorm => Format::R8Unorm,
        PixelFormat::R16Float => Format::R16Float,
        PixelFormat::R32Float => Format::R32Float,
        PixelFormat::R32Uint => Format::R32Uint,
        PixelFormat::Rg32Uint => Format::Rg32Uint,
        PixelFormat::Rgba32Uint => Format::Rgba32Uint,
        PixelFormat::Depth32Float => Format::D32Float,
        PixelFormat::Depth32FloatStencil8 => Format::D32FloatS8,
        PixelFormat::Depth16Unorm => Format::D16Unorm,
        _ => return None,
    })
}

/// Unified memory means there is one useful storage mode. `Memory` still exists in the shared
/// surface because a discrete Vulkan device genuinely has three; here it collapses.
pub(crate) fn storage(memory: Memory) -> StorageMode {
    match memory {
        Memory::Device | Memory::Upload | Memory::Readback => StorageMode::Shared,
    }
}

pub(crate) fn usage(usage: Usage) -> TextureUsage {
    let mut out = TextureUsage::UNKNOWN;
    if usage.has(Usage::SAMPLED) {
        out = out | TextureUsage::SHADER_READ;
    }
    if usage.has(Usage::STORAGE) {
        out = out | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    }
    if usage.has(Usage::COLOR) || usage.has(Usage::DEPTH_STENCIL) {
        out = out | TextureUsage::RENDER_TARGET;
    }
    out
}

pub(crate) fn topology(topology: Topology) -> PrimitiveType {
    match topology {
        Topology::TriangleList => PrimitiveType::Triangle,
        Topology::TriangleStrip => PrimitiveType::TriangleStrip,
        Topology::LineList => PrimitiveType::Line,
        Topology::LineStrip => PrimitiveType::LineStrip,
        Topology::PointList => PrimitiveType::Point,
    }
}

/// The pipeline only needs the *class*, and only to let the driver reject a mismatched draw.
pub(crate) fn topology_class(topology: Topology) -> PrimitiveTopologyClass {
    match topology {
        Topology::TriangleList | Topology::TriangleStrip => PrimitiveTopologyClass::Triangle,
        Topology::LineList | Topology::LineStrip => PrimitiveTopologyClass::Line,
        Topology::PointList => PrimitiveTopologyClass::Point,
    }
}

pub(crate) fn cull(mode: Cull) -> CullMode {
    match mode {
        Cull::None => CullMode::None,
        Cull::Front => CullMode::Front,
        Cull::Back => CullMode::Back,
    }
}

pub(crate) fn winding(winding: Winding) -> minmetal::Winding {
    match winding {
        Winding::Clockwise => minmetal::Winding::Clockwise,
        Winding::CounterClockwise => minmetal::Winding::CounterClockwise,
    }
}

pub(crate) fn compare_op(op: CompareOp) -> CompareFunction {
    match op {
        CompareOp::Never => CompareFunction::Never,
        CompareOp::Less => CompareFunction::Less,
        CompareOp::Equal => CompareFunction::Equal,
        CompareOp::LessEqual => CompareFunction::LessEqual,
        CompareOp::Greater => CompareFunction::Greater,
        CompareOp::NotEqual => CompareFunction::NotEqual,
        CompareOp::GreaterEqual => CompareFunction::GreaterEqual,
        CompareOp::Always => CompareFunction::Always,
    }
}

/// The index into `Context::depth_states`. Sixteen depth-stencil objects are built once and
/// looked up by value, because Metal has no dynamic depth state and building one per call
/// would allocate inside a render pass.
pub(crate) fn depth_index(state: &DepthState) -> usize {
    let compare = match state.compare {
        CompareOp::Never => 0,
        CompareOp::Less => 1,
        CompareOp::Equal => 2,
        CompareOp::LessEqual => 3,
        CompareOp::Greater => 4,
        CompareOp::NotEqual => 5,
        CompareOp::GreaterEqual => 6,
        CompareOp::Always => 7,
    };
    compare * 2 + state.write as usize
}

pub(crate) fn compare_at(index: usize) -> CompareFunction {
    match index {
        0 => CompareFunction::Never,
        1 => CompareFunction::Less,
        2 => CompareFunction::Equal,
        3 => CompareFunction::LessEqual,
        4 => CompareFunction::Greater,
        5 => CompareFunction::NotEqual,
        6 => CompareFunction::GreaterEqual,
        _ => CompareFunction::Always,
    }
}

pub(crate) fn filter(filter: Filter) -> SamplerMinMagFilter {
    match filter {
        Filter::Nearest => SamplerMinMagFilter::Nearest,
        Filter::Linear => SamplerMinMagFilter::Linear,
    }
}

pub(crate) fn mip_mode(filter: Filter) -> SamplerMipFilter {
    match filter {
        Filter::Nearest => SamplerMipFilter::Nearest,
        Filter::Linear => SamplerMipFilter::Linear,
    }
}

pub(crate) fn address_mode(mode: AddressMode) -> SamplerAddressMode {
    match mode {
        AddressMode::ClampToEdge => SamplerAddressMode::ClampToEdge,
        AddressMode::Repeat => SamplerAddressMode::Repeat,
        AddressMode::MirrorRepeat => SamplerAddressMode::MirrorRepeat,
    }
}

pub(crate) fn blend_factor(factor: BlendFactor) -> minmetal::BlendFactor {
    match factor {
        BlendFactor::Zero => minmetal::BlendFactor::Zero,
        BlendFactor::One => minmetal::BlendFactor::One,
        BlendFactor::SrcColor => minmetal::BlendFactor::SourceColor,
        BlendFactor::OneMinusSrcColor => minmetal::BlendFactor::OneMinusSourceColor,
        BlendFactor::SrcAlpha => minmetal::BlendFactor::SourceAlpha,
        BlendFactor::OneMinusSrcAlpha => minmetal::BlendFactor::OneMinusSourceAlpha,
        BlendFactor::DstColor => minmetal::BlendFactor::DestinationColor,
        BlendFactor::OneMinusDstColor => minmetal::BlendFactor::OneMinusDestinationColor,
        BlendFactor::DstAlpha => minmetal::BlendFactor::DestinationAlpha,
        BlendFactor::OneMinusDstAlpha => minmetal::BlendFactor::OneMinusDestinationAlpha,
    }
}

/// `(src_rgb, dst_rgb, src_alpha, dst_alpha)`, or `None` for no blending at all. Both backends
/// use `BlendOperation::Add` throughout, so the op is not part of [`BlendMode`].
pub(crate) fn blend(
    mode: BlendMode,
) -> Option<(BlendFactor, BlendFactor, BlendFactor, BlendFactor)> {
    match mode {
        BlendMode::Opaque => None,
        BlendMode::Alpha => Some((
            BlendFactor::SrcAlpha,
            BlendFactor::OneMinusSrcAlpha,
            BlendFactor::One,
            BlendFactor::OneMinusSrcAlpha,
        )),
        BlendMode::Premultiplied => Some((
            BlendFactor::One,
            BlendFactor::OneMinusSrcAlpha,
            BlendFactor::One,
            BlendFactor::OneMinusSrcAlpha,
        )),
        BlendMode::Additive => Some((
            BlendFactor::One,
            BlendFactor::One,
            BlendFactor::One,
            BlendFactor::One,
        )),
        BlendMode::Screen => Some((
            BlendFactor::One,
            BlendFactor::OneMinusSrcColor,
            BlendFactor::One,
            BlendFactor::OneMinusSrcAlpha,
        )),
    }
}

/// Vulkan's mask is `R=1, G=2, B=4, A=8`; Metal's is the same four bits in the opposite order.
/// `ColorTarget::write_mask` carries the Vulkan spelling, so it is reversed here.
pub(crate) fn write_mask(mask: u8) -> ColorWriteMask {
    let mut out = ColorWriteMask::NONE;
    if mask & 0x1 != 0 {
        out = out | ColorWriteMask::RED;
    }
    if mask & 0x2 != 0 {
        out = out | ColorWriteMask::GREEN;
    }
    if mask & 0x4 != 0 {
        out = out | ColorWriteMask::BLUE;
    }
    if mask & 0x8 != 0 {
        out = out | ColorWriteMask::ALPHA;
    }
    out
}

/// `MTL4RenderPipelineDescriptor.depthAttachmentPixelFormat` has no minmetal setter, and a
/// pipeline that renders into a depth attachment has to declare the format or Metal rejects
/// the pass. One message send beats forking the dependency.
fn set_depth_format(pipeline: &M4RenderPipelineDescriptor, format: PixelFormat) {
    unsafe {
        let send: unsafe extern "C" fn(id, minmetal::SEL, usize) =
            std::mem::transmute(minmetal::objc_msgSend as *const std::ffi::c_void);
        send(
            pipeline.raw,
            minmetal::sel(b"setDepthAttachmentPixelFormat:\0"),
            format.as_raw(),
        );
    }
}

/// `Stage` is the blog post's vocabulary; Metal's encoder stages are a shorter list, so several
/// nga stages land on the same one. That is safe — a barrier that names more stages than it
/// needs is conservative, never wrong.
pub(crate) fn stage(stage: Stage) -> Stages {
    let mut out = Stages::NONE;
    if stage.has(Stage::TRANSFER) {
        out = out | Stages::BLIT;
    }
    if stage.has(Stage::COMPUTE) {
        out = out | Stages::DISPATCH;
    }
    // Metal has no indirect stage: indirect arguments are read by whichever stage consumes
    // them, which is the vertex stage for a draw and the dispatch stage for a dispatch. Naming
    // only `DISPATCH` here left the render encoder's argument fetch unordered against the
    // compute pass that wrote it — the missing bands in `gpu_driven`.
    if stage.has(Stage::INDIRECT) {
        out = out | Stages::DISPATCH | Stages::VERTEX | Stages::OBJECT | Stages::MESH;
    }
    if stage.has(Stage::VERTEX) {
        out = out | Stages::VERTEX | Stages::OBJECT | Stages::MESH;
    }
    if stage.has(Stage::PIXEL) || stage.has(Stage::COLOR_OUT) || stage.has(Stage::DEPTH_STENCIL) {
        out = out | Stages::FRAGMENT;
    }
    match out {
        Stages::NONE => Stages::ALL,
        _ => out,
    }
}

/// Metal has one visibility bit where Vulkan has an access mask per resource kind. Anything
/// that is not "execution ordering only" becomes a device-scope flush.
pub(crate) fn hazard(hazard: Hazard) -> M4VisibilityOptions {
    match hazard {
        Hazard::NONE => M4VisibilityOptions::NONE,
        _ => M4VisibilityOptions::DEVICE,
    }
}

/// The specialization payload. Metal takes the same bytes Vulkan does, keyed by the same index,
/// so `Constant` crosses unchanged — only the type tag has to be translated.
pub(crate) fn constants(values: &[Constant]) -> FunctionConstantValues {
    let out = FunctionConstantValues::new();
    for value in values {
        out.set_bytes(
            value.index as usize,
            data_type(value.scalar),
            &value.bytes[..value.len as usize],
        );
    }
    out
}

pub(crate) fn data_type(scalar: Scalar) -> minmetal::DataType {
    use minmetal::DataType;
    match scalar {
        Scalar::Bool => DataType::Bool,
        Scalar::F16 => DataType::Half,
        Scalar::F32 => DataType::Float,
        Scalar::U32 => DataType::UInt,
        Scalar::I32 => DataType::Int,
        Scalar::U64 => DataType::ULong,
        Scalar::F16x2 => DataType::Half2,
        Scalar::F16x3 => DataType::Half3,
        Scalar::F16x4 => DataType::Half4,
        Scalar::F32x2 => DataType::Float2,
        Scalar::F32x3 => DataType::Float3,
        Scalar::F32x4 => DataType::Float4,
        Scalar::U32x2 => DataType::UInt2,
        Scalar::U32x3 => DataType::UInt3,
        Scalar::U32x4 => DataType::UInt4,
        Scalar::I32x2 => DataType::Int2,
        Scalar::I32x3 => DataType::Int3,
        Scalar::I32x4 => DataType::Int4,
    }
}

/// What the shader finds at argument-table index 0. The Metal target lowers
/// `[[vk::push_constant]] ConstantBuffer<nga_push>` to `constant nga_push* [[buffer(0)]]`, so
/// unlike Vulkan's push constants this is a *pointer* — the struct itself is allocated from the
/// frame arena at bind time.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Push {
    root: u64,
    resources: u32,
    samplers: u32,
}

// -- device ------------------------------------------------------------------------------

/// The device, the one queue, and the residency set every allocation registers itself in.
///
/// Metal 4 requires resources to be resident before a command buffer touches them. Rather than
/// make that the caller's problem, `Arena`, `GpuHeap` and `Surface` add themselves here on
/// creation; the queue holds the set, so later additions are picked up without rebinding.
pub struct Device {
    pub(crate) device: minmetal::Device,
    pub(crate) queue: M4CommandQueue,
    pub(crate) residency: ResidencySet,
    /// Signalled by `wait_idle` only. Frames use the context's own event, so the two counters
    /// never collide.
    pub(crate) idle_event: SharedEvent,
    pub(crate) idle: AtomicU64,
    pub(crate) name: String,
}

impl Device {
    /// `validation` has no effect: Metal's validation layer is selected by the
    /// `METAL_DEVICE_WRAPPER_TYPE` environment variable before the process starts, not by the
    /// API. The argument stays so the two backends open a device the same way.
    pub fn new(_validation: bool) -> Result<Self, Error> {
        let device = minmetal::Device::system_default()
            .ok_or(Error::new("no Metal device"))?;
        let queue = device.new_m4_command_queue().ok_or(Error::new(
            "MTL4CommandQueue is unavailable — Metal 4 requires macOS 26",
        ))?;
        let residency = device.new_residency_set(&ResidencySetDescriptor::new()?)?;
        residency.commit()?;
        queue.add_residency_set(&residency);
        Ok(Self {
            name: device.name().to_string(),
            idle_event: device.new_shared_event()?,
            idle: AtomicU64::new(0),
            device,
            queue,
            residency,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Metal has no `vkDeviceWaitIdle`. The equivalent is to signal an event *behind* every
    /// command buffer already committed and wait for it: the queue is in-order, so the signal
    /// cannot land before the work in front of it has.
    pub fn wait_idle(&self) -> Result<(), Error> {
        let value = self.idle.fetch_add(1, Ordering::Relaxed) + 1;
        self.queue.signal_event(&self.idle_event.as_event(), value);
        match self.idle_event.wait_until_signaled_value(value, 10_000) {
            true => Ok(()),
            false => Err(Error::new("timed out waiting for the device to go idle")),
        }
    }

    /// Whether this adapter can do something outside the shared surface.
    ///
    /// Deliberately conservative in the same way `vulkan.rs` is: `true` only where the backend
    /// has a path today. Mesh, tile and indirect-command support is real Metal functionality
    /// that has not been ported back yet, so it answers `false` until it has.
    pub fn supports(&self, feature: Feature) -> bool {
        match feature {
            Feature::TileShaders => false,
            Feature::MeshShaders => false,
            Feature::IndirectCommands => false,
            // `VK_EXT_multi_draw` has no counterpart; a run of draws is a run of calls here.
            Feature::MultiDraw => false,
            Feature::Timestamps => true,
            Feature::Occlusion => true,
        }
    }

    /// Metal's argument buffers have no descriptor ceiling — the handle is the resource ID, not
    /// an index into a fixed table — so this is the heap's own capacity, whatever it was asked
    /// for. Vulkan's answer is a driver limit.
    pub fn max_resources(&self) -> u32 {
        u32::MAX
    }

    /// There is no multi-draw call, so the longest accepted run is one.
    pub fn max_multi_draw(&self) -> u32 {
        1
    }

    /// A sampler object, to be written into a heap slot. `support_argument_buffers` is what
    /// gives it a `MTLResourceID`; without it the handle would be zero.
    pub fn sampler(&self, desc: &SamplerDesc) -> Result<SamplerState, Error> {
        let descriptor = SamplerDescriptor::new();
        descriptor.set_min_filter(filter(desc.min));
        descriptor.set_mag_filter(filter(desc.mag));
        descriptor.set_mip_filter(mip_mode(desc.mip));
        descriptor.set_s_address_mode(address_mode(desc.address));
        descriptor.set_t_address_mode(address_mode(desc.address));
        descriptor.set_r_address_mode(address_mode(desc.address));
        descriptor.set_max_anisotropy(desc.anisotropy.max(1) as usize);
        if let Some(op) = desc.compare {
            descriptor.set_compare_function(compare_op(op));
        }
        descriptor.set_support_argument_buffers(true);
        Ok(SamplerState(self.device.new_sampler_state(&descriptor)?))
    }

    /// Registers an allocation with the queue's residency set. Every type in this file that owns
    /// device memory calls it once, at creation.
    pub(crate) fn make_resident(&self, allocation: &Allocation) -> Result<(), Error> {
        self.residency.add_allocation(allocation)?;
        self.residency.commit()?;
        Ok(())
    }
}

// -- arena -------------------------------------------------------------------------------

/// A bump allocator over one shared buffer. Identical in shape to the Vulkan one; the
/// difference is that there is no BAR window to run out of, so `Memory` picks nothing.
pub struct Arena {
    pub(crate) buffer: Buffer,
    pub(crate) base: u64,
    pub(crate) cpu: *mut u8,
    pub(crate) size: usize,
    pub(crate) head: AtomicUsize,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new(device: &Device, size: usize, memory: Memory) -> Result<Self, Error> {
        let options = ResourceOptions::new(
            CpuCacheMode::DefaultCache,
            storage(memory),
            HazardTrackingMode::Untracked,
        );
        let buffer = device.device.new_buffer(size, options)?;
        device.make_resident(&Allocation::from_buffer(&buffer))?;
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

// -- heaps -------------------------------------------------------------------------------

pub struct HeapBuffer {
    pub(crate) buffer: Buffer,
    pub(crate) gpu: u64,
    pub(crate) cpu: *mut u8,
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
    pub(crate) texture: Texture,
    pub(crate) format: Format,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) alloc: Alloc,
}

impl HeapTexture {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> Format {
        self.format
    }
}

#[derive(Clone, Copy)]
pub struct TextureDesc {
    pub(crate) format: Format,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) usage: Usage,
}

impl TextureDesc {
    pub fn texture_2d(fmt: Format, width: u32, height: u32) -> Self {
        Self {
            format: fmt,
            width,
            height,
            usage: Usage::SAMPLED | Usage::COPY_DST,
        }
    }

    pub fn depth(fmt: Format, width: u32, height: u32) -> Self {
        Self {
            format: fmt,
            width,
            height,
            usage: Usage::DEPTH_STENCIL,
        }
    }

    /// Extra usage beyond what the constructor implies — storage writes, colour attachment.
    pub fn usage(mut self, extra: Usage) -> Self {
        self.usage = self.usage | extra;
        self
    }

    fn descriptor(&self) -> TextureDescriptor {
        let desc = TextureDescriptor::new();
        desc.set_pixel_format(format(self.format));
        desc.set_width(self.width as usize);
        desc.set_height(self.height as usize);
        desc.set_usage(usage(self.usage));
        desc
    }
}

/// One placement heap with a TLSF allocator over it. Unlike the Vulkan heap this is host
/// visible, so there is no upload ring and no staging — `write` is a memcpy and `flush` is a
/// no-op kept only so the two surfaces match.
pub struct GpuHeap {
    pub(crate) device: minmetal::Device,
    pub(crate) heap: Heap,
    pub(crate) tlsf: Tlsf,
    pub(crate) options: ResourceOptions,
}

impl GpuHeap {
    pub fn new(device: &Device, size: u64) -> Result<Self, Error> {
        Self::with_ring(device, size, 0)
    }

    /// `ring` is the Vulkan upload ring, which does not exist here. Accepted and ignored.
    pub fn with_ring(device: &Device, size: u64, _ring: usize) -> Result<Self, Error> {
        let options = ResourceOptions::new(
            CpuCacheMode::DefaultCache,
            StorageMode::Shared,
            HazardTrackingMode::Untracked,
        );
        let desc = HeapDescriptor::new();
        desc.set_heap_type(HeapType::Placement);
        desc.set_storage_mode(StorageMode::Shared);
        desc.set_hazard_tracking_mode(HazardTrackingMode::Untracked);
        desc.set_size(size as usize);
        let heap = device.device.new_heap(&desc)?;
        device.make_resident(&Allocation::from_heap(&heap))?;
        Ok(Self {
            device: device.device.clone(),
            heap,
            tlsf: Tlsf::new(size, 256),
            options,
        })
    }

    pub fn alloc_buffer(&mut self, len: usize) -> Result<HeapBuffer, Error> {
        let need = self.device.heap_buffer_size_and_align(len, self.options);
        let alloc = self
            .tlsf
            .alloc(need.size as u64, need.align as u64)
            .ok_or(Error::new("gpu heap is full"))?;
        let buffer = self
            .heap
            .new_buffer_at_offset(len, self.options, alloc.offset as usize)?;
        Ok(HeapBuffer {
            gpu: buffer.gpu_address()?,
            cpu: buffer.contents() as *mut u8,
            len,
            alloc,
            buffer,
        })
    }

    pub fn alloc_texture(&mut self, desc: &TextureDesc) -> Result<HeapTexture, Error> {
        let descriptor = desc.descriptor();
        descriptor.set_storage_mode(StorageMode::Shared);
        let need = self.device.heap_texture_size_and_align(&descriptor);
        let alloc = self
            .tlsf
            .alloc(need.size as u64, need.align as u64)
            .ok_or(Error::new("gpu heap is full"))?;
        Ok(HeapTexture {
            texture: self
                .heap
                .new_texture_at_offset(&descriptor, alloc.offset as usize)?,
            format: desc.format,
            width: desc.width,
            height: desc.height,
            alloc,
        })
    }

    /// A memcpy. This is the half of unified memory Vulkan cannot give back: there, the same
    /// call stages through a ring and records a copy.
    pub fn write<T: Copy>(&mut self, buffer: &HeapBuffer, offset: usize, values: &[T]) {
        self.try_write(buffer, offset, values)
            .expect("buffer write failed")
    }

    pub fn try_write<T: Copy>(
        &mut self,
        buffer: &HeapBuffer,
        offset: usize,
        values: &[T],
    ) -> Result<(), Error> {
        let bytes = size_of_val(values);
        debug_assert!(offset + bytes <= buffer.len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr() as *const u8,
                buffer.cpu.add(offset),
                bytes,
            )
        };
        Ok(())
    }

    /// `replaceRegion` writes straight into the texture — no staging buffer, no command buffer.
    /// This is the call `VK_EXT_host_image_copy` was meant to be.
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
        let row = bytes.len() / height as usize;
        texture.texture.replace_region(
            Region::new_2d(0, 0, width as usize, height as usize),
            0,
            bytes,
            row,
        );
        Ok(())
    }

    /// Nothing is staged, so there is nothing to submit.
    pub fn flush(&mut self) -> Result<(), Error> {
        Ok(())
    }

    /// The timeline value a frame would wait on. Zero: an upload here is already visible.
    pub fn flush_async(&mut self) -> Result<u64, Error> {
        Ok(0)
    }

    pub fn free(&mut self, alloc: Alloc) {
        self.tlsf.free(alloc.block);
    }

    /// Metal has no image layouts, so there is nothing to transition into. The method exists
    /// because Vulkan cannot do without it.
    pub fn transition(&mut self, _texture: &HeapTexture, _usage: Usage) {}

    pub fn free_buffer(&mut self, buffer: HeapBuffer) {
        self.tlsf.free(buffer.alloc.block);
    }

    pub fn free_texture(&mut self, texture: HeapTexture) {
        self.tlsf.free(texture.alloc.block);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HeapRange {
    pub(crate) base: u32,
    pub(crate) len: u32,
}

/// On Metal a bindless handle *is* the `MTLResourceID`, stored inline in the root struct the
/// shader dereferences — see the argument-buffer members slangc emits for `DescriptorHandle<T>`.
/// So there is no descriptor memory to manage and this type is a slot bookkeeper: it exists so
/// that `alloc_range` / `write_texture` / `write_sampler` mean the same thing on both backends,
/// where Vulkan really does have three descriptor arrays behind them.
pub struct ResourceHeap {
    pub(crate) cap: u32,
    pub(crate) head: u32,
}

impl ResourceHeap {
    pub fn new(_device: &Device, cap: u32) -> Result<Self, Error> {
        Ok(Self { cap, head: 0 })
    }

    pub fn alloc_range(&mut self, len: u32) -> Result<HeapRange, Error> {
        if self.head + len > self.cap {
            return Err(Error::new("resource heap is full"));
        }
        let base = self.head;
        self.head += len;
        Ok(HeapRange { base, len })
    }

    pub fn write_texture(&self, range: HeapRange, slot: u32, texture: &HeapTexture) -> Texture2d {
        debug_assert!(slot < range.len && range.base + slot < self.cap);
        Texture2d(resource_id(&texture.texture))
    }

    /// The Vulkan version needs the usage to pick an image layout for the descriptor. There is
    /// no layout here, so the argument is ignored.
    pub fn write_texture_in(
        &self,
        range: HeapRange,
        slot: u32,
        texture: &HeapTexture,
        _usage: Usage,
    ) -> Texture2d {
        self.write_texture(range, slot, texture)
    }

    pub fn write_storage(&self, range: HeapRange, slot: u32, texture: &HeapTexture) -> Texture2dRW {
        debug_assert!(slot < range.len && range.base + slot < self.cap);
        Texture2dRW(resource_id(&texture.texture))
    }

    pub fn write_sampler(&self, range: HeapRange, slot: u32, sampler: &SamplerState) -> Sampler {
        debug_assert!(slot < range.len && range.base + slot < self.cap);
        Sampler(Handle(
            sampler
                .0
                .gpu_resource_id()
                .expect("sampler has no resource id")
                .impl_,
        ))
    }
}

fn resource_id(texture: &Texture) -> Handle {
    Handle(
        texture
            .gpu_resource_id()
            .expect("texture has no resource id")
            .impl_,
    )
}

// -- surface -----------------------------------------------------------------------------

/// A `CAMetalLayer` attached to an `NSView`. `handle` is the view pointer, which is what the
/// Vulkan side spells `hwnd`; keeping one name for it is the only way `Surface::new` can have
/// one signature.
pub struct Surface {
    pub(crate) layer: MetalLayer,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Surface {
    pub fn new(device: &Device, view: isize, width: u32, height: u32) -> Result<Self, Error> {
        let layer = unsafe {
            MetalLayer::attach_to_view(
                id(view as *mut std::ffi::c_void),
                &device.device,
                PixelFormat::Bgra8Unorm,
                width as usize,
                height as usize,
                1.0,
            )?
        };
        Ok(Self {
            layer,
            width,
            height,
        })
    }

    /// Resizing a `CAMetalLayer` is a property write, where Vulkan rebuilds the whole swapchain.
    pub fn rebuild(&mut self, _device: &Device, width: u32, height: u32) -> Result<(), Error> {
        self.layer.set_drawable_size(width as usize, height as usize);
        self.width = width;
        self.height = height;
        Ok(())
    }

    pub fn acquire(&mut self, _device: &Device) -> Option<Drawable> {
        let drawable = self.layer.next_drawable()?;
        let texture = drawable.texture().ok()?;
        Some(Drawable { drawable, texture })
    }

    pub fn format(&self) -> Format {
        from_metal_format(self.layer.pixel_format()).expect("layer format is outside nga::Format")
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Presenting happens inside `Frame::submit`, because the queue has to signal the drawable
    /// before it is presented. This is the Vulkan-shaped hook, and it never asks for a rebuild:
    /// a layer has no `ERROR_OUT_OF_DATE_KHR` equivalent.
    pub fn present(&mut self, _device: &Device, _drawable: &Drawable) -> bool {
        false
    }

    pub fn destroy(&mut self, _device: &Device) {}
}

pub struct Drawable {
    pub(crate) drawable: minmetal::Drawable,
    pub(crate) texture: Texture,
}

// -- shaders -----------------------------------------------------------------------------

/// One `MTLLibrary` holds every function, the way one SPIR-V module holds every entry point,
/// so pipelines name their stages by string exactly as they do under Vulkan.
pub struct Shaders {
    pub(crate) compiler: M4Compiler,
    pub(crate) library: minmetal::Library,
    /// The pipeline objects a [`Pipeline`] points into. Dropping `Shaders` invalidates every
    /// handle it handed out, which is exactly what the Vulkan side does.
    pub(crate) render: Vec<minmetal::RenderPipelineState>,
    pub(crate) compute: Vec<minmetal::ComputePipelineState>,
}

impl Shaders {
    /// `source` is the MSL `build.rs` produced from the Slang, as bytes — the same shape the
    /// Vulkan side takes SPIR-V in.
    pub fn new(device: &Device, source: &[u8]) -> Result<Self, Error> {
        let source = std::str::from_utf8(source)
            .map_err(|_| Error::new("shader source is not valid UTF-8"))?;
        let compiler = device
            .device
            .new_m4_compiler_with_descriptor(&M4CompilerDescriptor::new())?;
        let desc = M4LibraryDescriptor::new();
        desc.set_source(source);
        Ok(Self {
            library: compiler.new_library_with_descriptor(&desc)?,
            compiler,
            render: Vec::new(),
            compute: Vec::new(),
        })
    }

    fn function(&self, name: &str, values: Option<&dyn Constants>) -> M4FunctionDescriptor {
        let function = M4LibraryFunctionDescriptor::new();
        function.set_name(name);
        function.set_library(Some(&self.library));
        let Some(values) = values else {
            return function.as_function_descriptor();
        };
        let specialized = M4SpecializedFunctionDescriptor::new();
        specialized.set_function_descriptor(Some(&function.as_function_descriptor()));
        specialized.set_constant_values(Some(&constants(&values.values())));
        specialized.set_specialized_name(name);
        M4FunctionDescriptor {
            raw: retain(specialized.raw),
        }
    }

    pub fn raster(&mut self, desc: &RasterDesc) -> Result<Pipeline, Error> {
        let pipeline = M4RenderPipelineDescriptor::new();
        pipeline.set_label(desc.vertex);
        pipeline.set_vertex_function_descriptor(Some(&self.function(desc.vertex, desc.constants)));
        if !desc.fragment.is_empty() {
            let fragment = self.function(desc.fragment, desc.constants);
            pipeline.set_fragment_function_descriptor(Some(&fragment));
        }
        pipeline.set_support_indirect_command_buffers(M4IndirectCommandBufferSupportState::Disabled);
        pipeline.set_rasterization_enabled(true);
        pipeline.set_input_primitive_topology(topology_class(desc.topology));
        pipeline.set_raster_sample_count(desc.samples as usize);
        pipeline.set_alpha_to_coverage_state(M4AlphaToCoverageState::Disabled);
        if let Some(depth) = desc.depth {
            set_depth_format(&pipeline, format(depth));
        }

        let attachments = pipeline.color_attachments();
        for (index, target) in desc.colors.iter().enumerate() {
            let attachment = attachments.object_at(index);
            attachment.set_pixel_format(format(target.format));
            attachment.set_write_mask(write_mask(target.write_mask));
            match blend(target.blend) {
                Some((src_rgb, dst_rgb, src_alpha, dst_alpha)) => {
                    attachment.set_blending_state(M4BlendState::Enabled);
                    attachment.set_blend_factors(
                        blend_factor(src_rgb),
                        blend_factor(dst_rgb),
                        minmetal::BlendOperation::Add,
                        blend_factor(src_alpha),
                        blend_factor(dst_alpha),
                        minmetal::BlendOperation::Add,
                    );
                }
                None => attachment.set_blending_state(M4BlendState::Disabled),
            }
        }

        let state = self
            .compiler
            .new_render_pipeline_state(&pipeline.as_pipeline_descriptor(), None)?;
        let raw = state.raw;
        self.render.push(state);
        Ok(Pipeline {
            raw,
            primitive: topology(desc.topology),
        })
    }

    pub fn compute(
        &mut self,
        entry: &str,
        constants: Option<&dyn Constants>,
    ) -> Result<Pipeline, Error> {
        let pipeline = M4ComputePipelineDescriptor::new();
        pipeline.set_label(entry);
        pipeline.set_compute_function_descriptor(Some(&self.function(entry, constants)));
        let state = self.compiler.new_compute_pipeline_state(&pipeline, None)?;
        let raw = state.raw;
        self.compute.push(state);
        Ok(Pipeline {
            raw,
            primitive: PrimitiveType::Triangle,
        })
    }
}

// -- passes ------------------------------------------------------------------------------

pub struct Pass {
    pub(crate) desc: M4RenderPassDescriptor,
}

impl Default for Pass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass {
    pub fn new() -> Self {
        Self {
            desc: M4RenderPassDescriptor::new(),
        }
    }

    pub fn color(self, index: usize, drawable: &Drawable, clear: Option<[f32; 4]>) -> Self {
        self.attach(index, &drawable.texture, clear)
    }

    /// Renders into a texture rather than the surface. The two differ only in where the texture
    /// came from, so offscreen rendering needs no separate pass type — and unlike Vulkan there
    /// is no tracked layout to update, because there is no layout.
    pub fn color_texture(
        self,
        index: usize,
        texture: &HeapTexture,
        clear: Option<[f32; 4]>,
    ) -> Self {
        self.attach(index, &texture.texture, clear)
    }

    fn attach(self, index: usize, texture: &Texture, clear: Option<[f32; 4]>) -> Self {
        let attachment = self.desc.color_attachments().object_at_indexed_subscript(index);
        attachment.base().set_texture(Some(texture));
        attachment.base().set_store_action(minmetal::StoreAction::Store);
        match clear {
            Some([r, g, b, a]) => {
                attachment.base().set_load_action(minmetal::LoadAction::Clear);
                attachment.set_clear_color(minmetal::ClearColor::new(
                    r as f64, g as f64, b as f64, a as f64,
                ));
            }
            None => attachment.base().set_load_action(minmetal::LoadAction::Load),
        }
        self
    }

    pub fn depth(self, texture: &HeapTexture, clear: Option<f32>, store: bool) -> Self {
        let attachment = self.desc.depth_attachment();
        attachment.base().set_texture(Some(&texture.texture));
        attachment.base().set_store_action(match store {
            true => minmetal::StoreAction::Store,
            false => minmetal::StoreAction::DontCare,
        });
        match clear {
            Some(depth) => {
                attachment.base().set_load_action(minmetal::LoadAction::Clear);
                attachment.set_clear_depth(depth as f64);
            }
            None => attachment.base().set_load_action(minmetal::LoadAction::Load),
        }
        self
    }
}

// -- recording ---------------------------------------------------------------------------

pub struct Raster<'a> {
    pub(crate) encoder: M4RenderCommandEncoder,
    pub(crate) table: &'a M4ArgumentTable,
    pub(crate) arena: &'a Arena,
    pub(crate) depth_states: &'a [DepthStencilState],
    pub(crate) primitive: Cell<PrimitiveType>,
}

impl<'a> Raster<'a> {
    pub(crate) fn new(
        command_buffer: &'a M4CommandBuffer,
        table: &'a M4ArgumentTable,
        arena: &'a Arena,
        depth_states: &'a [DepthStencilState],
        pass: &Pass,
        width: u32,
        height: u32,
    ) -> Self {
        let encoder = command_buffer
            .render_command_encoder_with_descriptor(&pass.desc)
            .expect("render command encoder");
        encoder.set_argument_table(
            table,
            RenderStages::VERTEX
                | RenderStages::FRAGMENT
                | RenderStages::TILE
                | RenderStages::OBJECT
                | RenderStages::MESH,
        );
        let raster = Self {
            encoder,
            table,
            arena,
            depth_states,
            primitive: Cell::new(PrimitiveType::Triangle),
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
        raster
    }

    /// Writes the root pointer where the shader expects it. slangc lowers the shared
    /// `nga_push` block to `constant nga_push* [[buffer(0)]]`, so what the argument table holds
    /// is the address of a struct, not the root itself — one 16-byte arena allocation per bind.
    fn bind<T>(&self, root: Gpu<'_, T>) {
        bind_push(self.table, self.arena, root.0.get());
    }

    pub fn pipeline(&self, pipeline: Pipeline) {
        self.primitive.set(pipeline.primitive);
        let state = ManuallyDrop::new(minmetal::RenderPipelineState { raw: pipeline.raw });
        self.encoder.set_render_pipeline_state(&state);
    }

    /// Nothing to bind: a handle carries the resource itself. Vulkan binds two descriptor
    /// buffers here.
    pub fn heap(&self, _heap: &ResourceHeap) {}

    pub fn depth(&self, state: &DepthState) {
        self.encoder
            .set_depth_stencil_state(&self.depth_states[depth_index(state)]);
    }

    pub fn cull(&self, mode: Cull, front: Winding) {
        self.encoder.set_cull_mode(cull(mode));
        self.encoder.set_front_facing_winding(winding(front));
    }

    pub fn viewport(&self, view: Viewport, clip: Scissor) {
        self.encoder.set_viewport(minmetal::Viewport {
            origin_x: view.x as f64,
            origin_y: view.y as f64,
            width: view.width as f64,
            height: view.height as f64,
            znear: view.min_depth as f64,
            zfar: view.max_depth as f64,
        });
        self.encoder.set_scissor_rect(minmetal::ScissorRect {
            x: clip.x as usize,
            y: clip.y as usize,
            width: clip.width as usize,
            height: clip.height as usize,
        });
    }

    pub fn draw_indexed<T, I: IndexFormat>(
        &self,
        root: Gpu<'_, T>,
        indices: &HeapBuffer,
        count: u32,
    ) {
        self.bind(root);
        self.encoder.draw_indexed_primitives(
            self.primitive.get(),
            count as usize,
            match I::WIDE {
                true => IndexType::UInt32,
                false => IndexType::UInt16,
            },
            indices.gpu,
            indices.len,
        );
    }

    /// Metal has no `vkCmdDrawIndirectCount`: the count lives on the GPU and only an indirect
    /// command buffer can consume it there. Until `Feature::IndirectCommands` is implemented
    /// the portable spelling is `max` indirect draws — an entry the encode pass never wrote
    /// has `instance_count == 0` and costs nothing beyond the call, which is why the caller
    /// must clear `draws` each frame.
    pub fn draw_indirect_count<T>(
        &self,
        root: Gpu<'_, T>,
        draws: &HeapBuffer,
        _count: &HeapBuffer,
        max: u32,
    ) {
        self.bind(root);
        for index in 0..max as u64 {
            self.encoder.draw_primitives_with_indirect_buffer(
                self.primitive.get(),
                draws.gpu + index * size_of::<crate::Draw>() as u64,
            );
        }
    }

    pub fn draw<T>(&self, root: Gpu<'_, T>, start: u32, count: u32) {
        self.bind(root);
        self.encoder
            .draw_primitives(self.primitive.get(), start as usize, count as usize);
    }
}

impl Drop for Raster<'_> {
    fn drop(&mut self) {
        self.encoder.end_encoding();
    }
}

pub struct Compute<'a> {
    pub(crate) encoder: M4ComputeCommandEncoder,
    pub(crate) table: &'a M4ArgumentTable,
    pub(crate) arena: &'a Arena,
}

impl<'a> Compute<'a> {
    pub(crate) fn new(
        command_buffer: &'a M4CommandBuffer,
        table: &'a M4ArgumentTable,
        arena: &'a Arena,
    ) -> Self {
        let encoder = command_buffer
            .compute_command_encoder()
            .expect("compute command encoder");
        encoder.set_argument_table(Some(table));
        Self {
            encoder,
            table,
            arena,
        }
    }

    fn bind<T>(&self, root: Gpu<'_, T>) {
        bind_push(self.table, self.arena, root.0.get());
    }

    pub fn pipeline(&self, pipeline: Pipeline) {
        let state = ManuallyDrop::new(minmetal::ComputePipelineState { raw: pipeline.raw });
        self.encoder.set_compute_pipeline_state(&state);
    }

    pub fn heap(&self, _heap: &ResourceHeap) {}

    pub fn dispatch<T>(&self, root: Gpu<'_, T>, threads: usize, group: usize) {
        self.bind(root);
        self.encoder
            .dispatch_threads(Size::new(threads, 1, 1), Size::new(group, 1, 1));
    }

    pub fn dispatch_2d<T>(&self, root: Gpu<'_, T>, width: u32, height: u32, group: u32) {
        self.bind(root);
        self.encoder.dispatch_threads(
            Size::new(width as usize, height as usize, 1),
            Size::new(group as usize, group as usize, 1),
        );
    }

    /// The whole barrier, with no resource list — exactly the design the post argues for. Metal
    /// spells it as stages plus one visibility bit, which is strictly less than
    /// `VkMemoryBarrier2` carries and lands in the same place.
    ///
    /// It takes *three* calls, because Metal 4 splits what one `vkCmdPipelineBarrier2` does
    /// into one in-encoder barrier and two queue-scoped halves, and a barrier that reaches only
    /// some of them is a race:
    ///
    /// - `barrier` orders against earlier commands in *this* encoder.
    /// - `barrierAfterQueueStages` orders this encoder after everything already committed —
    ///   the only thing that sees a *preceding* render pass.
    /// - `barrierBeforeQueueStages` publishes this encoder's writes to everything committed
    ///   after it — the only thing a *following* render pass sees.
    ///
    /// Both queue halves were found the same way, by a race the shared surface cannot express
    /// any other way: without the first, the offscreen raster test failed six runs in twenty;
    /// without the second, `gpu_driven` drew bands of empty draw slots because the render
    /// encoder fetched indirect arguments the encode pass had not published yet.
    pub fn barrier(&self, before: Stage, after: Stage, hazards: Hazard) {
        let (before, after, hazard) = (stage(before), stage(after), hazard(hazards));
        self.encoder.barrier(before, after, hazard);
        self.encoder.barrier_after_queue_stages(before, after, hazard);
        self.encoder
            .barrier_before_queue_stages(before, after, hazard);
    }

    pub fn copy_buffer(
        &self,
        source: &HeapBuffer,
        source_offset: usize,
        destination: &HeapBuffer,
        destination_offset: usize,
        size: usize,
    ) {
        self.encoder.copy_buffer_to_buffer(
            &source.buffer,
            source_offset,
            &destination.buffer,
            destination_offset,
            size,
        );
    }

    /// No layouts on Metal, so nothing to do. Vulkan cannot do without it.
    pub fn transition(&self, _texture: &HeapTexture, _usage: Usage) {}

    /// Reads a whole texture back into host-visible memory, tightly packed. The counterpart of
    /// `GpuHeap::upload`, and what makes a headless render checkable without a screenshot.
    pub fn read_texture(&self, texture: &HeapTexture, arena: &Arena, dest: GpuMut<'_, [u8]>) {
        let offset = dest
            .0
            .get()
            .checked_sub(arena.base)
            .expect("the readback destination is not in this arena") as usize;
        let row = texture.width as usize * texture.format.size();
        self.encoder.copy_texture_to_buffer(
            &texture.texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(texture.width as usize, texture.height as usize, 1),
            &arena.buffer,
            offset,
            row,
            row * texture.height as usize,
            minmetal::BlitOption::NONE,
        );
    }

    /// Metal's fill takes a byte, Vulkan's a word. The shared surface takes the word and this
    /// splits it, which is exact for the only case either backend is used for — clearing.
    pub fn fill_buffer(&self, buffer: &HeapBuffer, offset: usize, size: usize, value: u32) {
        let bytes = value.to_ne_bytes();
        debug_assert!(
            bytes.iter().all(|&byte| byte == bytes[0]),
            "Metal fills a buffer with one repeated byte, so {value:#x} cannot be expressed"
        );
        self.encoder
            .fill_buffer(&buffer.buffer, minmetal::Range::new(offset, size), bytes[0]);
    }
}

impl Drop for Compute<'_> {
    fn drop(&mut self) {
        self.encoder.end_encoding();
    }
}

fn bind_push(table: &M4ArgumentTable, arena: &Arena, root: u64) {
    let (cpu, gpu) = arena
        .alloc_bytes(size_of::<Push>(), 16)
        .expect("frame arena is full: no room for a root binding");
    unsafe {
        (cpu as *mut Push).write(Push {
            root,
            resources: 0,
            samplers: 0,
        })
    };
    table.set_address(gpu, 0);
}

// -- frames ------------------------------------------------------------------------------

pub struct Context {
    pub(crate) queue: M4CommandQueue,
    pub(crate) event: SharedEvent,
    pub(crate) command_buffers: Vec<M4CommandBuffer>,
    pub(crate) allocators: Vec<M4CommandAllocator>,
    pub(crate) tables: Vec<M4ArgumentTable>,
    pub(crate) arenas: Vec<Arena>,
    /// Every (compare, write) pair, built once. Metal has no dynamic depth state, and a render
    /// pass is the wrong place to be creating objects.
    pub(crate) depth_states: Vec<DepthStencilState>,
    pub(crate) threads: usize,
    pub(crate) frame: u64,
}

impl Context {
    /// `shaders` is what the Vulkan side takes its pipeline layout from. Metal binds through the
    /// argument table instead, so it is unused here — the argument stays for one signature.
    pub fn new(
        device: &Device,
        _shaders: &Shaders,
        arena_size: usize,
        threads: usize,
    ) -> Result<Self, Error> {
        debug_assert!(threads > 0);
        let slots = FRAMES_IN_FLIGHT as usize * threads;

        let mut arenas = Vec::new();
        let mut allocators = Vec::new();
        let mut command_buffers = Vec::new();
        for _ in 0..slots {
            arenas.push(Arena::new(device, arena_size, Memory::Upload)?);
            allocators.push(
                device
                    .device
                    .new_m4_command_allocator()
                    .ok_or(Error::new("newCommandAllocator returned nil"))?,
            );
            command_buffers.push(
                device
                    .device
                    .new_m4_command_buffer()
                    .ok_or(Error::new("newCommandBuffer returned nil"))?,
            );
        }

        // One buffer binding: index 0 is the root block every shader dereferences.
        let table_desc = M4ArgumentTableDescriptor::new();
        table_desc.set_max_buffer_bind_count(1);
        let mut tables = Vec::new();
        for _ in 0..slots {
            tables.push(
                device
                    .device
                    .new_m4_argument_table_with_descriptor(&table_desc)?,
            );
        }

        let mut depth_states = Vec::with_capacity(16);
        for index in 0..16 {
            let desc = DepthStencilDescriptor::new();
            desc.set_depth_compare_function(compare_at(index / 2));
            desc.set_depth_write_enabled(index % 2 == 1);
            depth_states.push(device.device.new_depth_stencil_state(&desc)?);
        }

        Ok(Self {
            queue: M4CommandQueue::from_raw(device.queue.raw),
            event: device.device.new_shared_event()?,
            command_buffers,
            allocators,
            tables,
            arenas,
            depth_states,
            threads,
            frame: 0,
        })
    }

    /// How many frames have been submitted. Callers use it to pick a per-frame slot; the
    /// context uses it to decide which arena and command allocator are free.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn begin_frame(&mut self) -> Result<Frame<'_>, Error> {
        let index = self.frame;
        self.frame += 1;
        if index >= FRAMES_IN_FLIGHT
            && !self
                .event
                .wait_until_signaled_value(index + 1 - FRAMES_IN_FLIGHT, 10_000)
        {
            return Err(Error::new("timed out waiting for a frame in flight"));
        }

        let base = (index % FRAMES_IN_FLIGHT) as usize * self.threads;
        let slots = base..base + self.threads;
        for slot in slots.clone() {
            self.allocators[slot].reset();
            self.arenas[slot].reset();
        }

        Ok(Frame {
            queue: &self.queue,
            event: &self.event,
            arenas: &self.arenas[slots.clone()],
            allocators: &self.allocators[slots.clone()],
            command_buffers: &self.command_buffers[slots.clone()],
            tables: &self.tables[slots],
            depth_states: &self.depth_states,
            recording: 0,
            value: index + 1,
        })
    }
}

pub struct Frame<'a> {
    pub(crate) queue: &'a M4CommandQueue,
    pub(crate) event: &'a SharedEvent,
    pub(crate) arenas: &'a [Arena],
    pub(crate) allocators: &'a [M4CommandAllocator],
    pub(crate) command_buffers: &'a [M4CommandBuffer],
    pub(crate) tables: &'a [M4ArgumentTable],
    pub(crate) depth_states: &'a [DepthStencilState],
    pub(crate) recording: usize,
    pub(crate) value: u64,
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
    /// One recorder per thread, each with its own command buffer, argument table and arena
    /// slot. `nga` takes a count per queue; with one queue this is `&[n]`.
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
        (0..count).map(|index| self.record(index)).collect()
    }

    pub fn recorder(&mut self) -> Recorder<'_> {
        let index = self.recording;
        self.recording += 1;
        debug_assert!(index < self.command_buffers.len());
        self.record(index)
    }

    fn record(&self, index: usize) -> Recorder<'_> {
        self.command_buffers[index].begin_command_buffer_with_allocator(&self.allocators[index]);
        Recorder {
            arena: &self.arenas[index],
            command_buffer: &self.command_buffers[index],
            table: &self.tables[index],
            depth_states: self.depth_states,
            not_sync: PhantomData,
        }
    }

    /// Uploads are memcpys here, so there is nothing for the GPU to wait on.
    pub fn after_uploads(&mut self, _value: u64, _heap: &GpuHeap) {}

    /// Unlike Vulkan, presenting is part of submission: the queue has to signal the drawable
    /// after the work that draws into it, and only then can it be presented.
    pub fn submit(self, drawable: Option<&Drawable>) -> Result<(), Error> {
        if let Some(drawable) = drawable {
            self.queue.wait_for_drawable(&drawable.drawable);
        }
        self.queue
            .commit(&self.command_buffers[..self.recording]);
        self.queue.signal_event(&self.event.as_event(), self.value);
        if let Some(drawable) = drawable {
            self.queue.signal_drawable(&drawable.drawable);
            drawable.drawable.present();
        }
        Ok(())
    }
}

pub struct Recorder<'a> {
    pub(crate) arena: &'a Arena,
    pub(crate) command_buffer: &'a M4CommandBuffer,
    pub(crate) table: &'a M4ArgumentTable,
    pub(crate) depth_states: &'a [DepthStencilState],
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
        Compute::new(self.command_buffer, self.table, self.arena)
    }

    pub fn raster(&mut self, pass: &Pass, width: u32, height: u32) -> Raster<'_> {
        Raster::new(
            self.command_buffer,
            self.table,
            self.arena,
            self.depth_states,
            pass,
            width,
            height,
        )
    }
}

impl Drop for Recorder<'_> {
    fn drop(&mut self) {
        self.command_buffer.end_command_buffer();
    }
}
