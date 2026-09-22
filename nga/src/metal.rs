//! The Metal 4 backend.
//!
//! ## Backend contract
//!
//! `lib.rs` ends with `pub use backend::*`, so this file and `vulkan.rs` must define the same
//! names with the same signatures. Neither is compiled on the other's platform, so the
//! compiler cannot check that, so this list is the checklist, and it is kept in the same order
//! in both files:
//!
//! `Device`, `Arena`, `GpuHeap`, `Retired`, `HeapBuffer`, `HeapTexture`, `TextureDesc`,
//! `ResourceHeap`, `HeapRange`, `Shaders`, `Pipeline`, `Surface`, `Drawable`, `Pass`, `Raster`,
//! `MultiDraw`, `MultiDrawIndexed`, `Compute`, `Context`, `Frame`, `Recorder`, `Fence`,
//! `Timers`, `Occlusion`.
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
    AddressMode, Alloc, BlendFactor, BlendMode, ColorTarget, Commands, CompareOp, Constant,
    Constants, Cull, DepthState, DrawIndexed, Error, FRAMES_IN_FLIGHT, Feature, Filter, Format,
    Gpu, GpuBox, GpuRw, Handle, Hazard, IndexFormat, Memory, MeshDesc, PipelineHandle, RasterDesc,
    ResolveMode, Sampler, SamplerDesc, Scalar, Scissor, Stage, StencilFace, StencilOp, Texture2d,
    Texture2dRW, TileDesc, Tlsf, Topology, Usage, Viewport, Winding,
};
use minmetal::{
    Allocation, Buffer, ColorWriteMask, CompareFunction, CpuCacheMode, CullMode,
    DepthStencilDescriptor, DepthStencilState as MetalDepthStencilState, FunctionConstantValues,
    HazardTrackingMode, Heap, HeapDescriptor, HeapType, IndexType, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, IndirectCommandBufferOptions, IndirectCommandType,
    M4AlphaToCoverageState, M4ArgumentTable, M4ArgumentTableDescriptor, M4BlendState,
    M4CommandAllocator, M4CommandBuffer, M4CommandQueue, M4Compiler, M4CompilerDescriptor,
    M4ComputeCommandEncoder, M4ComputePipelineDescriptor, M4CounterHeap, M4CounterHeapDescriptor,
    M4CounterHeapType, M4FunctionDescriptor, M4IndirectCommandBufferSupportState,
    M4LibraryDescriptor, M4LibraryFunctionDescriptor, M4MeshRenderPipelineDescriptor,
    M4RenderCommandEncoder, M4RenderPassDescriptor, M4RenderPipelineColorAttachmentDescriptorArray,
    M4RenderPipelineDescriptor, M4SpecializedFunctionDescriptor, M4TileRenderPipelineDescriptor,
    M4TimestampGranularity, M4VisibilityOptions, MetalError, MetalLayer,
    MultisampleDepthResolveFilter, Origin, PixelFormat, PrimitiveTopologyClass, PrimitiveType,
    Region, RenderStages, ResidencySet, ResidencySetDescriptor, ResourceOptions,
    ResourceViewPoolDescriptor, SamplerAddressMode, SamplerDescriptor, SamplerMinMagFilter,
    SamplerMipFilter, SharedEvent, Size, Stages, StencilDescriptor, StencilOperation,
    StorageMode, Texture, TextureDescriptor, TextureType, TextureUsage, TextureViewPool,
    VisibilityResultMode, VisibilityResultType, id, retain,
};
use std::cell::Cell;
use std::collections::VecDeque;
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
    /// The `MTLResourceID` a shader names when encoding this pipeline into an
    /// [`IndirectCommands`] buffer. Zero unless the descriptor asked for it.
    handle: PipelineHandle,
}

impl Pipeline {
    /// The handle a shader writes into a command it encodes. Only meaningful when the pipeline
    /// was built with `indirect`; zero otherwise, and zero always on Vulkan.
    pub fn handle(&self) -> PipelineHandle {
        self.handle
    }
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

/// A sampler *object*. `Sampler` in `lib.rs` is the bindless handle that points at one of
/// these, which is why this needs a different name.
pub struct SamplerState(minmetal::SamplerState);

/// A depth-stencil object created before command recording.
#[doc(hidden)]
pub struct DepthStencilState(MetalDepthStencilState);

/// One run of a multi-draw. Layout matches `VkMultiDrawInfoEXT`, so on Vulkan a slice passes
/// straight into `vkCmdDrawMultiEXT`; here it is walked one draw at a time.
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

pub(crate) fn stencil_op(op: StencilOp) -> StencilOperation {
    match op {
        StencilOp::Keep => StencilOperation::Keep,
        StencilOp::Zero => StencilOperation::Zero,
        StencilOp::Replace => StencilOperation::Replace,
        StencilOp::IncrementClamp => StencilOperation::IncrementClamp,
        StencilOp::DecrementClamp => StencilOperation::DecrementClamp,
        StencilOp::Invert => StencilOperation::Invert,
        StencilOp::IncrementWrap => StencilOperation::IncrementWrap,
        StencilOp::DecrementWrap => StencilOperation::DecrementWrap,
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

fn stencil_descriptor(face: StencilFace) -> StencilDescriptor {
    let desc = StencilDescriptor::new();
    desc.set_stencil_compare_function(compare_op(face.compare));
    desc.set_stencil_failure_operation(stencil_op(face.fail));
    desc.set_depth_failure_operation(stencil_op(face.depth_fail));
    desc.set_depth_stencil_pass_operation(stencil_op(face.pass));
    desc.set_read_mask(face.read_mask);
    desc.set_write_mask(face.write_mask);
    desc
}

fn depth_stencil_descriptor(state: &DepthState) -> DepthStencilDescriptor {
    let desc = DepthStencilDescriptor::new();
    desc.set_depth_compare_function(compare_op(state.compare));
    desc.set_depth_write_enabled(state.write);
    if let Some(stencil) = state.stencil {
        desc.set_front_face_stencil(&stencil_descriptor(stencil.front));
        desc.set_back_face_stencil(&stencil_descriptor(stencil.back));
    }
    desc
}

fn metal_depth_state(
    device: &minmetal::Device,
    compare: CompareOp,
    write: bool,
) -> Result<MetalDepthStencilState, Error> {
    let desc = depth_stencil_descriptor(&DepthState::new(compare, write));
    Ok(device.new_depth_stencil_state(&desc)?)
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
/// `Stage` is the blog post's vocabulary; Metal's encoder stages are a shorter list, so several
/// nga stages land on the same one. That is safe: a barrier that names more stages than it
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
    // compute pass that wrote it, which is where the missing bands in `gpu_driven` came from.
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
/// so `Constant` crosses unchanged and only the type tag has to be translated.
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
/// unlike Vulkan's push constants this is a pointer. The struct itself is allocated from the
/// frame arena at bind time.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Push {
    root: u64,
    resources: u64,
    resources_stride: u32,
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
        let device = minmetal::Device::system_default().ok_or(Error::new("no Metal device"))?;
        let queue = device.new_m4_command_queue().ok_or(Error::new(
            "MTL4CommandQueue is unavailable, Metal 4 requires macOS 26",
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

    /// Whether this adapter can do something outside the shared API.
    ///
    /// `true` only where the backend has a path today, not merely where the hardware could.
    /// Every method behind a feature exists on both backends; on the one without the path it
    /// fails or does nothing, which is why asking first is not optional.
    pub fn supports(&self, feature: Feature) -> bool {
        match feature {
            Feature::TileShaders => true,
            Feature::MeshShaders => true,
            Feature::IndirectCommands => true,
            // `VK_EXT_multi_draw` has no counterpart; a run of draws is a run of calls here.
            Feature::MultiDraw => false,
            Feature::Timestamps => true,
            Feature::Occlusion => true,
        }
    }

    /// Metal's argument buffers have no descriptor ceiling, because the handle is the resource
    /// ID rather than an index into a fixed table. So this is the heap's own capacity, whatever
    /// it was asked for. Vulkan's answer is a driver limit.
    /// Metal's API validation aborts rather than reporting, so there is nothing to count.
    /// The name exists because the Vulkan side has it and a test may check either.
    pub fn validation_errors(&self) -> u64 {
        0
    }

    pub fn max_resources(&self) -> u32 {
        u32::MAX
    }

    /// There is no multi-draw call, so the longest accepted run is one.
    pub fn max_multi_draw(&self) -> u32 {
        1
    }

    /// Build a combined depth-stencil object before command recording.
    #[doc(hidden)]
    pub fn depth_stencil_state(&self, state: &DepthState) -> Result<DepthStencilState, Error> {
        Ok(DepthStencilState(self.device.new_depth_stencil_state(
            &depth_stencil_descriptor(state),
        )?))
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
            gpu: crate::device_ptr(gpu),
            _mapped: PhantomData,
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
            gpu: crate::device_ptr(gpu),
            _mapped: PhantomData,
        })
    }

    /// Reserves mapped memory for the GPU to fill.
    ///
    /// # Safety
    ///
    /// The caller must not read any element through the returned [`GpuBox`] until GPU work has
    /// initialised every element and the host has waited for that work to complete. Metal clears
    /// the bytes for deterministic captures, but an all-zero byte pattern is not a valid value
    /// for every `T`.
    pub unsafe fn alloc_scratch<T>(&self, len: usize) -> Option<GpuBox<'_, [T]>> {
        let bytes = size_of::<T>() * len;
        let (cpu, gpu) = self.alloc_bytes(bytes, align_of::<T>().max(16))?;
        unsafe { std::ptr::write_bytes(cpu, 0, bytes) };
        Some(GpuBox {
            cpu: NonNull::slice_from_raw_parts(
                unsafe { NonNull::new_unchecked(cpu as *mut T) },
                len,
            ),
            gpu: crate::device_ptr(gpu),
            _mapped: PhantomData,
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
    /// The device address, for a field of a struct a shader dereferences.
    ///
    /// Copying it does not keep this buffer alive. [`GpuHeap::free_later`] is what waits
    /// until the frame that still reads the address has retired.
    pub fn gpu<T: ?Sized>(&self) -> Gpu<T> {
        crate::device_ptr(self.gpu)
    }

    /// The raw device address.
    pub fn address(&self) -> u64 {
        self.gpu
    }

    pub fn gpu_rw<T: ?Sized>(&self) -> GpuRw<T> {
        crate::device_ptr_rw(self.gpu)
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
    pub(crate) mips: u32,
    pub(crate) layers: u32,
    pub(crate) alloc: Alloc,
}

impl HeapTexture {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn mips(&self) -> u32 {
        self.mips
    }

    pub fn layers(&self) -> u32 {
        self.layers
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
    pub(crate) samples: u32,
    pub(crate) mips: u32,
    pub(crate) layers: u32,
}

impl TextureDesc {
    pub fn texture_2d(fmt: Format, width: u32, height: u32) -> Self {
        Self {
            format: fmt,
            width,
            height,
            usage: Usage::SAMPLED | Usage::COPY_DST,
            samples: 1,
            mips: 1,
            layers: 1,
        }
    }

    pub fn depth(fmt: Format, width: u32, height: u32) -> Self {
        Self {
            format: fmt,
            width,
            height,
            usage: Usage::DEPTH_STENCIL,
            samples: 1,
            mips: 1,
            layers: 1,
        }
    }

    /// Build the texture with `mips` levels, each half the extent of the one above and rounded
    /// down to a minimum of one. Filling them is the caller's job: [`GpuHeap::upload`] takes the
    /// level it writes. Metal does have `generateMipmaps` on a blit encoder, but nga does not
    /// expose it, because Vulkan has no counterpart and levels computed offline filter better.
    pub fn mips(mut self, mips: u32) -> Self {
        let full = 32 - self.width.max(self.height).leading_zeros();
        assert!(
            (1..=full).contains(&mips),
            "{mips} is outside the 1..={full} mip levels a {}x{} image has",
            self.width,
            self.height
        );
        self.mips = mips;
        self
    }

    /// Build a 2D array texture. A layered pass writes `Pass::layers` consecutive slices,
    /// starting at the slice selected by its attachment.
    pub fn layers(mut self, layers: u32) -> Self {
        assert!(layers > 0, "a texture must have at least one array layer");
        self.layers = layers;
        self
    }

    /// Extra usage beyond what the constructor implies: storage writes, colour attachment.
    pub fn usage(mut self, extra: Usage) -> Self {
        self.usage = self.usage | extra;
        self
    }

    /// Build the texture with `samples` samples per texel, for use as a multisample attachment
    /// under a pipeline built with the matching [`RasterDesc::samples`]. A multisample texture
    /// is only ever an attachment and a resolve source, so this also drops the usage the
    /// constructor assumed, the same narrowing Vulkan needs for the same reason.
    pub fn samples(mut self, samples: u32) -> Self {
        assert!(
            matches!(samples, 1 | 2 | 4 | 8),
            "{samples} is not a Metal sample count"
        );
        self.samples = samples;
        if samples > 1 {
            // A multisample texture is one level by definition, and it is never sampled, so the
            // count the caller may have asked for is dropped along with the usage.
            self.mips = 1;
            self.usage = match self.usage.has(Usage::DEPTH_STENCIL) {
                true => Usage::DEPTH_STENCIL,
                false => Usage::COLOR,
            };
        }
        self
    }

    fn descriptor(&self) -> TextureDescriptor {
        let desc = TextureDescriptor::new();
        desc.set_pixel_format(format(self.format));
        desc.set_width(self.width as usize);
        desc.set_height(self.height as usize);
        let mut texture_usage = usage(self.usage);
        if self.layers > 1 {
            texture_usage = texture_usage | TextureUsage::PIXEL_FORMAT_VIEW;
            desc.set_array_length(self.layers as usize);
        }
        desc.set_usage(texture_usage);
        if self.mips > 1 {
            desc.set_mipmap_level_count(self.mips as usize);
        }
        if self.samples > 1 && self.layers > 1 {
            desc.set_texture_type(TextureType::D2MultisampleArray);
            desc.set_sample_count(self.samples as usize);
        } else if self.samples > 1 {
            desc.set_texture_type(TextureType::D2Multisample);
            desc.set_sample_count(self.samples as usize);
        } else if self.layers > 1 {
            desc.set_texture_type(TextureType::D2Array);
        }
        desc
    }
}

/// One placement heap with a TLSF allocator over it. Unlike the Vulkan heap this is host
/// visible, so there is no upload ring and no staging. `write` is a memcpy and `flush` is a
/// no-op kept only so the two backends match.
pub struct GpuHeap {
    pub(crate) device: minmetal::Device,
    pub(crate) heap: Heap,
    pub(crate) tlsf: Tlsf,
    pub(crate) options: ResourceOptions,
    pub(crate) retired: VecDeque<(u64, Retired)>,
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
            retired: VecDeque::new(),
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
            mips: desc.mips,
            layers: desc.layers,
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

    /// `replaceRegion` writes straight into the texture. No staging buffer, no command buffer.
    /// This is the call `VK_EXT_host_image_copy` was meant to be.
    pub fn upload(&mut self, texture: &HeapTexture, level: u32, bytes: &[u8]) -> Result<(), Error> {
        self.upload_layer(texture, level, 0, bytes)
    }

    /// Upload one complete mip level of one array layer.
    pub fn upload_layer(
        &mut self,
        texture: &HeapTexture,
        level: u32,
        layer: u32,
        bytes: &[u8],
    ) -> Result<(), Error> {
        if level >= texture.mips {
            return Err(Error::new(format!(
                "mip level {level} is outside the {} this texture has",
                texture.mips
            )));
        }
        if layer >= texture.layers {
            return Err(Error::new(format!(
                "array layer {layer} is outside the {} this texture has",
                texture.layers
            )));
        }
        let width = (texture.width >> level).max(1);
        let height = (texture.height >> level).max(1);
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
        let view;
        let target;
        let target_level;
        if texture.layers > 1 {
            view = texture.texture.new_texture_view_with_type_and_ranges(
                format(texture.format),
                TextureType::D2,
                minmetal::Range::new(level as usize, 1),
                minmetal::Range::new(layer as usize, 1),
            )?;
            target = &view;
            target_level = 0;
        } else {
            target = &texture.texture;
            target_level = level as usize;
        }
        target.replace_region(
            Region::new_2d(0, 0, width as usize, height as usize),
            target_level,
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

    /// Hand a resource over to be released once the frame that last used it has finished.
    ///
    /// `frame` is [`Context::frame`] at the point of the hand-over. A [`Gpu`] copied out of the
    /// resource does not delay this. Pass the frame that is still reading it.
    pub fn free_later(&mut self, frame: u64, retired: Retired) {
        self.retired.push_back((frame, retired));
    }

    /// Release everything handed over during a frame that has since completed. `completed` is
    /// [`Context::completed`]; call this once a frame, before allocating.
    pub fn retire(&mut self, completed: u64) {
        while self
            .retired
            .front()
            .is_some_and(|&(frame, _)| frame < completed)
        {
            match self.retired.pop_front().unwrap().1 {
                Retired::Buffer(buffer) => self.free_buffer(buffer),
                Retired::Texture(texture) => self.free_texture(texture),
            }
        }
    }
}

/// A resource waiting for the GPU to be done with it. See [`GpuHeap::free_later`].
pub enum Retired {
    Buffer(HeapBuffer),
    Texture(HeapTexture),
}

#[derive(Clone, Copy, Debug)]
pub struct HeapRange {
    pub(crate) base: u32,
    pub(crate) len: u32,
}

/// Backed by an `MTLTextureViewPool`. A bindless handle is a 32-bit slot index, matching Vulkan.
/// The shader synthesizes the required `MTLResourceID` on the fly from the pool's base resource ID
/// and descriptor stride.
pub struct ResourceHeap {
    pub(crate) cap: u32,
    pub(crate) head: u32,
    pub(crate) pool: TextureViewPool,
    pub(crate) base_id: u64,
    pub(crate) stride: u32,
}

impl ResourceHeap {
    pub fn new(device: &Device, cap: u32) -> Result<Self, Error> {
        let desc = ResourceViewPoolDescriptor::new()
            .map_err(|e| Error::new(format!("failed to create pool descriptor: {e:?}")))?;
        desc.set_resource_view_count(cap.max(2) as usize);
        let pool = device
            .device
            .new_texture_view_pool_with_descriptor(&desc)
            .map_err(|e| Error::new(format!("failed to create texture view pool: {e:?}")))?;
        let base_id = pool.base_resource_id().impl_;

        let tex_desc = TextureDescriptor::texture_2d(PixelFormat::Rgba8Unorm, 1, 1, false);
        let dummy = device
            .device
            .new_texture(&tex_desc)
            .map_err(|e| Error::new(format!("failed to create dummy texture: {e:?}")))?;
        let id0 = pool
            .set_texture_view(&dummy, 0)
            .map_err(|e| Error::new(format!("failed to set dummy view 0: {e:?}")))?
            .impl_;
        let id1 = pool
            .set_texture_view(&dummy, 1)
            .map_err(|e| Error::new(format!("failed to set dummy view 1: {e:?}")))?
            .impl_;
        let stride = (id1 - id0) as u32;

        device.make_resident(&Allocation { raw: pool.raw })?;

        Ok(Self {
            cap,
            head: 0,
            pool,
            base_id,
            stride,
        })
    }

    pub fn alloc_range(&mut self, len: u32) -> Result<HeapRange, Error> {
        if self.head + len > self.cap {
            return Err(Error::new(format!(
                "a range of {len} slots does not fit: {} of {} are taken",
                self.head, self.cap
            )));
        }
        let base = self.head;
        self.head += len;
        Ok(HeapRange { base, len })
    }

    pub fn write_texture(&self, range: HeapRange, slot: u32, texture: &HeapTexture) -> Texture2d {
        debug_assert!(slot < range.len && range.base + slot < self.cap);
        let index = (range.base + slot) as usize;
        self.pool
            .set_texture_view(&texture.texture, index)
            .expect("failed to set texture view in pool");
        Texture2d(Handle((range.base + slot) as u64))
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
        let index = (range.base + slot) as usize;
        self.pool
            .set_texture_view(&texture.texture, index)
            .expect("failed to set storage texture view in pool");
        Texture2dRW(Handle((range.base + slot) as u64))
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
        self.layer
            .set_drawable_size(width as usize, height as usize);
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
    /// `source` is generated MSL, as bytes, the same shape the Vulkan side takes SPIR-V in.
    pub fn new(device: &Device,entries: &[&'static crate::ir::GpuFn]) -> Result<Self, Error> {
        let source = crate::msl::module(entries);
        let source = std::str::from_utf8(source.as_bytes())
            .map_err(|_| Error::new("shader source is not valid UTF-8"))?;
        let compiler = device
            .device
            .new_m4_compiler_with_descriptor(&M4CompilerDescriptor::new())?;
        let desc = M4LibraryDescriptor::new();
        desc.set_source(source);
        let library = compiler
            .new_library_with_descriptor(&desc)
            .map_err(|error| {
                Error::new(crate::msl::format_compile_error(source, &error.to_string()))
            })?;
        Ok(Self {
            library,
            compiler,
            render: Vec::new(),
            compute: Vec::new(),
        })
    }

    fn function(&self, name: &str, values: Option<&dyn Constants>) -> M4FunctionDescriptor {
        // The emitter renames anything MSL reserves, so a pipeline naming its stage `vertex`
        // has to ask the library for the same spelling `msl.rs` printed.
        let name = &crate::msl::escape(name);
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
        pipeline.set_support_indirect_command_buffers(indirect(desc.indirect));
        pipeline.set_rasterization_enabled(true);
        pipeline.set_input_primitive_topology(topology_class(desc.topology));
        pipeline.set_raster_sample_count(desc.samples as usize);
        pipeline.set_alpha_to_coverage_state(M4AlphaToCoverageState::Disabled);
        // `desc.depth` is deliberately unused. `MTL4RenderPipelineDescriptor` carries no depth
        // attachment format, since Metal 4 takes it from the render pass. Vulkan bakes it
        // into `VkPipelineRenderingCreateInfo` and refuses to render without a match.

        color_targets(&pipeline.color_attachments(), desc.colors);

        let state = self
            .compiler
            .new_render_pipeline_state(&pipeline.as_pipeline_descriptor(), None)?;
        let raw = state.raw;
        let handle = pipeline_handle(&state, desc.indirect)?;
        self.render.push(state);
        Ok(Pipeline {
            raw,
            primitive: topology(desc.topology),
            handle,
        })
    }

    /// A mesh pipeline. Metal-only; the Vulkan backend answers with an error until
    /// `VK_EXT_mesh_shader` is requested at device creation.
    pub fn mesh(&mut self, desc: &MeshDesc) -> Result<Pipeline, Error> {
        let pipeline = M4MeshRenderPipelineDescriptor::new();
        pipeline.set_label(desc.mesh);
        if !desc.object.is_empty() {
            pipeline
                .set_object_function_descriptor(Some(&self.function(desc.object, desc.constants)));
            pipeline.set_payload_memory_length(desc.payload);
        }
        pipeline.set_mesh_function_descriptor(Some(&self.function(desc.mesh, desc.constants)));
        if !desc.fragment.is_empty() {
            pipeline.set_fragment_function_descriptor(Some(
                &self.function(desc.fragment, desc.constants),
            ));
        }
        pipeline.set_support_indirect_command_buffers(indirect(desc.indirect));
        pipeline.set_rasterization_enabled(!desc.fragment.is_empty());
        pipeline.set_raster_sample_count(desc.samples as usize);
        pipeline.set_alpha_to_coverage_state(M4AlphaToCoverageState::Disabled);
        color_targets(&pipeline.color_attachments(), desc.colors);

        let state = self
            .compiler
            .new_render_pipeline_state(&pipeline.as_pipeline_descriptor(), None)?;
        let raw = state.raw;
        let handle = pipeline_handle(&state, desc.indirect)?;
        self.render.push(state);
        Ok(Pipeline {
            raw,
            // A mesh pipeline declares its own topology in the shader, so nothing is picked here.
            primitive: PrimitiveType::Triangle,
            handle,
        })
    }

    /// A tile pipeline: a compute function that runs inside a render pass over the imageblock.
    /// Apple-only for good. See [`TileDesc`].
    pub fn tile(&mut self, desc: &TileDesc) -> Result<Pipeline, Error> {
        let pipeline = M4TileRenderPipelineDescriptor::new();
        pipeline.set_label(desc.function);
        pipeline.set_tile_function_descriptor(Some(&self.function(desc.function, desc.constants)));
        pipeline.set_raster_sample_count(desc.samples as usize);
        let attachments = pipeline.color_attachments();
        for (index, &target) in desc.colors.iter().enumerate() {
            attachments
                .object_at(index)
                .set_pixel_format(format(target));
        }

        let state = self
            .compiler
            .new_render_pipeline_state(&pipeline.as_pipeline_descriptor(), None)?;
        let raw = state.raw;
        self.render.push(state);
        Ok(Pipeline {
            raw,
            primitive: PrimitiveType::Triangle,
            handle: PipelineHandle(Handle(0)),
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
            handle: PipelineHandle(Handle(0)),
        })
    }
}

// -- passes ------------------------------------------------------------------------------

/// The stages a compute encoder can name about itself. Anything else in `stage`'s output
/// belongs to a render encoder and is only meaningful on the *queue* half of a barrier.
fn encoder_stages(stages: Stages) -> Stages {
    let compute = Stages::DISPATCH | Stages::BLIT | Stages::ACCELERATION_STRUCTURE;
    match Stages(stages.0 & compute.0) {
        Stages::NONE => compute,
        kept => kept,
    }
}

fn indirect(enabled: bool) -> M4IndirectCommandBufferSupportState {
    match enabled {
        true => M4IndirectCommandBufferSupportState::Enabled,
        false => M4IndirectCommandBufferSupportState::Disabled,
    }
}

/// A pipeline only has a resource ID when it was built to be encoded into an indirect command
/// buffer, so asking unconditionally would fail on every ordinary pipeline.
fn pipeline_handle(
    state: &minmetal::RenderPipelineState,
    indirect: bool,
) -> Result<PipelineHandle, Error> {
    match indirect {
        true => Ok(PipelineHandle(Handle(state.gpu_resource_id()?.impl_))),
        false => Ok(PipelineHandle(Handle(0))),
    }
}

fn size(dims: [u32; 3]) -> Size {
    Size::new(dims[0] as usize, dims[1] as usize, dims[2] as usize)
}

/// The colour-attachment half of a pipeline descriptor. Raster and mesh pipelines take the same
/// array type and the same `ColorTarget`s, and the blend translation is twenty lines either way.
fn color_targets(
    attachments: &M4RenderPipelineColorAttachmentDescriptorArray,
    colors: &[ColorTarget],
) {
    for (index, target) in colors.iter().enumerate() {
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
}

pub struct Pass {
    pub(crate) desc: M4RenderPassDescriptor,
    pub(crate) occlusion_base: u32,
    pub(crate) layers: u32,
    pub(crate) available_layers: u32,
}

impl Default for Pass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass {
    pub fn new() -> Self {
        let desc = M4RenderPassDescriptor::new();
        desc.set_render_target_array_length(1);
        Self {
            desc,
            occlusion_base: 0,
            layers: 1,
            available_layers: u32::MAX,
        }
    }

    /// Render each primitive into one of `layers` consecutive attachment slices. The vertex
    /// shader selects the relative slice with `render_target_array_index`.
    pub fn layers(mut self, layers: u32) -> Self {
        assert!(layers > 0, "a pass must render at least one layer");
        assert!(
            layers <= self.available_layers,
            "the pass asks for {layers} layers but an attachment has only {} available",
            self.available_layers
        );
        self.layers = layers;
        self.desc.set_render_target_array_length(layers as usize);
        self
    }

    fn use_attachment(mut self, available_layers: u32) -> Self {
        assert!(
            self.layers <= available_layers,
            "the pass asks for {} layers but this attachment has only {available_layers} available",
            self.layers
        );
        self.available_layers = self.available_layers.min(available_layers);
        self
    }

    /// Size the tile and its imageblock for a tile pipeline. Apple-only; see [`TileDesc`].
    ///
    /// `imageblock_bytes` is the per-sample struct the tile function reads and writes, and it
    /// has to match what the MSL declares or the pass will not encode.
    pub fn tile(self, width: u32, height: u32, imageblock_bytes: usize) -> Self {
        self.desc.set_tile_width(width as usize);
        self.desc.set_tile_height(height as usize);
        self.desc.set_imageblock_sample_length(imageblock_bytes);
        self
    }

    /// Count samples that pass the depth test into `queries` for the draws in this pass.
    /// Which slot each draw counts into is [`Raster::occlusion`].
    pub fn occlusion(self, queries: &Occlusion) -> Self {
        self.desc
            .set_visibility_result_buffer(Some(&queries.buffer));
        // `Reset` zeroes the *whole* buffer at pass start, which would wipe the other frame's
        // half before the host has read it. `Occlusion::begin` clears this frame's half itself,
        // so accumulating onto a known zero is the same measurement without the collateral.
        self.desc
            .set_visibility_result_type(VisibilityResultType::Accumulate);
        Self {
            occlusion_base: queries.base as u32,
            ..self
        }
    }

    pub fn color(self, index: usize, drawable: &Drawable, clear: Option<[f32; 4]>) -> Self {
        self.attach(index, &drawable.texture, 0, 0, 1, clear)
    }

    /// Renders into a texture rather than the surface. The two differ only in where the texture
    /// came from, so offscreen rendering needs no separate pass type. Unlike Vulkan there
    /// is no tracked layout to update either, because there is no layout.
    pub fn color_texture(
        self,
        index: usize,
        texture: &HeapTexture,
        clear: Option<[f32; 4]>,
    ) -> Self {
        self.color_texture_at(index, texture, 0, 0, clear)
    }

    /// Render into one mip level, starting at `layer`. A layered pass uses consecutive slices
    /// from that point.
    pub fn color_texture_at(
        self,
        index: usize,
        texture: &HeapTexture,
        level: u32,
        layer: u32,
        clear: Option<[f32; 4]>,
    ) -> Self {
        assert!(level < texture.mips, "mip level is outside the texture");
        assert!(layer < texture.layers, "array layer is outside the texture");
        self.attach(
            index,
            &texture.texture,
            level,
            layer,
            texture.layers - layer,
            clear,
        )
    }

    fn attach(
        mut self,
        index: usize,
        texture: &Texture,
        level: u32,
        layer: u32,
        available_layers: u32,
        clear: Option<[f32; 4]>,
    ) -> Self {
        self = self.use_attachment(available_layers);
        let attachment = self
            .desc
            .color_attachments()
            .object_at_indexed_subscript(index);
        attachment.base().set_texture(Some(texture));
        attachment.base().set_level(level as usize);
        attachment.base().set_slice(layer as usize);
        attachment
            .base()
            .set_store_action(minmetal::StoreAction::Store);
        match clear {
            Some([r, g, b, a]) => {
                attachment
                    .base()
                    .set_load_action(minmetal::LoadAction::Clear);
                attachment.set_clear_color(minmetal::ClearColor::new(
                    r as f64, g as f64, b as f64, a as f64,
                ));
            }
            None => attachment
                .base()
                .set_load_action(minmetal::LoadAction::Load),
        }
        self
    }

    /// Collapse multisample colour attachment `index` into the surface at pass end. The pass
    /// still needs the multisample texture bound with [`Pass::color_texture`]; this only names
    /// where it lands.
    pub fn resolve(self, index: usize, drawable: &Drawable) -> Self {
        self.attach_resolve(index, &drawable.texture, 0, 0, 1)
    }

    /// Resolve into a texture rather than the surface.
    pub fn resolve_texture(self, index: usize, texture: &HeapTexture) -> Self {
        self.resolve_texture_at(index, texture, 0, 0)
    }

    pub fn resolve_texture_at(
        self,
        index: usize,
        texture: &HeapTexture,
        level: u32,
        layer: u32,
    ) -> Self {
        assert!(level < texture.mips, "mip level is outside the texture");
        assert!(layer < texture.layers, "array layer is outside the texture");
        self.attach_resolve(
            index,
            &texture.texture,
            level,
            layer,
            texture.layers - layer,
        )
    }

    fn attach_resolve(
        mut self,
        index: usize,
        texture: &Texture,
        level: u32,
        layer: u32,
        available_layers: u32,
    ) -> Self {
        self = self.use_attachment(available_layers);
        let attachment = self
            .desc
            .color_attachments()
            .object_at_indexed_subscript(index);
        attachment.base().set_resolve_texture(Some(texture));
        attachment.base().set_resolve_level(level as usize);
        attachment.base().set_resolve_slice(layer as usize);
        // The multisample texture itself is never read again, so it is resolved rather than
        // stored. Vulkan says the same thing by giving the resolve target its own store op.
        attachment
            .base()
            .set_store_action(minmetal::StoreAction::MultisampleResolve);
        self
    }

    /// Collapse the multisample depth attachment into a single-sample texture. Unlike colour
    /// the mode is not implied. See [`ResolveMode`].
    pub fn depth_resolve(self, texture: &HeapTexture, mode: ResolveMode) -> Self {
        self.depth_resolve_at(texture, 0, 0, mode)
    }

    pub fn depth_resolve_at(
        mut self,
        texture: &HeapTexture,
        level: u32,
        layer: u32,
        mode: ResolveMode,
    ) -> Self {
        assert!(level < texture.mips, "mip level is outside the texture");
        assert!(layer < texture.layers, "array layer is outside the texture");
        self = self.use_attachment(texture.layers - layer);
        let attachment = self.desc.depth_attachment();
        attachment
            .base()
            .set_resolve_texture(Some(&texture.texture));
        attachment.base().set_resolve_level(level as usize);
        attachment.base().set_resolve_slice(layer as usize);
        attachment
            .base()
            .set_store_action(minmetal::StoreAction::MultisampleResolve);
        attachment
            .set_depth_resolve_filter(match mode {
                ResolveMode::SampleZero => MultisampleDepthResolveFilter::Sample0,
                ResolveMode::Min => MultisampleDepthResolveFilter::Min,
                ResolveMode::Max => MultisampleDepthResolveFilter::Max,
            })
            .expect("this OS has no depthResolveFilter");
        self
    }

    pub fn depth(self, texture: &HeapTexture, clear: Option<f32>, store: bool) -> Self {
        self.depth_at(texture, 0, 0, clear, store)
    }

    pub fn depth_at(
        mut self,
        texture: &HeapTexture,
        level: u32,
        layer: u32,
        clear: Option<f32>,
        store: bool,
    ) -> Self {
        assert!(level < texture.mips, "mip level is outside the texture");
        assert!(layer < texture.layers, "array layer is outside the texture");
        self = self.use_attachment(texture.layers - layer);
        let attachment = self.desc.depth_attachment();
        attachment.base().set_texture(Some(&texture.texture));
        attachment.base().set_level(level as usize);
        attachment.base().set_slice(layer as usize);
        attachment.base().set_store_action(match store {
            true => minmetal::StoreAction::Store,
            false => minmetal::StoreAction::DontCare,
        });
        match clear {
            Some(depth) => {
                attachment
                    .base()
                    .set_load_action(minmetal::LoadAction::Clear);
                attachment.set_clear_depth(depth as f64);
            }
            None => attachment
                .base()
                .set_load_action(minmetal::LoadAction::Load),
        }
        self
    }

    pub fn depth_stencil(
        self,
        texture: &HeapTexture,
        depth_clear: Option<f32>,
        stencil_clear: Option<u32>,
        store: bool,
    ) -> Self {
        self.depth_stencil_at(texture, 0, 0, depth_clear, stencil_clear, store)
    }

    pub fn depth_stencil_at(
        self,
        texture: &HeapTexture,
        level: u32,
        layer: u32,
        depth_clear: Option<f32>,
        stencil_clear: Option<u32>,
        store: bool,
    ) -> Self {
        assert!(
            texture.format().has_stencil(),
            "Pass::depth_stencil needs a stencil format"
        );
        let pass = self.depth_at(texture, level, layer, depth_clear, store);
        let attachment = pass.desc.stencil_attachment();
        attachment.base().set_texture(Some(&texture.texture));
        attachment.base().set_level(level as usize);
        attachment.base().set_slice(layer as usize);
        attachment.base().set_store_action(match store {
            true => minmetal::StoreAction::Store,
            false => minmetal::StoreAction::DontCare,
        });
        match stencil_clear {
            Some(stencil) => {
                attachment
                    .base()
                    .set_load_action(minmetal::LoadAction::Clear);
                attachment.set_clear_stencil(stencil);
            }
            None => attachment
                .base()
                .set_load_action(minmetal::LoadAction::Load),
        }
        pass
    }
}

// -- recording ---------------------------------------------------------------------------

pub struct Raster<'a> {
    pub(crate) encoder: M4RenderCommandEncoder,
    pub(crate) table: &'a M4ArgumentTable,
    pub(crate) arena: &'a Arena,
    pub(crate) depth_states: &'a [MetalDepthStencilState; 16],
    pub(crate) primitive: Cell<PrimitiveType>,
    pub(crate) occlusion_base: u32,
    pub(crate) resources: Cell<(u64, u32)>,
}

impl<'a> Raster<'a> {
    pub(crate) fn new(
        command_buffer: &'a M4CommandBuffer,
        table: &'a M4ArgumentTable,
        arena: &'a Arena,
        depth_states: &'a [MetalDepthStencilState; 16],
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
            occlusion_base: pass.occlusion_base,
            resources: Cell::new((0, 0)),
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
        raster.stencil_reference(0);
        raster
    }

    /// Writes the root pointer where the shader expects it. `msl.rs` gives every entry point
    /// a `constant nga_push* [[buffer(0)]]` parameter, so what the argument table holds
    /// is the address of a struct, not the root itself, so a bind costs one 16-byte arena
    /// allocation.
    fn bind<T>(&self, root: Gpu<T>) {
        let (resources, resources_stride) = self.resources.get();
        bind_push(
            self.table,
            self.arena,
            root.0.get(),
            resources,
            resources_stride,
        );
    }

    pub fn pipeline(&self, pipeline: Pipeline) {
        self.primitive.set(pipeline.primitive);
        let state = ManuallyDrop::new(minmetal::RenderPipelineState { raw: pipeline.raw });
        self.encoder.set_render_pipeline_state(&state);
    }

    pub fn heap(&self, heap: &ResourceHeap) {
        self.resources.set((heap.base_id, heap.stride));
    }

    pub fn depth(&self, state: &DepthState) {
        assert!(
            state.stencil.is_none(),
            "create stencil states with Device::depth_stencil_state"
        );
        self.encoder
            .set_depth_stencil_state(&self.depth_states[depth_index(state)]);
    }

    #[doc(hidden)]
    pub fn depth_stencil(&self, state: &DepthStencilState) {
        self.encoder.set_depth_stencil_state(&state.0);
    }

    pub fn stencil_reference(&self, reference: u32) {
        self.encoder.set_stencil_reference_value(reference);
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
        root: Gpu<T>,
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

    pub fn draw_indexed_indirect<I: IndexFormat, T>(
        &self,
        root: Gpu<T>,
        indices: &HeapBuffer,
        _arena: &Arena,
        draws: Gpu<[DrawIndexed]>,
        count: u32,
    ) {
        self.bind(root);
        let index_type = match I::WIDE {
            true => IndexType::UInt32,
            false => IndexType::UInt16,
        };
        for index in 0..count as u64 {
            self.encoder.draw_indexed_primitives_with_indirect_buffer(
                self.primitive.get(),
                index_type,
                indices.gpu,
                indices.len,
                draws.0.get() + index * size_of::<DrawIndexed>() as u64,
            );
        }
    }

    /// Replays a GPU-compacted batch with one driver command.
    ///
    /// `range` begins with Metal's `{ location, length }` execution range. The producer
    /// increments `length` and encodes the same compact slot through [`command_draw`].
    /// `draws` is the parallel Vulkan argument array and is unused on Metal.
    pub fn draw_indirect_count<T>(
        &self,
        root: Gpu<T>,
        commands: &IndirectCommands,
        _draws: &HeapBuffer,
        range: &HeapBuffer,
    ) {
        debug_assert!(range.len >= size_of::<[u32; 2]>());
        self.bind(root);
        self.encoder
            .execute_commands_in_buffer_with_indirect_buffer(&commands.icb, range.gpu);
    }

    /// A run of draws sharing every piece of state, including the root pointer.
    ///
    /// `VK_EXT_multi_draw` collapses this into one driver call; Metal has no such call, so it
    /// is a loop and `Device::supports(Feature::MultiDraw)` answers `false`. The result is
    /// identical, which is why the method exists at all. Portable code should not have to
    /// carry two spellings of the same draw.
    pub fn draw_multi<T>(&self, root: Gpu<T>, draws: &[MultiDraw]) {
        self.bind(root);
        for draw in draws {
            self.encoder.draw_primitives(
                self.primitive.get(),
                draw.first_vertex as usize,
                draw.vertex_count as usize,
            );
        }
    }

    /// The indexed counterpart of [`Raster::draw_multi`]. `first_index` is applied by moving
    /// the index-buffer address, which is why the buffer length shrinks with it.
    ///
    /// `vertex_offset` must be zero: `MTL4RenderCommandEncoder`'s indexed draw takes no base
    /// vertex, and silently ignoring the field would be worse than refusing it.
    pub fn draw_multi_indexed<T, I: IndexFormat>(
        &self,
        root: Gpu<T>,
        indices: &HeapBuffer,
        draws: &[MultiDrawIndexed],
    ) {
        self.bind(root);
        let width = match I::WIDE {
            true => 4,
            false => 2,
        };
        for draw in draws {
            debug_assert_eq!(
                draw.vertex_offset, 0,
                "Metal's indexed draw has no base vertex; bake the offset into the indices"
            );
            let offset = draw.first_index as usize * width;
            self.encoder.draw_indexed_primitives(
                self.primitive.get(),
                draw.index_count as usize,
                match I::WIDE {
                    true => IndexType::UInt32,
                    false => IndexType::UInt16,
                },
                indices.gpu + offset as u64,
                indices.len - offset,
            );
        }
    }

    /// Runs a mesh pipeline: `groups` object threadgroups, each amplifying into mesh
    /// threadgroups of `mesh_group` threads. With no object stage, `groups` is the mesh grid
    /// directly and `object_group` is ignored.
    ///
    /// Metal-only; ask [`Device::supports`] with [`Feature::MeshShaders`] first.
    pub fn draw_mesh<T>(
        &self,
        root: Gpu<T>,
        groups: [u32; 3],
        object_group: [u32; 3],
        mesh_group: [u32; 3],
    ) {
        self.bind(root);
        self.encoder
            .draw_mesh_threadgroups(size(groups), size(object_group), size(mesh_group));
    }

    /// Runs a tile pipeline over the pass's imageblock. One dispatch covers one tile, so the
    /// grid is the tile size the pass was given, not the render target. Apple-only.
    pub fn dispatch_tile<T>(&self, root: Gpu<T>, width: u32, height: u32) {
        self.bind(root);
        self.encoder
            .dispatch_threads_per_tile(Size::new(width as usize, height as usize, 1));
    }

    /// Replays draws a shader encoded into `commands`. The GPU wrote both the arguments and the
    /// pipeline selection, which is what an [`IndirectCommands`] buffer has over
    /// `draw_indirect_count`. Metal-only.
    pub fn execute(&self, commands: &IndirectCommands, range: std::ops::Range<u32>) {
        self.encoder.execute_commands_in_buffer(
            &commands.icb,
            minmetal::Range::new(range.start as usize, range.len()),
        );
    }

    /// Count the samples the following draws write into slot `index` of the [`Occlusion`] set
    /// the pass was given. Calling it again moves to another slot; [`Raster::occlusion_off`]
    /// stops counting.
    pub fn occlusion(&self, index: u32) {
        self.encoder.set_visibility_result_mode(
            VisibilityResultMode::Counting,
            (self.occlusion_base + index) as usize * size_of::<u64>(),
        );
    }

    pub fn occlusion_off(&self) {
        self.encoder
            .set_visibility_result_mode(VisibilityResultMode::Disabled, 0);
    }

    /// Records the GPU clock into slot `index`, after everything drawn so far in this pass has
    /// left the fragment stage. Vulkan spells the same thing `vkCmdWriteTimestamp2`.
    pub fn timestamp(&self, timers: &Timers, index: usize) {
        debug_assert!(index < timers.count);
        self.encoder.write_timestamp_with_granularity(
            M4TimestampGranularity::Precise,
            RenderStages::FRAGMENT,
            &timers.heap,
            timers.base + index,
        );
    }

    pub fn draw<T>(&self, root: Gpu<T>, start: u32, count: u32) {
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
    pub(crate) resources: Cell<(u64, u32)>,
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
            resources: Cell::new((0, 0)),
        }
    }

    fn bind<T>(&self, root: Gpu<T>) {
        let (resources, resources_stride) = self.resources.get();
        bind_push(
            self.table,
            self.arena,
            root.0.get(),
            resources,
            resources_stride,
        );
    }

    pub fn pipeline(&self, pipeline: Pipeline) {
        let state = ManuallyDrop::new(minmetal::ComputePipelineState { raw: pipeline.raw });
        self.encoder.set_compute_pipeline_state(&state);
    }

    pub fn heap(&self, heap: &ResourceHeap) {
        self.resources.set((heap.base_id, heap.stride));
    }

    pub fn dispatch<T>(&self, root: Gpu<T>, threads: usize, group: usize) {
        self.bind(root);
        self.encoder
            .dispatch_threads(Size::new(threads, 1, 1), Size::new(group, 1, 1));
    }

    pub fn dispatch_2d<T>(&self, root: Gpu<T>, width: u32, height: u32, group: u32) {
        self.bind(root);
        self.encoder.dispatch_threads(
            Size::new(width as usize, height as usize, 1),
            Size::new(group as usize, group as usize, 1),
        );
    }

    /// The whole barrier, with no resource list, exactly the design the post argues for. Metal
    /// spells it as stages plus one visibility bit, which is strictly less than
    /// `VkMemoryBarrier2` carries and lands in the same place.
    ///
    /// It takes *three* calls, because Metal 4 splits what one `vkCmdPipelineBarrier2` does
    /// into one in-encoder barrier and two queue-scoped halves, and a barrier that reaches only
    /// some of them is a race:
    ///
    /// - `barrier` orders against earlier commands in this encoder.
    /// - `barrierAfterQueueStages` orders this encoder after everything already committed. It
    ///   is the only thing that sees a preceding render pass.
    /// - `barrierBeforeQueueStages` publishes this encoder's writes to everything committed
    ///   after it. It is the only thing a following render pass sees.
    ///
    /// Both queue halves were found the same way, by a race the shared API cannot express
    /// any other way: without the first, the offscreen raster test failed six runs in twenty;
    /// without the second, `gpu_driven` drew bands of empty draw slots because the render
    /// encoder fetched indirect arguments the encode pass had not published yet.
    /// The three calls take a mix of encoder stages and queue stages, and an encoder stage
    /// has to be one this encoder actually has. A compute encoder naming `Stage::INDIRECT`,
    /// which is a vertex fetch on Metal, is a validation failure, and with the layer off it is
    /// a magenta frame with no diagnostic at all.
    pub fn barrier(&self, before: Stage, after: Stage, hazards: Hazard) {
        let (before, after, hazard) = (stage(before), stage(after), hazard(hazards));
        let (local_before, local_after) = (encoder_stages(before), encoder_stages(after));
        self.encoder.barrier(local_before, local_after, hazard);
        // Queue halves are only needed when a named stage lives on another encoder.
        // Compute→compute stays in this encoder; compute→vertex and color→transfer do not.
        let compute = Stages::DISPATCH | Stages::BLIT | Stages::ACCELERATION_STRUCTURE;
        if before.0 & !compute.0 != 0 {
            self.encoder
                .barrier_after_queue_stages(before, local_after, hazard);
        }
        if after.0 & !compute.0 != 0 {
            self.encoder
                .barrier_before_queue_stages(local_before, after, hazard);
        }
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
    pub fn read_texture(&self, texture: &HeapTexture, arena: &Arena, dest: GpuRw<[u8]>) {
        self.read_texture_at(texture, 0, 0, arena, dest)
    }

    /// Read one mip level of one array layer into tightly packed host-visible memory.
    pub fn read_texture_at(
        &self,
        texture: &HeapTexture,
        level: u32,
        layer: u32,
        arena: &Arena,
        dest: GpuRw<[u8]>,
    ) {
        assert!(level < texture.mips, "mip level is outside the texture");
        assert!(layer < texture.layers, "array layer is outside the texture");
        let offset = dest
            .0
            .get()
            .checked_sub(arena.base)
            .expect("the readback destination is not in this arena") as usize;
        let width = (texture.width >> level).max(1);
        let height = (texture.height >> level).max(1);
        let row = width as usize * texture.format.size();
        self.encoder.copy_texture_to_buffer(
            &texture.texture,
            layer as usize,
            level as usize,
            Origin::new(0, 0, 0),
            Size::new(width as usize, height as usize, 1),
            &arena.buffer,
            offset,
            row,
            row * height as usize,
            minmetal::BlitOption::NONE,
        );
    }

    /// Replays the commands a shader encoded into `commands`. Metal-only.
    pub fn execute(&self, commands: &IndirectCommands, range: std::ops::Range<u32>) {
        self.encoder.execute_commands_in_buffer(
            &commands.icb,
            minmetal::Range::new(range.start as usize, range.len()),
        );
    }

    /// Clears a range of encoded commands so the next frame starts from nothing. An unreset
    /// slot replays whatever it last held, which is the indirect-command-buffer equivalent of
    /// the stale `instance_count` `draw_indirect_count` has to guard against.
    pub fn reset(&self, commands: &IndirectCommands, range: std::ops::Range<u32>) {
        self.encoder.reset_commands_in_buffer(
            &commands.icb,
            minmetal::Range::new(range.start as usize, range.len()),
        );
    }

    /// Records the GPU clock into slot `index`, after everything dispatched so far in this
    /// encoder has finished.
    pub fn timestamp(&self, timers: &Timers, index: usize) {
        debug_assert!(index < timers.count);
        self.encoder.write_timestamp_with_granularity(
            M4TimestampGranularity::Precise,
            &timers.heap,
            timers.base + index,
        );
    }

    /// Metal's fill takes a byte, Vulkan's a word. The shared API takes the word and this
    /// splits it, which is exact for the only case either backend is used for: clearing.
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

fn bind_push(
    table: &M4ArgumentTable,
    arena: &Arena,
    root: u64,
    resources: u64,
    resources_stride: u32,
) {
    let (cpu, gpu) = arena
        .alloc_bytes(size_of::<Push>(), 16)
        .expect("frame arena is full: no room for a root binding");
    unsafe {
        (cpu as *mut Push).write(Push {
            root,
            resources,
            resources_stride,
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
    /// Every depth-only (compare, write) pair, indexed without a lookup while recording.
    pub(crate) depth_states: [MetalDepthStencilState; 16],
    pub(crate) threads: usize,
    pub(crate) frame: u64,
}

impl Context {
    /// `shaders` is what the Vulkan side takes its pipeline layout from. Metal binds through the
    /// argument table instead, so it is unused here. The argument stays for one signature.
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

        let depth_states = [
            metal_depth_state(&device.device, CompareOp::Never, false)?,
            metal_depth_state(&device.device, CompareOp::Never, true)?,
            metal_depth_state(&device.device, CompareOp::Less, false)?,
            metal_depth_state(&device.device, CompareOp::Less, true)?,
            metal_depth_state(&device.device, CompareOp::Equal, false)?,
            metal_depth_state(&device.device, CompareOp::Equal, true)?,
            metal_depth_state(&device.device, CompareOp::LessEqual, false)?,
            metal_depth_state(&device.device, CompareOp::LessEqual, true)?,
            metal_depth_state(&device.device, CompareOp::Greater, false)?,
            metal_depth_state(&device.device, CompareOp::Greater, true)?,
            metal_depth_state(&device.device, CompareOp::NotEqual, false)?,
            metal_depth_state(&device.device, CompareOp::NotEqual, true)?,
            metal_depth_state(&device.device, CompareOp::GreaterEqual, false)?,
            metal_depth_state(&device.device, CompareOp::GreaterEqual, true)?,
            metal_depth_state(&device.device, CompareOp::Always, false)?,
            metal_depth_state(&device.device, CompareOp::Always, true)?,
        ];

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

    /// How many frames have finished on the GPU. Frame `n` is complete once this exceeds `n`,
    /// which is exactly the comparison [`GpuHeap::retire`] makes.
    pub fn completed(&self) -> u64 {
        self.event.signaled_value()
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
    pub(crate) depth_states: &'a [MetalDepthStencilState; 16],
    pub(crate) recording: usize,
    pub(crate) value: u64,
}

impl<'a> Frame<'a> {
    /// This frame's per-thread arena. The bump is reset when this slot is recorded again, after
    /// the frames in flight have retired. A [`GpuBox`] borrows the mapping for as long as the
    /// arena does. Copying [`GpuBox::gpu`] out does not.
    ///
    /// The returned borrow is tied to the context, not to `self`, so allocating from it does
    /// not lock the frame. Recording can still take `&mut self` afterwards.
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
        self.queue.commit(&self.command_buffers[..self.recording]);
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
    pub(crate) depth_states: &'a [MetalDepthStencilState; 16],
    pub(crate) not_sync: PhantomData<Cell<()>>,
}

impl<'a> Recorder<'a> {
    /// This recorder's slice of the frame arena, where root data for the commands being
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

// -- fences ------------------------------------------------------------------------------

/// A monotonic counter both the host and the GPU can see: `MTLSharedEvent` here, a timeline
/// semaphore on Vulkan. Frames have their own counter inside [`Context`]; this is for the
/// sequencing a caller does itself, such as an asset load that must land before a frame reads
/// what it produced.
pub struct Fence(SharedEvent);

impl Fence {
    pub fn new(device: &Device) -> Result<Self, Error> {
        Ok(Self(device.device.new_shared_event()?))
    }

    /// Raise the counter from the host. Values only ever go up; signalling backwards is a
    /// programming error on both backends and is not checked.
    pub fn signal(&self, value: u64) {
        self.0.set_signaled_value(value);
    }

    pub fn value(&self) -> u64 {
        self.0.signaled_value()
    }

    /// Blocks until the counter reaches `value`, or ten seconds pass. `false` is the timeout,
    /// a hang rather than an error worth a `Result`, since there is nothing to do about it but
    /// say so.
    pub fn wait(&self, value: u64) -> bool {
        self.0.wait_until_signaled_value(value, 10_000)
    }
}

// -- indirect command buffers ------------------------------------------------------------

/// A buffer of draws or dispatches a *shader* encodes, pipeline selection included.
///
/// This is what `draw_indirect_count` is not: there, the GPU picks the arguments and the host
/// picks the pipeline, so every replayed draw shares one pipeline and one root pointer. Here a
/// compute shader writes whole commands, pipeline and buffers and primitive and counts, and
/// the render pass replays them.
///
/// Metal-only. Vulkan's nearest equivalent is `VK_NV_device_generated_commands`, which is not
/// in the shared API; ask [`Device::supports`] with [`Feature::IndirectCommands`].
pub struct IndirectCommands {
    pub(crate) icb: IndirectCommandBuffer,
    pub(crate) handle: Commands,
    pub(crate) count: u32,
}

impl IndirectCommands {
    /// A compact batch whose draw count is written by the same shader that encodes its slots.
    ///
    /// The render pass supplies the pipeline, root pointer, and dynamic state. The producer
    /// only calls [`command_draw`] for each compacted slot, so Metal can replay the batch with
    /// one counted ICB command rather than one CPU driver call per possible draw.
    pub fn counted_draws(device: &Device, count: u32) -> Result<Self, Error> {
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(IndirectCommandType::DRAW);
        desc.set_inherit_pipeline_state(true);
        desc.set_inherit_buffers(true);
        Self::inherit_render_state(&desc);
        Self::new(device, &desc, count)
    }

    /// `count` draw slots. `indexed` picks between `drawPrimitives` and
    /// `drawIndexedPrimitives`; a buffer cannot hold both.
    pub fn draws(device: &Device, count: u32, indexed: bool) -> Result<Self, Error> {
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(match indexed {
            true => IndirectCommandType::DRAW_INDEXED,
            false => IndirectCommandType::DRAW,
        });
        // The encoding shader names the pipeline in every command it writes, which is the
        // point of the whole mechanism. Inheriting instead would make the encoder's pipeline
        // the one that runs, and with none bound `executeCommandsInBuffer` dereferences a
        // null pipeline inside the driver rather than complaining.
        desc.set_inherit_pipeline_state(false);
        desc.set_inherit_buffers(false);
        desc.set_max_vertex_buffer_bind_count(2);
        desc.set_max_fragment_buffer_bind_count(2);
        Self::inherit_render_state(&desc);
        Self::new(device, &desc, count)
    }

    /// `count` mesh-pipeline draw slots.
    pub fn meshes(device: &Device, count: u32) -> Result<Self, Error> {
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(IndirectCommandType::DRAW_MESH_THREADGROUPS);
        desc.set_inherit_pipeline_state(false);
        desc.set_inherit_buffers(false);
        desc.set_max_object_buffer_bind_count(2);
        desc.set_max_mesh_buffer_bind_count(2);
        desc.set_max_fragment_buffer_bind_count(2);
        Self::inherit_render_state(&desc);
        Self::new(device, &desc, count)
    }

    /// `count` dispatch slots, for a compute pass replaying work another shader queued.
    pub fn dispatches(device: &Device, count: u32) -> Result<Self, Error> {
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
        desc.set_inherit_pipeline_state(false);
        desc.set_inherit_buffers(false);
        desc.set_max_kernel_buffer_bind_count(2);
        Self::new(device, &desc, count)
    }

    /// macOS 26 split six pieces of render state out of the encoder and into the command, each
    /// defaulting to not inherited. An MSL `render_command` has no setter for any of them, so
    /// leaving the defaults means every replayed draw runs with undefined cull, depth and fill
    /// state, which the driver renders as a magenta frame rather than reporting.
    fn inherit_render_state(desc: &IndirectCommandBufferDescriptor) {
        desc.set_inherit_depth_stencil_state(true);
        desc.set_inherit_depth_bias(true);
        desc.set_inherit_depth_clip_mode(true);
        desc.set_inherit_cull_mode(true);
        desc.set_inherit_front_facing_winding(true);
        desc.set_inherit_triangle_fill_mode(true);
    }

    fn new(
        device: &Device,
        desc: &IndirectCommandBufferDescriptor,
        count: u32,
    ) -> Result<Self, Error> {
        let icb = device.device.new_indirect_command_buffer(
            desc,
            count as usize,
            IndirectCommandBufferOptions::NONE,
        )?;
        device.make_resident(&Allocation::from_indirect_command_buffer(&icb))?;
        Ok(Self {
            handle: Commands(Handle(icb.gpu_resource_id()?.impl_)),
            icb,
            count,
        })
    }

    /// The handle a shader dereferences to encode into this buffer. Eight bytes, exactly like a
    /// texture or sampler handle, and written into the root struct the same way.
    pub fn handle(&self) -> Commands {
        self.handle
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    /// Every slot, for the common `execute(&commands, commands.all())` case.
    pub fn all(&self) -> std::ops::Range<u32> {
        0..self.count
    }
}

// -- queries -----------------------------------------------------------------------------

#[link(name = "System", kind = "dylib")]
unsafe extern "C" {
    fn mach_timebase_info(info: *mut [u32; 2]) -> i32;
}

/// `count` GPU timestamps per frame, doubled so a frame in flight never overwrites the results
/// the host has not read yet. `begin` picks this frame's half and resolves the other one, so
/// the numbers a caller reads are always from `FRAMES_IN_FLIGHT` frames ago and never stall.
pub struct Timers {
    pub(crate) heap: M4CounterHeap,
    pub(crate) count: usize,
    pub(crate) base: usize,
    pub(crate) ticks: Vec<u64>,
    pub(crate) ns_per_tick: f64,
}

impl Timers {
    pub fn new(device: &Device, count: usize) -> Result<Self, Error> {
        let total = count * FRAMES_IN_FLIGHT as usize;
        let desc = M4CounterHeapDescriptor::new();
        desc.set_type(M4CounterHeapType::Timestamp);
        desc.set_count(total);
        let heap = device.device.new_m4_counter_heap_with_descriptor(&desc)?;
        heap.invalidate_counter_range(minmetal::Range::new(0, total));
        // Metal reports timestamps in mach ticks, which are not nanoseconds on Apple silicon.
        let mut timebase = [0u32; 2];
        debug_assert_eq!(unsafe { mach_timebase_info(&mut timebase) }, 0);
        Ok(Self {
            heap,
            count,
            base: 0,
            ticks: vec![0; count],
            ns_per_tick: timebase[0] as f64 / timebase[1] as f64,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Take this frame's half of the heap and read back what the last frame to use it recorded.
    /// Call once per frame, after `begin_frame`. That wait is what makes the read safe.
    ///
    /// Only slots written every frame are meaningful. There is deliberately no per-frame
    /// `invalidateCounterRange:`: it is queue-ordered, not immediate, and issuing it here raced
    /// timestamps the GPU had already recorded, and one run in four came back empty.
    pub fn begin(&mut self, frame: &Frame) {
        self.base = ((frame.value - 1) % FRAMES_IN_FLIGHT) as usize * self.count;
        self.ticks.fill(0);
        let range = minmetal::Range::new(self.base, self.count);
        if let Some(data) = self.heap.resolve_counter_range(range) {
            for (tick, entry) in self.ticks.iter_mut().zip(data.chunks_exact(8)) {
                *tick = u64::from_ne_bytes(entry.try_into().unwrap());
            }
        }
    }

    /// Whether slot `index` holds a timestamp from the last frame to use this half. A slot that
    /// was never written reads as zero on both backends, and zero is also what [`Timers::ns`]
    /// reports for it, so this is how to tell the two apart.
    pub fn written(&self, index: usize) -> bool {
        self.ticks[index] != 0
    }

    /// Nanoseconds between two slots. Zero if either was never written or the pair is out of
    /// order, which is what an unwritten slot looks like on both backends.
    pub fn ns(&self, start: usize, end: usize) -> f64 {
        let (start, end) = (self.ticks[start], self.ticks[end]);
        match start == 0 || end < start {
            true => 0.0,
            false => (end - start) as f64 * self.ns_per_tick,
        }
    }

    pub fn ms(&self, start: usize, end: usize) -> f64 {
        self.ns(start, end) / 1e6
    }
}

/// `count` occlusion counters per frame, doubled the same way [`Timers`] is.
///
/// Metal counts straight into a visibility-result buffer the pass names; Vulkan counts into a
/// query pool and the results are fetched at `begin`. Either way what a caller sees is the
/// sample count from `FRAMES_IN_FLIGHT` frames ago.
pub struct Occlusion {
    pub(crate) buffer: Buffer,
    pub(crate) count: u32,
    pub(crate) base: u32,
    pub(crate) samples: Vec<u64>,
}

impl Occlusion {
    pub fn new(device: &Device, count: u32) -> Result<Self, Error> {
        let total = count as usize * FRAMES_IN_FLIGHT as usize;
        let buffer = device.device.new_buffer(
            total * size_of::<u64>(),
            ResourceOptions::new(
                CpuCacheMode::DefaultCache,
                StorageMode::Shared,
                HazardTrackingMode::Untracked,
            ),
        )?;
        device.make_resident(&Allocation::from_buffer(&buffer))?;
        unsafe { std::ptr::write_bytes(buffer.contents() as *mut u8, 0, total * size_of::<u64>()) };
        Ok(Self {
            buffer,
            count,
            base: 0,
            samples: vec![0; count as usize],
        })
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    /// Take this frame's half of the buffer and read back what the last frame to use it counted.
    pub fn begin(&mut self, frame: &Frame) {
        self.base = ((frame.value - 1) % FRAMES_IN_FLIGHT) as u32 * self.count;
        let counters = unsafe { (self.buffer.contents() as *mut u64).add(self.base as usize) };
        for (index, out) in self.samples.iter_mut().enumerate() {
            unsafe {
                *out = counters.add(index).read();
                counters.add(index).write(0);
            }
        }
    }

    /// Samples that passed the depth test in slot `index`, from the last frame to use this
    /// half of the buffer.
    pub fn samples(&self, index: u32) -> u64 {
        self.samples[index as usize]
    }
}
