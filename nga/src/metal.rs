use crate::{
    AddressMode, Alloc, BlendFactor, BlendMode, ColorTarget, Commands, CompareOp, ComputeShader,
    Constant, Constants, Cull, DepthState, DispatchThreads, DrawIndexed, Error, FRAMES_IN_FLIGHT,
    Feature, Filter, Format, FragmentShader, Gpu, GpuBox, GpuRw, Handle, Hazard, HeapUsage, IndexFormat,
    Memory, MeshDesc, PipelineHandle, RasterBuilder, RasterDesc, ResolveMode, Sampler, SamplerDesc,
    Scalar, Scissor, Stage, StencilFace, StencilOp, Texture2d, Texture2dRW, TextureCube, TextureCubeRW, TileDesc, Tlsf,
    Topology, Usage, VertexShader, Viewport, Winding,
};
use minmetal::{
    Allocation, Buffer, ColorWriteMask, CompareFunction, CpuCacheMode, CullMode,
    DepthStencilDescriptor, DepthStencilState as MetalDepthStencilState, FunctionConstantValues,
    HazardTrackingMode, Heap, HeapDescriptor, HeapType, IndexType, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, IndirectCommandBufferOptions, IndirectCommandType,
    M4AlphaToCoverageState, M4ArgumentTable, M4ArgumentTableDescriptor, M4BlendState,
    M4CommandAllocator, M4CommandBuffer, M4CommandQueue, M4CommitFeedbackHandler, M4CommitOptions,
    M4Compiler, M4CompilerDescriptor, M4ComputeCommandEncoder, M4ComputePipelineDescriptor,
    M4CounterHeap, M4CounterHeapDescriptor, M4CounterHeapType, M4FunctionDescriptor,
    M4IndirectCommandBufferSupportState, M4LibraryDescriptor, M4LibraryFunctionDescriptor,
    M4MeshRenderPipelineDescriptor, M4RenderCommandEncoder, M4RenderPassDescriptor,
    M4RenderPipelineColorAttachmentDescriptorArray, M4RenderPipelineDescriptor,
    M4SpecializedFunctionDescriptor, M4TileRenderPipelineDescriptor, M4TimestampGranularity,
    M4VisibilityOptions, MetalError, MetalLayer, MultisampleDepthResolveFilter, Origin,
    PixelFormat, PrimitiveTopologyClass, PrimitiveType, Region, RenderStages, ResidencySet,
    ResidencySetDescriptor, ResourceOptions, SamplerAddressMode,
    SamplerDescriptor, SamplerMinMagFilter, SamplerMipFilter, SharedEvent, Size, Stages,
    StencilDescriptor, StencilOperation, StorageMode, Texture, TextureDescriptor, TextureType,
    TextureUsage, VisibilityResultMode, VisibilityResultType, id,
};
use std::cell::{Cell, UnsafeCell};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pipeline {
    raw: id,
    group: [u32; 3],

    primitive: PrimitiveType,

    handle: PipelineHandle,
}

impl Pipeline {

    pub fn handle(&self) -> PipelineHandle {
        self.handle
    }
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

pub struct SamplerState(minmetal::SamplerState);

#[doc(hidden)]
pub struct DepthStencilState(MetalDepthStencilState);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MultiDraw {
    pub first_vertex: u32,
    pub vertex_count: u32,
}

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
        Format::Bc1RgbaUnorm => PixelFormat::Bc1Rgba,
        Format::Bc1RgbaUnormSrgb => PixelFormat::Bc1RgbaSrgb,
        Format::Bc2RgbaUnorm => PixelFormat::Bc2Rgba,
        Format::Bc2RgbaUnormSrgb => PixelFormat::Bc2RgbaSrgb,
        Format::Bc3RgbaUnorm => PixelFormat::Bc3Rgba,
        Format::Bc3RgbaUnormSrgb => PixelFormat::Bc3RgbaSrgb,
        Format::Bc4RUnorm => PixelFormat::Bc4RUnorm,
        Format::Bc4RSnorm => PixelFormat::Bc4RSnorm,
        Format::Bc5RgUnorm => PixelFormat::Bc5RgUnorm,
        Format::Bc5RgSnorm => PixelFormat::Bc5RgSnorm,
        Format::Bc6hRgbFloat => PixelFormat::Bc6HRgbFloat,
        Format::Bc6hRgbUfloat => PixelFormat::Bc6HRgbUfloat,
        Format::Bc7RgbaUnorm => PixelFormat::Bc7RgbaUnorm,
        Format::Bc7RgbaUnormSrgb => PixelFormat::Bc7RgbaUnormSrgb,
        Format::Astc4x4Unorm => PixelFormat::Astc4x4Ldr,
        Format::Astc4x4UnormSrgb => PixelFormat::Astc4x4Srgb,
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
        PixelFormat::Bc1Rgba => Format::Bc1RgbaUnorm,
        PixelFormat::Bc1RgbaSrgb => Format::Bc1RgbaUnormSrgb,
        PixelFormat::Bc2Rgba => Format::Bc2RgbaUnorm,
        PixelFormat::Bc2RgbaSrgb => Format::Bc2RgbaUnormSrgb,
        PixelFormat::Bc3Rgba => Format::Bc3RgbaUnorm,
        PixelFormat::Bc3RgbaSrgb => Format::Bc3RgbaUnormSrgb,
        PixelFormat::Bc4RUnorm => Format::Bc4RUnorm,
        PixelFormat::Bc4RSnorm => Format::Bc4RSnorm,
        PixelFormat::Bc5RgUnorm => Format::Bc5RgUnorm,
        PixelFormat::Bc5RgSnorm => Format::Bc5RgSnorm,
        PixelFormat::Bc6HRgbFloat => Format::Bc6hRgbFloat,
        PixelFormat::Bc6HRgbUfloat => Format::Bc6hRgbUfloat,
        PixelFormat::Bc7RgbaUnorm => Format::Bc7RgbaUnorm,
        PixelFormat::Bc7RgbaUnormSrgb => Format::Bc7RgbaUnormSrgb,
        PixelFormat::Astc4x4Ldr => Format::Astc4x4Unorm,
        PixelFormat::Astc4x4Srgb => Format::Astc4x4UnormSrgb,
        _ => return None,
    })
}

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

pub(crate) fn stage(stage: Stage) -> Stages {
    let mut out = Stages::NONE;
    if stage.has(Stage::TRANSFER) {
        out = out | Stages::BLIT;
    }
    if stage.has(Stage::COMPUTE) {
        out = out | Stages::DISPATCH;
    }

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

pub(crate) fn hazard(hazard: Hazard) -> M4VisibilityOptions {
    match hazard {
        Hazard::NONE => M4VisibilityOptions::NONE,
        _ => M4VisibilityOptions::DEVICE,
    }
}

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

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Push {
    root: u64,
    resources: u64,
}

pub struct Device {
    pub(crate) device: minmetal::Device,
    pub(crate) queue: M4CommandQueue,
    pub(crate) residency: ResidencySet,

    pub(crate) idle_event: SharedEvent,
    pub(crate) idle: AtomicU64,
    pub(crate) name: String,
    submission_error: Arc<OnceLock<String>>,
}

impl Device {

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
            submission_error: Arc::new(OnceLock::new()),
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

    pub fn allocated_bytes(&self) -> Option<usize> { Some(self.device.current_allocated_size()) }

    pub fn start_capture(&self, path: &std::path::Path) -> Result<(), Error> {
        let manager = minmetal::CaptureManager::shared()?;
        if manager.is_capturing() { return Err(Error::new("a Metal capture is already active")); }
        if !manager.supports_destination(minmetal::CaptureDestination::GpuTraceDocument) {
            return Err(Error::new("Metal file capture is unavailable; launch with MTL_CAPTURE_ENABLED=1"));
        }
        let descriptor = minmetal::CaptureDescriptor::new();
        descriptor.set_capture_object(self.device.raw);
        descriptor.set_destination(minmetal::CaptureDestination::GpuTraceDocument);
        let absolute = if path.is_absolute() { path.to_path_buf() } else {
            std::env::current_dir().map_err(|e| Error::new(e.to_string()))?.join(path)
        };
        descriptor.set_output_url(absolute.to_str().ok_or_else(|| Error::new("capture path must be UTF-8"))?);
        manager.start_capture(&descriptor)?;
        Ok(())
    }

    pub fn stop_capture(&self) -> Result<(), Error> {
        minmetal::CaptureManager::shared()?.stop_capture();
        Ok(())
    }

    pub fn wait_idle(&self) -> Result<(), Error> {
        let value = self.idle.fetch_add(1, Ordering::Relaxed) + 1;
        self.queue.signal_event(&self.idle_event.as_event(), value);
        match self.idle_event.wait_until_signaled_value(value, 10_000) {
            true => check_submission_error(&self.submission_error),
            false => Err(Error::new("timed out waiting for the device to go idle")),
        }
    }

    pub fn supports(&self, feature: Feature) -> bool {
        match feature {
            Feature::TileShaders => true,
            Feature::MeshShaders => true,
            Feature::IndirectCommands => true,

            Feature::MultiDraw => false,
            Feature::Timestamps => true,
            Feature::Occlusion => true,
            Feature::TextureCompressionBc => self.device.supports_bc_texture_compression(),
            Feature::TextureCompressionAstc => {
                self.device.supports_family(minmetal::GPUFamily::Apple2)
            }
        }
    }

    pub fn validation_errors(&self) -> u64 {
        0
    }

    pub fn max_resources(&self) -> u32 {
        u32::MAX
    }

    pub fn max_multi_draw(&self) -> u32 {
        1
    }

    #[doc(hidden)]
    pub fn depth_stencil_state(&self, state: &DepthState) -> Result<DepthStencilState, Error> {
        Ok(DepthStencilState(self.device.new_depth_stencil_state(
            &depth_stencil_descriptor(state),
        )?))
    }

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

    pub fn shaders(&self, entries: &[&'static crate::ir::GpuFn]) -> Result<Shaders, Error> {
        Shaders::new(self, entries)
    }

    pub(crate) fn make_resident(&self, allocation: &Allocation) -> Result<(), Error> {
        self.residency.add_allocation(allocation)?;
        self.residency.commit()?;
        Ok(())
    }

    pub fn surface(&self, view: isize, width: u32, height: u32) -> Result<Surface, Error> {
        let layer = unsafe {
            MetalLayer::attach_to_view(
                id(view as *mut std::ffi::c_void),
                &self.device,
                PixelFormat::Bgra8Unorm,
                width as usize,
                height as usize,
                1.0,
            )?
        };
        let Some(residency) = layer.residency_set()? else {
            return Err(Error::new("surface has no residency set"));
        };
        self.queue.add_residency_set(&residency);
        Ok(Surface {
            layer,
            residency,
            queue: self.queue.clone(),
            width,
            height,
        })
    }
}

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

pub struct HeapBuffer<'heap> {
    pub(crate) buffer: Buffer,
    pub(crate) gpu: u64,
    pub(crate) cpu: *mut u8,
    pub(crate) len: usize,
    _allocation: HeapAllocation<'heap>,
}

impl HeapBuffer<'_> {

    pub fn gpu<T: ?Sized>(&self) -> Gpu<T> {
        crate::device_ptr(self.gpu)
    }

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

    pub fn as_slice<T>(&self) -> Option<&[T]> {
        if self.cpu.is_null() {
            return None;
        }
        let count = self.len / size_of::<T>();
        Some(unsafe { std::slice::from_raw_parts(self.cpu as *const T, count) })
    }
}

pub struct HeapTexture<'heap> {
    storage_views: Vec<Texture>,
    pub(crate) texture: Texture,
    pub(crate) format: Format,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mips: u32,
    pub(crate) layers: u32,
    pub(crate) cube: bool,
    _allocation: HeapAllocation<'heap>,
}

impl HeapTexture<'_> {
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
    pub(crate) cube: bool,
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
            cube: false,
        }
    }

    pub fn texture_cube(fmt: Format, size: u32) -> Self {
        assert!(size > 0, "a cubemap must have a nonzero face size");
        Self { layers: 6, cube: true, ..Self::texture_2d(fmt, size, size) }
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
            cube: false,
        }
    }

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

    pub fn layers(mut self, layers: u32) -> Self {
        assert!(!self.cube, "cubemap arrays are not supported");
        assert!(layers > 0, "a texture must have at least one array layer");
        self.layers = layers;
        self
    }

    pub fn usage(mut self, extra: Usage) -> Self {
        self.usage = self.usage | extra;
        self
    }

    pub fn samples(mut self, samples: u32) -> Self {
        assert!(
            !self.cube || samples == 1,
            "cubemaps cannot be multisampled"
        );
        assert!(
            matches!(samples, 1 | 2 | 4 | 8),
            "{samples} is not a Metal sample count"
        );
        self.samples = samples;
        if samples > 1 {

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
            desc.set_array_length(if self.cube { 1 } else { self.layers as usize });
        }
        desc.set_usage(texture_usage);
        if self.mips > 1 {
            desc.set_mipmap_level_count(self.mips as usize);
        }
        if self.cube {
            desc.set_texture_type(TextureType::Cube);
        } else if self.samples > 1 && self.layers > 1 {
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

enum PendingResource {
    Buffer(Buffer),
    Texture(Texture, Vec<Texture>),
}

struct HeapStorage {
    tlsf: UnsafeCell<Tlsf>,
    heap: Heap,
}

struct HeapAllocation<'heap> {
    block: u32,
    storage: &'heap HeapStorage,
}

impl Drop for HeapAllocation<'_> {
    fn drop(&mut self) {
        unsafe { (*self.storage.tlsf.get()).free(self.block) };
    }
}

pub struct GpuHeap {
    pub(crate) device: minmetal::Device,
    storage: HeapStorage,
    pub(crate) options: ResourceOptions,
    retired: UnsafeCell<VecDeque<(u64, u32, PendingResource)>>,
}

impl GpuHeap {
    pub fn usage(&self) -> HeapUsage {
        let tlsf = unsafe { &*self.storage.tlsf.get() };
        let retired = unsafe { &*self.retired.get() };
        let mut usage = HeapUsage { capacity: tlsf.size, allocated: tlsf.used, peak: tlsf.peak, retired: 0,
            allocator_metadata: size_of::<Self>() + tlsf.blocks.capacity() * size_of::<crate::tlsf::Block>()
                + tlsf.spare.capacity() * size_of::<u32>() + retired.capacity() * size_of::<(u64, u32, PendingResource)>() };
        for (_, block, resource) in retired {
            usage.retired += tlsf.blocks[*block as usize].size;
            if let PendingResource::Texture(_, views) = resource { usage.allocator_metadata += views.capacity() * size_of::<Texture>(); }
        }
        usage
    }

    pub fn new(device: &Device, size: u64) -> Result<Self, Error> {
        Self::with_ring(device, size, 0)
    }

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
            storage: HeapStorage {
                heap,
                tlsf: UnsafeCell::new(Tlsf::new(size, 256)),
            },
            options,
            retired: UnsafeCell::new(VecDeque::new()),
        })
    }

    pub fn alloc_buffer(&self, len: usize) -> Result<HeapBuffer<'_>, Error> {
        let need = self.device.heap_buffer_size_and_align(len, self.options);
        let alloc = unsafe { &mut *self.storage.tlsf.get() }
            .alloc(need.size as u64, need.align as u64)
            .ok_or(Error::new("gpu heap is full"))?;
        let allocation = HeapAllocation {
            block: alloc.block,
            storage: &self.storage,
        };
        let buffer =
            self.storage
                .heap
                .new_buffer_at_offset(len, self.options, alloc.offset as usize)?;
        Ok(HeapBuffer {
            gpu: buffer.gpu_address()?,
            cpu: buffer.contents() as *mut u8,
            len,
            _allocation: allocation,
            buffer,
        })
    }

    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> Result<HeapBuffer<'_>, Error> {
        if size_of_val(values) == 0 {
            return Err(Error::new("a GPU buffer must contain at least one byte"));
        }
        let buffer = self.alloc_buffer(size_of_val(values))?;
        if let Err(error) = self.write(&buffer, 0, values) {
            self.free_buffer(buffer);
            return Err(error);
        }
        Ok(buffer)
    }

    pub fn alloc_texture(&self, desc: &TextureDesc) -> Result<HeapTexture<'_>, Error> {
        if desc.format.is_compressed() {
            if desc.samples != 1
                || desc.usage.0 & !(Usage::SAMPLED | Usage::COPY_SRC | Usage::COPY_DST).0 != 0
            {
                return Err(Error::new(
                    "compressed textures only support sampling and copying",
                ));
            }
            let supported = match desc.format {
                Format::Astc4x4Unorm | Format::Astc4x4UnormSrgb => {
                    self.device.supports_family(minmetal::GPUFamily::Apple2)
                }
                _ => self.device.supports_bc_texture_compression(),
            };
            if !supported {
                return Err(Error::new(
                    "texture compression format is not supported by this device",
                ));
            }
        }
        let descriptor = desc.descriptor();
        descriptor.set_storage_mode(StorageMode::Shared);
        let need = self.device.heap_texture_size_and_align(&descriptor);
        let alloc = unsafe { &mut *self.storage.tlsf.get() }
            .alloc(need.size as u64, need.align as u64)
            .ok_or(Error::new("gpu heap is full"))?;
        let allocation = HeapAllocation {
            block: alloc.block,
            storage: &self.storage,
        };
        let texture = self.storage.heap.new_texture_at_offset(&descriptor, alloc.offset as usize)?;
        let mut storage_views = Vec::new();
        if desc.cube && desc.usage.0 & Usage::STORAGE.0 != 0 {
            for mip in 0..desc.mips {
                storage_views.push(texture.new_texture_view_with_type_and_ranges(format(desc.format), TextureType::Cube,
                    minmetal::Range::new(mip as usize, 1), minmetal::Range::new(0, desc.layers as usize))?);
            }
        }
        Ok(HeapTexture {
            storage_views,
            texture,
            format: desc.format,
            width: desc.width,
            height: desc.height,
            mips: desc.mips,
            layers: desc.layers,
            cube: desc.cube,
            _allocation: allocation,
        })
    }

    pub fn write<T: Copy>(
        &self,
        buffer: &HeapBuffer,
        offset: usize,
        values: &[T],
    ) -> Result<(), Error> {
        let bytes = size_of_val(values);
        if offset > buffer.len || bytes > buffer.len - offset {
            return Err(Error::new("buffer write is out of bounds"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr() as *const u8,
                buffer.cpu.add(offset),
                bytes,
            )
        };
        Ok(())
    }

    pub fn upload(&self, texture: &HeapTexture, level: u32, bytes: &[u8]) -> Result<(), Error> {
        self.upload_layer(texture, level, 0, bytes)
    }

    pub fn upload_layer(
        &self,
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
        let expected = texture.format.image_size(width, height);
        if bytes.len() != expected {
            return Err(Error::new(format!(
                "{} bytes does not match the {expected} bytes required for a {width}x{height} {:?} image",
                bytes.len(),
                texture.format
            )));
        }
        let row = texture.format.row_size(width);
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

    pub fn flush(&self) -> Result<(), Error> {
        Ok(())
    }

    pub fn flush_async(&self) -> Result<u64, Error> {
        Ok(0)
    }

    pub fn free(&self, alloc: Alloc) {
        unsafe { (*self.storage.tlsf.get()).free(alloc.block) };
    }

    pub fn transition(&self, _texture: &HeapTexture, _usage: Usage) {}

    pub fn free_buffer(&self, buffer: HeapBuffer) {
        drop(buffer);
    }

    pub fn free_texture(&self, texture: HeapTexture) {
        drop(texture);
    }

    pub fn free_later(&self, frame: u64, retired: Retired<'_>) {
        let (block, resource) = match retired {
            Retired::Buffer(buffer) => {
                assert!(std::ptr::eq(buffer._allocation.storage, &self.storage));
                let HeapBuffer {
                    buffer,
                    _allocation,
                    ..
                } = buffer;
                let block = _allocation.block;
                std::mem::forget(_allocation);
                (block, PendingResource::Buffer(buffer))
            }
            Retired::Texture(texture) => {
                assert!(std::ptr::eq(texture._allocation.storage, &self.storage));
                let HeapTexture {
                    texture,
                    storage_views,
                    _allocation,
                    ..
                } = texture;
                let block = _allocation.block;
                std::mem::forget(_allocation);
                (block, PendingResource::Texture(texture, storage_views))
            }
        };
        unsafe { &mut *self.retired.get() }.push_back((frame, block, resource));
    }

    pub fn retire(&self, completed: u64) {
        let retired = unsafe { &mut *self.retired.get() };
        while let Some(&(frame, _, _)) = retired.front() {
            if frame >= completed && completed != u64::MAX {
                break;
            }
            let (_, block, resource) = retired.pop_front().unwrap();
            match resource {
                PendingResource::Buffer(buffer) => drop(buffer),
                PendingResource::Texture(texture, views) => { drop(views); drop(texture); },
            }
            unsafe { (*self.storage.tlsf.get()).free(block) };
        }
    }
}

impl Drop for GpuHeap {
    fn drop(&mut self) {
        self.retire(u64::MAX);
    }
}

pub enum Retired<'heap> {
    Buffer(HeapBuffer<'heap>),
    Texture(HeapTexture<'heap>),
}

#[derive(Clone, Copy, Debug)]
pub struct HeapRange {
    pub(crate) base: u32,
    pub(crate) len: u32,
}

pub struct ResourceHeap {
    pub(crate) cap: u32,
    pub(crate) head: u32,
    ids: Buffer,
    address: u64,
}

impl ResourceHeap {
    pub fn new(device: &Device, cap: u32) -> Result<Self, Error> {
        let ids = device.device.new_buffer(
            cap.max(1) as usize * size_of::<u64>(),
            ResourceOptions::new(
                CpuCacheMode::DefaultCache,
                StorageMode::Shared,
                HazardTrackingMode::Untracked,
            ),
        )?;
        unsafe { std::ptr::write_bytes(ids.contents().cast::<u64>(), 0, cap.max(1) as usize) };
        device.make_resident(&Allocation::from_buffer(&ids))?;
        Ok(Self {
            cap,
            head: 0,
            address: ids.gpu_address()?,
            ids,
        })
    }

    pub fn alloc_range(&mut self, len: u32) -> Result<HeapRange, Error> {
        if len > self.cap - self.head {
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
        assert!(slot < range.len && range.base < self.cap && slot < self.cap - range.base);
        let index = (range.base + slot) as usize;
        let resource_id = texture.texture.gpu_resource_id().expect("texture resource id");
        unsafe {
            self.ids
                .contents()
                .cast::<u64>()
                .add(index)
                .write(resource_id.impl_)
        };
        Texture2d(Handle((range.base + slot) as u64))
    }

    pub fn write_cube(&self, range: HeapRange, slot: u32, texture: &HeapTexture) -> TextureCube {
        assert!(texture.cube, "write_cube requires a cubemap");
        TextureCube(self.write_texture(range, slot, texture).0)
    }

    pub fn write_texture_in(
        &self,
        range: HeapRange,
        slot: u32,
        texture: &HeapTexture,
        _usage: Usage,
    ) -> Texture2d {
        self.write_texture(range, slot, texture)
    }

    pub fn write_storage_cube_mip(&self, range: HeapRange, slot: u32, texture: &HeapTexture, mip: u32) -> Result<TextureCubeRW, Error> {
        assert!(slot < range.len && range.base < self.cap && slot < self.cap - range.base);
        let view = texture.storage_views.get(mip as usize).ok_or_else(|| Error::new("storage cube mip is unavailable"))?;
        let resource_id = view.gpu_resource_id()?;
        unsafe { self.ids.contents().cast::<u64>().add((range.base + slot) as usize).write(resource_id.impl_); }
        Ok(TextureCubeRW(Handle((range.base + slot) as u64)))
    }

    pub fn write_storage(&self, range: HeapRange, slot: u32, texture: &HeapTexture) -> Texture2dRW {
        assert!(slot < range.len && range.base < self.cap && slot < self.cap - range.base);
        let index = (range.base + slot) as usize;
        let resource_id = texture.texture.gpu_resource_id().expect("texture resource id");
        unsafe {
            self.ids
                .contents()
                .cast::<u64>()
                .add(index)
                .write(resource_id.impl_)
        };
        Texture2dRW(Handle((range.base + slot) as u64))
    }

    pub fn write_sampler(&self, range: HeapRange, slot: u32, sampler: &SamplerState) -> Sampler {
        assert!(slot < range.len && range.base < self.cap && slot < self.cap - range.base);
        Sampler(Handle(
            sampler
                .0
                .gpu_resource_id()
                .expect("sampler has no resource id")
                .impl_,
        ))
    }
}

pub struct Surface {
    pub(crate) layer: MetalLayer,
    residency: ResidencySet,
    queue: M4CommandQueue,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Surface {

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

    pub fn present(&mut self, _device: &Device, _drawable: &Drawable) -> bool {
        false
    }

    pub fn destroy(&mut self, _device: &Device) {}
}

impl Drop for Surface {
    fn drop(&mut self) {
        self.queue.remove_residency_set(&self.residency);
    }
}

pub struct Drawable {
    pub(crate) drawable: minmetal::Drawable,
    pub(crate) texture: Texture,
}

pub struct Shaders {
    pub(crate) compiler: M4Compiler,
    pub(crate) library: minmetal::Library,

    pub(crate) render: Vec<minmetal::RenderPipelineState>,
    pub(crate) compute: Vec<minmetal::ComputePipelineState>,
}

impl Shaders {

    pub fn new(device: &Device, entries: &[&'static crate::ir::GpuFn]) -> Result<Self, Error> {
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
        specialized.as_function_descriptor()
    }

    pub fn raster(&mut self, vertex: VertexShader, fragment: FragmentShader) -> RasterBuilder<'_> {
        RasterBuilder {
            shaders: self,
            vertex: vertex.0.name,
            fragment: fragment.0.name,
            colors: [ColorTarget::new(Format::Rgba8Unorm); 4],
            color_count: 0,
            depth: None,
            topology: Topology::TriangleList,
            samples: 1,
            constants: None,
            indirect: false,
        }
    }

    pub fn raster_desc(&mut self, desc: &RasterDesc) -> Result<Pipeline, Error> {
        self.build_raster(desc)
    }

    pub(crate) fn build_raster(&mut self, desc: &RasterDesc) -> Result<Pipeline, Error> {
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

        color_targets(&pipeline.color_attachments(), desc.colors);

        let state = self
            .compiler
            .new_render_pipeline_state(&pipeline.as_pipeline_descriptor(), None)?;
        let raw = state.raw;
        let handle = pipeline_handle(&state, desc.indirect)?;
        self.render.push(state);
        Ok(Pipeline {
            raw,
            group: [0; 3],
            primitive: topology(desc.topology),
            handle,
        })
    }

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
            group: [0; 3],

            primitive: PrimitiveType::Triangle,
            handle,
        })
    }

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
            group: [0; 3],
            primitive: PrimitiveType::Triangle,
            handle: PipelineHandle(Handle(0)),
        })
    }

    pub fn compute(&mut self, entry: ComputeShader) -> Result<Pipeline, Error> {
        self.compute_with(entry, None)
    }

    pub fn compute_with(
        &mut self,
        entry: ComputeShader,
        constants: Option<&dyn Constants>,
    ) -> Result<Pipeline, Error> {
        let crate::ir::Stage::Compute(group) = entry.0.stage else {
            return Err(Error::new("expected a compute shader"));
        };
        if group.contains(&0) {
            return Err(Error::new("a compute group has size zero"));
        }
        let pipeline = M4ComputePipelineDescriptor::new();
        pipeline.set_label(entry.0.name);
        pipeline.set_compute_function_descriptor(Some(&self.function(entry.0.name, constants)));
        let state = self.compiler.new_compute_pipeline_state(&pipeline, None)?;
        let raw = state.raw;
        self.compute.push(state);
        Ok(Pipeline {
            raw,
            group,
            primitive: PrimitiveType::Triangle,
            handle: PipelineHandle(Handle(0)),
        })
    }
}

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
    sizes: [[u32; 2]; 9],
    resolve_sizes: [[u32; 2]; 9],
    area: Option<[u32; 2]>,
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
            sizes: [[u32::MAX; 2]; 9],
            resolve_sizes: [[u32::MAX; 2]; 9],
            area: None,
            desc,
            occlusion_base: 0,
            layers: 1,
            available_layers: u32::MAX,
        }
    }

    pub fn render_area(mut self, width: u32, height: u32) -> Self {
        assert!(width > 0 && height > 0, "a render area must be nonempty");
        self.area = Some([width, height]);
        self
    }

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

    pub fn tile(self, width: u32, height: u32, imageblock_bytes: usize) -> Self {
        self.desc.set_tile_width(width as usize);
        self.desc.set_tile_height(height as usize);
        self.desc.set_imageblock_sample_length(imageblock_bytes);
        self
    }

    pub fn occlusion(self, queries: &Occlusion) -> Self {
        self.desc
            .set_visibility_result_buffer(Some(&queries.buffer));

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

    pub fn color_texture(
        self,
        index: usize,
        texture: &HeapTexture,
        clear: Option<[f32; 4]>,
    ) -> Self {
        self.color_texture_at(index, texture, 0, 0, clear)
    }

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
        assert!(index < 8, "a pass supports at most 8 color attachments");
        self.sizes[index] = [
            ((texture.width() as u32) >> level).max(1),
            ((texture.height() as u32) >> level).max(1),
        ];
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

    pub fn resolve(self, index: usize, drawable: &Drawable) -> Self {
        self.attach_resolve(index, &drawable.texture, 0, 0, 1)
    }

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
        assert!(index < 8, "a pass supports at most 8 color attachments");
        self.resolve_sizes[index] = [
            ((texture.width() as u32) >> level).max(1),
            ((texture.height() as u32) >> level).max(1),
        ];
        self = self.use_attachment(available_layers);
        let attachment = self
            .desc
            .color_attachments()
            .object_at_indexed_subscript(index);
        attachment.base().set_resolve_texture(Some(texture));
        attachment.base().set_resolve_level(level as usize);
        attachment.base().set_resolve_slice(layer as usize);

        attachment
            .base()
            .set_store_action(minmetal::StoreAction::MultisampleResolve);
        self
    }

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
        self.resolve_sizes[8] = [
            (texture.width >> level).max(1),
            (texture.height >> level).max(1),
        ];
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
        self.sizes[8] = [
            (texture.width >> level).max(1),
            (texture.height >> level).max(1),
        ];
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

pub struct Raster<'a> {
    pub(crate) encoder: M4RenderCommandEncoder,
    pub(crate) table: &'a M4ArgumentTable,
    pub(crate) arena: &'a Arena,
    pub(crate) depth_states: &'a [MetalDepthStencilState; 16],
    pub(crate) primitive: Cell<PrimitiveType>,
    pub(crate) occlusion_base: u32,
    pub(crate) resources: Cell<u64>,
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
        pass.desc.set_render_target_width(width as usize);
        pass.desc.set_render_target_height(height as usize);
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
            resources: Cell::new(0),
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

    fn bind<T>(&self, root: Gpu<T>) -> Result<(), Error> {
        let resources = self.resources.get();
        bind_push(self.table, self.arena, root.0.get(), resources)
    }

    fn prepare<T>(&self, pipeline: Pipeline, root: T) -> Result<(), Error> {
        self.pipeline(pipeline);
        self.bind(alloc_root(self.arena, root)?)
    }

    pub fn pipeline(&self, pipeline: Pipeline) {
        self.primitive.set(pipeline.primitive);

        let state = unsafe { minmetal::RenderPipelineState::from_raw_borrowed(&pipeline.raw) };
        self.encoder.set_render_pipeline_state(state);
    }

    pub fn heap(&self, heap: &ResourceHeap) {
        self.resources.set(heap.address);
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

    pub fn depth_bias(&self, bias: f32, slope_scale: f32, clamp: f32) {
        self.encoder.set_depth_bias(bias, slope_scale, clamp);
    }

    pub fn depth_clip_mode(&self, clamp: bool) {
        self.encoder.set_depth_clip_mode(if clamp {
            minmetal::DepthClipMode::Clamp
        } else {
            minmetal::DepthClipMode::Clip
        });
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
        pipeline: Pipeline,
        root: T,
        indices: &HeapBuffer,
        count: u32,
    ) -> Result<(), Error> {
        self.prepare(pipeline, root)?;
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
        Ok(())
    }

    pub fn draw_indexed_indirect<I: IndexFormat, T>(
        &self,
        pipeline: Pipeline,
        root: T,
        indices: &HeapBuffer,
        _arena: &Arena,
        draws: Gpu<[DrawIndexed]>,
        count: u32,
    ) -> Result<(), Error> {
        self.prepare(pipeline, root)?;
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
        Ok(())
    }

    pub fn draw_indirect_count<T>(
        &self,
        pipeline: Pipeline,
        root: T,
        commands: &IndirectCommands,
        _draws: &HeapBuffer,
        range: &HeapBuffer,
    ) -> Result<(), Error> {
        debug_assert!(range.len >= size_of::<[u32; 2]>());
        self.prepare(pipeline, root)?;
        self.encoder
            .execute_commands_in_buffer_with_indirect_buffer(&commands.icb, range.gpu);
        Ok(())
    }

    pub fn draw_multi<T>(
        &self,
        pipeline: Pipeline,
        root: T,
        draws: &[MultiDraw],
    ) -> Result<(), Error> {
        self.prepare(pipeline, root)?;
        for draw in draws {
            self.encoder.draw_primitives(
                self.primitive.get(),
                draw.first_vertex as usize,
                draw.vertex_count as usize,
            );
        }
        Ok(())
    }

    pub fn draw_multi_indexed<T, I: IndexFormat>(
        &self,
        pipeline: Pipeline,
        root: T,
        indices: &HeapBuffer,
        draws: &[MultiDrawIndexed],
    ) -> Result<(), Error> {
        self.prepare(pipeline, root)?;
        let width = match I::WIDE {
            true => 4,
            false => 2,
        };
        for draw in draws {
            let offset = draw.first_index as usize * width;
            self.encoder.draw_indexed_primitives_with_base_vertex(
                self.primitive.get(),
                draw.index_count as usize,
                match I::WIDE {
                    true => IndexType::UInt32,
                    false => IndexType::UInt16,
                },
                indices.gpu + offset as u64,
                indices.len - offset,
                1,
                draw.vertex_offset as isize,
                0,
            );
        }
        Ok(())
    }

    pub fn draw_mesh<T>(
        &self,
        pipeline: Pipeline,
        root: T,
        groups: [u32; 3],
        object_group: [u32; 3],
        mesh_group: [u32; 3],
    ) -> Result<(), Error> {
        self.prepare(pipeline, root)?;
        self.encoder
            .draw_mesh_threadgroups(size(groups), size(object_group), size(mesh_group));
        Ok(())
    }

    pub fn dispatch_tile<T>(
        &self,
        pipeline: Pipeline,
        root: T,
        width: u32,
        height: u32,
    ) -> Result<(), Error> {
        self.prepare(pipeline, root)?;
        self.encoder
            .dispatch_threads_per_tile(Size::new(width as usize, height as usize, 1));
        Ok(())
    }

    pub fn draw_indexed_commands<T>(&self, pipeline: Pipeline, root: T, commands: &IndirectCommands, ranges: &[std::ops::Range<u32>]) -> Result<(), Error> {
        self.prepare(pipeline, root)?;
        for range in ranges {
            if range.start > range.end || range.end > commands.count {
                return Err(Error::new("ICB execution range exceeds command count"));
            }
            self.execute(commands, range.clone());
        }
        Ok(())
    }

    pub fn execute(&self, commands: &IndirectCommands, range: std::ops::Range<u32>) {
        self.encoder.execute_commands_in_buffer(
            &commands.icb,
            minmetal::Range::new(range.start as usize, range.len()),
        );
    }

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

    pub fn timestamp(&self, timers: &Timers, index: usize) {
        debug_assert!(index < timers.count);
        if !timers.enabled { return; }
        self.encoder.write_timestamp_with_granularity(
            M4TimestampGranularity::Precise,
            RenderStages::FRAGMENT,
            &timers.heaps[timers.base],
            index,
        );
    }

    pub fn draw<T>(
        &self,
        pipeline: Pipeline,
        root: T,
        vertices: std::ops::Range<u32>,
    ) -> Result<(), Error> {
        if vertices.start > vertices.end {
            return Err(Error::new("vertex range ends before it starts"));
        }
        let count = vertices.end - vertices.start;
        self.prepare(pipeline, root)?;
        self.encoder.draw_primitives(
            self.primitive.get(),
            vertices.start as usize,
            count as usize,
        );
        Ok(())
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
    pub(crate) resources: Cell<u64>,
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
            resources: Cell::new(0),
        }
    }

    fn bind<T>(&self, root: Gpu<T>) -> Result<(), Error> {
        let resources = self.resources.get();
        bind_push(self.table, self.arena, root.0.get(), resources)
    }

    fn prepare<T>(&self, pipeline: Pipeline, root: T) -> Result<(), Error> {
        self.pipeline(pipeline);
        self.bind(alloc_root(self.arena, root)?)
    }

    pub fn pipeline(&self, pipeline: Pipeline) {

        let state = unsafe { minmetal::ComputePipelineState::from_raw_borrowed(&pipeline.raw) };
        self.encoder.set_compute_pipeline_state(state);
    }

    pub fn heap(&self, heap: &ResourceHeap) {
        self.resources.set(heap.address);
    }

    pub fn dispatch<T>(
        &self,
        pipeline: Pipeline,
        root: T,
        launch: DispatchThreads,
    ) -> Result<(), Error> {
        let threads = launch.threads;
        let group = pipeline.group;
        if group.contains(&0) {
            return Err(Error::new("dispatch requires a compute pipeline"));
        }
        self.prepare(pipeline, root)?;
        self.encoder.dispatch_threads(
            Size::new(
                threads[0] as usize,
                threads[1] as usize,
                threads[2] as usize,
            ),
            Size::new(group[0] as usize, group[1] as usize, group[2] as usize),
        );
        Ok(())
    }

    pub fn barrier(&self, before: Stage, after: Stage, hazards: Hazard) {
        let (before, after, hazard) = (stage(before), stage(after), hazard(hazards));
        let (local_before, local_after) = (encoder_stages(before), encoder_stages(after));
        self.encoder.barrier(local_before, local_after, hazard);

        let compute = Stages::DISPATCH | Stages::BLIT | Stages::ACCELERATION_STRUCTURE;
        if before.0 & !compute.0 != 0 {
            self.encoder
                .barrier_after_queue_stages(before, local_after, hazard);
        }
        if after.0 & !compute.0 != 0 {
            self.encoder
                .barrier_before_queue_stages(before, after, hazard);
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

    pub fn transition(&self, _texture: &HeapTexture, _usage: Usage) {}

    pub fn read_texture(&self, texture: &HeapTexture, arena: &Arena, dest: GpuRw<[u8]>) {
        self.read_texture_at(texture, 0, 0, arena, dest)
    }

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
        let row = texture.format.row_size(width);
        let size = texture.format.image_size(width, height);
        assert!(
            offset <= arena.size && size <= arena.size - offset,
            "texture readback exceeds the arena"
        );
        assert!(
            !texture.format.is_compressed() || offset.is_multiple_of(texture.format.block_size()),
            "texture readback offset is not block-aligned"
        );
        self.encoder.copy_texture_to_buffer(
            &texture.texture,
            layer as usize,
            level as usize,
            Origin::new(0, 0, 0),
            Size::new(width as usize, height as usize, 1),
            &arena.buffer,
            offset,
            row,
            size,
            minmetal::BlitOption::NONE,
        );
    }

    pub fn execute(&self, commands: &IndirectCommands, range: std::ops::Range<u32>) {
        self.encoder.execute_commands_in_buffer(
            &commands.icb,
            minmetal::Range::new(range.start as usize, range.len()),
        );
    }

    pub fn reset(&self, commands: &IndirectCommands, range: std::ops::Range<u32>) {
        self.encoder.reset_commands_in_buffer(
            &commands.icb,
            minmetal::Range::new(range.start as usize, range.len()),
        );
    }

    pub fn timestamp(&self, timers: &Timers, index: usize) {
        debug_assert!(index < timers.count);
        if !timers.enabled { return; }
        self.encoder.write_timestamp_with_granularity(
            M4TimestampGranularity::Precise,
            &timers.heaps[timers.base],
            index,
        );
    }

    pub fn fill_buffer(&self, buffer: &HeapBuffer, offset: usize, size: usize, value: u32) {
        let bytes = value.to_ne_bytes();
        assert!(
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

fn alloc_root<T>(arena: &Arena, value: T) -> Result<Gpu<T>, Error> {
    arena
        .alloc(value)
        .map(|boxed| boxed.gpu)
        .ok_or_else(|| Error::new("frame arena is full"))
}

fn bind_push(
    table: &M4ArgumentTable,
    arena: &Arena,
    root: u64,
    resources: u64,
) -> Result<(), Error> {
    let Some((cpu, gpu)) = arena.alloc_bytes(size_of::<Push>(), 16) else {
        return Err(Error::new(
            "frame arena is full: no room for a root binding",
        ));
    };
    unsafe { (cpu as *mut Push).write(Push { root, resources }) };
    table.set_address(gpu, 0);
    Ok(())
}

fn check_submission_error(error: &OnceLock<String>) -> Result<(), Error> {
    match error.get() {
        Some(message) => Err(Error::new(message)),
        None => Ok(()),
    }
}

struct FrameFeedback {
    completed: SharedEvent,
    value: AtomicU64,
    gpu_ns: AtomicU64,
}

pub struct Context {
    commit_options: M4CommitOptions,
    feedback: Vec<(Arc<FrameFeedback>, M4CommitFeedbackHandler)>,
    submission_error: Arc<OnceLock<String>>,
    pub(crate) queue: M4CommandQueue,
    pub(crate) residency: ResidencySet,
    pub(crate) event: SharedEvent,
    pub(crate) command_buffers: Vec<M4CommandBuffer>,
    pub(crate) allocators: Vec<M4CommandAllocator>,
    pub(crate) tables: Vec<M4ArgumentTable>,
    pub(crate) arenas: Vec<Arena>,

    pub(crate) depth_states: [MetalDepthStencilState; 16],
    pub(crate) threads: usize,
    pub(crate) frame: u64,
    recorders: Box<[MaybeUninit<Recorder<'static>>]>,
}

impl Context {

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

        let commit_options = M4CommitOptions::new();
        let mut feedback = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            let result = Arc::new(FrameFeedback {
                completed: device.device.new_shared_event()?,
                value: AtomicU64::new(0),
                gpu_ns: AtomicU64::new(f64::NAN.to_bits()),
            });
            let state = Arc::clone(&result);
            let submission_error = Arc::clone(&device.submission_error);
            let handler = M4CommitFeedbackHandler::new(move |feedback| {
                let value = state.value.load(Ordering::Acquire);
                if let Some(error) = feedback.error_description() {
                    let _ = submission_error.set(error.to_string());
                }
                let start = feedback.gpu_start_time();
                let end = feedback.gpu_end_time();
                let ns = if start.is_finite() && end.is_finite() && start > 0.0 && end > start { (end - start) * 1e9 } else { f64::NAN };
                state.gpu_ns.store(ns.to_bits(), Ordering::Release);
                state.completed.set_signaled_value(value);
            });
            feedback.push((result, handler));
        }

        Ok(Self {
            queue: device.queue.clone(),
            residency: device.residency.clone(),
            commit_options,
            feedback,
            submission_error: device.submission_error.clone(),
            event: device.device.new_shared_event()?,
            command_buffers,
            allocators,
            tables,
            arenas,
            depth_states,
            threads,
            frame: 0,
            recorders: Box::new_uninit_slice(threads),
        })
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn completed(&self) -> u64 {
        self.event.signaled_value()
    }

    pub fn begin_frame(&mut self) -> Result<Frame<'_>, Error> {
        check_submission_error(&self.submission_error)?;
        let index = self.frame;
        if index >= FRAMES_IN_FLIGHT
            && !self
                .event
                .wait_until_signaled_value(index + 1 - FRAMES_IN_FLIGHT, 10_000)
        {
            return Err(Error::new("timed out waiting for a frame in flight"));
        }

        let (feedback, feedback_handler) = &self.feedback[(index % FRAMES_IN_FLIGHT) as usize];
        if index >= FRAMES_IN_FLIGHT
            && !feedback.completed.wait_until_signaled_value(index + 1 - FRAMES_IN_FLIGHT, 10_000)
        {
            return Err(Error::new("timed out waiting for GPU submission feedback"));
        }
        check_submission_error(&self.submission_error)?;
        let base = (index % FRAMES_IN_FLIGHT) as usize * self.threads;
        let slots = base..base + self.threads;
        for slot in slots.clone() {
            self.allocators[slot].reset();
            self.arenas[slot].reset();
        }

        Ok(Frame {
            commit_options: &self.commit_options,
            feedback,
            feedback_handler,
            submission_error: &self.submission_error,
            queue: &self.queue,
            residency: &self.residency,
            event: &self.event,
            arenas: &self.arenas[slots.clone()],
            allocators: &self.allocators[slots.clone()],
            command_buffers: &self.command_buffers[slots.clone()],
            tables: &self.tables[slots],
            depth_states: &self.depth_states,
            recording: 0,
            recorders: unsafe {
                std::slice::from_raw_parts_mut(self.recorders.as_mut_ptr().cast(), self.threads)
            },
            frame: &mut self.frame,
            value: index + 1,
        })
    }
}

pub struct Frame<'a> {
    commit_options: &'a M4CommitOptions,
    feedback: &'a FrameFeedback,
    feedback_handler: &'a M4CommitFeedbackHandler,
    submission_error: &'a OnceLock<String>,
    pub(crate) queue: &'a M4CommandQueue,
    pub(crate) residency: &'a ResidencySet,
    pub(crate) event: &'a SharedEvent,
    pub(crate) arenas: &'a [Arena],
    pub(crate) allocators: &'a [M4CommandAllocator],
    pub(crate) command_buffers: &'a [M4CommandBuffer],
    pub(crate) tables: &'a [M4ArgumentTable],
    pub(crate) depth_states: &'a [MetalDepthStencilState; 16],
    pub(crate) recording: usize,
    recorders: &'a mut [MaybeUninit<Recorder<'a>>],
    pub(crate) value: u64,
    frame: &'a mut u64,
}

impl<'a> Frame<'a> {
    pub fn completed_gpu_ms(&self) -> Option<f64> {
        if self.value <= FRAMES_IN_FLIGHT { return None; }
        let ns = f64::from_bits(self.feedback.gpu_ns.load(Ordering::Acquire));
        if ns.is_finite() && ns > 0.0 { Some(ns / 1e6) } else { None }
    }

    pub fn arena(&self, thread: usize) -> &'a Arena {
        &self.arenas[thread]
    }
}

impl<'a> Frame<'a> {

    pub fn recorders(&mut self, counts: &[usize]) -> &mut [Recorder<'a>] {
        assert!(
            self.recording == 0,
            "recorders already taken for this frame"
        );
        let count: usize = counts.iter().sum();
        assert!(count <= self.command_buffers.len());
        for _ in 0..count {
            self.recorder();
        }
        unsafe { std::slice::from_raw_parts_mut(self.recorders.as_mut_ptr().cast(), count) }
    }

    pub fn recorder(&mut self) -> &mut Recorder<'a> {
        let index = self.recording;
        assert!(index < self.command_buffers.len());
        self.command_buffers[index].begin_command_buffer_with_allocator(&self.allocators[index]);
        self.command_buffers[index].use_residency_set(self.residency);
        let recorder = self.recorders[index].write(Recorder {
            arena: &self.arenas[index],
            command_buffer: &self.command_buffers[index],
            table: &self.tables[index],
            depth_states: self.depth_states,
            not_sync: PhantomData,
            compute: None,
            raster: None,
        });
        self.recording += 1;
        recorder
    }

    pub fn after_uploads(&mut self, _value: u64, _heap: &GpuHeap) {}

    pub fn submit(mut self, drawable: Option<&Drawable>) -> Result<(), Error> {
        check_submission_error(self.submission_error)?;
        let recording = self.recording;
        while self.recording > 0 {
            self.recording -= 1;
            unsafe { self.recorders[self.recording].assume_init_drop() };
        }
        if let Some(drawable) = drawable {
            self.queue.wait_for_drawable(&drawable.drawable);
        }
        if recording > 0 {
            self.feedback.value.store(self.value, Ordering::Release);
            self.commit_options.add_feedback_handler(self.feedback_handler);
        } else {
            self.feedback.gpu_ns.store(f64::NAN.to_bits(), Ordering::Release);
            self.feedback.completed.set_signaled_value(self.value);
        }
        self.queue
            .commit_with_options(&self.command_buffers[..recording], self.commit_options);
        self.queue.signal_event(&self.event.as_event(), self.value);
        if let Some(drawable) = drawable {
            self.queue.signal_drawable(&drawable.drawable);
            drawable.drawable.present();
        }
        *self.frame = self.value;
        Ok(())
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        while self.recording > 0 {
            self.recording -= 1;
            unsafe { self.recorders[self.recording].assume_init_drop() };
        }
    }
}

pub struct Recorder<'a> {
    pub(crate) arena: &'a Arena,
    pub(crate) command_buffer: &'a M4CommandBuffer,
    pub(crate) table: &'a M4ArgumentTable,
    pub(crate) depth_states: &'a [MetalDepthStencilState; 16],
    pub(crate) not_sync: PhantomData<Cell<()>>,
    compute: Option<Compute<'a>>,
    raster: Option<Raster<'a>>,
}

impl<'a> Recorder<'a> {

    pub fn arena(&self) -> &'a Arena {
        self.arena
    }
}

impl<'a> Recorder<'a> {
    pub fn timestamp(&mut self, timers: &Timers, index: usize) {
        debug_assert!(index < timers.count);
        if !timers.enabled { return; }
        self.compute = None;
        self.raster = None;
        self.command_buffer.write_timestamp_into_heap(&timers.heaps[timers.base], index);
    }

    pub fn compute(&mut self) -> &Compute<'a> {
        self.raster = None;
        if self.compute.is_none() {
            self.compute = Some(Compute::new(self.command_buffer, self.table, self.arena));
        }
        let compute = self.compute.as_ref().unwrap();
        compute.resources.set(0);
        compute
    }

    pub fn raster(&mut self, pass: &Pass) -> &Raster<'a> {
        self.compute = None;
        self.raster = None;
        let mut extent = [u32::MAX; 2];
        for size in pass.sizes.iter().chain(&pass.resolve_sizes) {
            extent[0] = extent[0].min(size[0]);
            extent[1] = extent[1].min(size[1]);
        }
        let [width, height] = pass.area.unwrap_or(extent);
        assert!(
            width > 0 && height > 0 && width < u32::MAX && height < u32::MAX,
            "a pass needs an attachment or an explicit render area"
        );
        assert!(
            width <= extent[0] && height <= extent[1],
            "the render area exceeds an attachment"
        );
        self.raster.insert(Raster::new(
            self.command_buffer,
            self.table,
            self.arena,
            self.depth_states,
            pass,
            width,
            height,
        ))
    }
}

impl Drop for Recorder<'_> {
    fn drop(&mut self) {
        self.compute = None;
        self.raster = None;
        self.command_buffer.end_command_buffer();
    }
}

pub struct Fence(SharedEvent);

impl Fence {
    pub fn new(device: &Device) -> Result<Self, Error> {
        Ok(Self(device.device.new_shared_event()?))
    }

    pub fn signal(&self, value: u64) {
        self.0.set_signaled_value(value);
    }

    pub fn value(&self) -> u64 {
        self.0.signaled_value()
    }

    pub fn wait(&self, value: u64) -> bool {
        self.0.wait_until_signaled_value(value, 10_000)
    }
}

pub struct IndirectCommands {
    pub(crate) icb: IndirectCommandBuffer,
    pub(crate) handle: Commands,
    pub(crate) count: u32,
    residency: ResidencySet,
}

impl Drop for IndirectCommands {
    fn drop(&mut self) {
        self.residency.remove_allocation(&Allocation::from_indirect_command_buffer(&self.icb)).ok();
        self.residency.commit().ok();
    }
}

impl IndirectCommands {
    pub fn indexed<I: IndexFormat>(device: &Device, topology: Topology, indices: &HeapBuffer, draws: &[DrawIndexed]) -> Result<Self, Error> {
        if draws.is_empty() { return Err(Error::new("an ICB must contain at least one command")); }
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(IndirectCommandType::DRAW_INDEXED);
        desc.set_inherit_pipeline_state(true);
        desc.set_inherit_buffers(true);
        Self::inherit_render_state(&desc);
        let commands = Self::new(device, &desc, draws.len() as u32)?;
        let width = if I::WIDE { 4 } else { 2 };
        for (slot, draw) in draws.iter().enumerate() {
            if (draw.index_start as u64 + draw.index_count as u64) * width as u64 > indices.len as u64 {
                return Err(Error::new("ICB index range exceeds the index buffer"));
            }
            commands.icb.render_command(slot)?.draw_indexed_primitives(
                self::topology(topology), draw.index_count as usize,
                if I::WIDE { IndexType::UInt32 } else { IndexType::UInt16 },
                &indices.buffer, draw.index_start as usize * width, draw.instance_count as usize,
                draw.base_vertex as isize, draw.base_instance as usize)?;
        }
        Ok(commands)
    }

    pub fn allocated_size(&self) -> usize {
        Allocation::from_indirect_command_buffer(&self.icb).allocated_size()
    }

    pub fn counted_draws(device: &Device, count: u32) -> Result<Self, Error> {
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(IndirectCommandType::DRAW);
        desc.set_inherit_pipeline_state(true);
        desc.set_inherit_buffers(true);
        Self::inherit_render_state(&desc);
        Self::new(device, &desc, count)
    }

    pub fn draws(device: &Device, count: u32, indexed: bool) -> Result<Self, Error> {
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(match indexed {
            true => IndirectCommandType::DRAW_INDEXED,
            false => IndirectCommandType::DRAW,
        });

        desc.set_inherit_pipeline_state(false);
        desc.set_inherit_buffers(false);
        desc.set_max_vertex_buffer_bind_count(2);
        desc.set_max_fragment_buffer_bind_count(2);
        Self::inherit_render_state(&desc);
        Self::new(device, &desc, count)
    }

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

    pub fn dispatches(device: &Device, count: u32) -> Result<Self, Error> {
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
        desc.set_inherit_pipeline_state(false);
        desc.set_inherit_buffers(false);
        desc.set_max_kernel_buffer_bind_count(2);
        Self::new(device, &desc, count)
    }

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
            residency: device.residency.clone(),
        })
    }

    pub fn handle(&self) -> Commands {
        self.handle
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn all(&self) -> std::ops::Range<u32> {
        0..self.count
    }
}

pub struct Timers {
    pub enabled: bool,
    pub(crate) heaps: Vec<M4CounterHeap>,
    pub(crate) count: usize,
    pub(crate) base: usize,
    pub(crate) ticks: Vec<u64>,
    pub(crate) ns_per_tick: f64,
}

impl Timers {
    pub fn new(device: &Device, count: usize) -> Result<Self, Error> {
        let desc = M4CounterHeapDescriptor::new();
        desc.set_type(M4CounterHeapType::Timestamp);
        desc.set_count(count);
        let mut heaps = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            let heap = device.device.new_m4_counter_heap_with_descriptor(&desc)?;
            heap.invalidate_counter_range(minmetal::Range::new(0, count));
            heaps.push(heap);
        }
        Ok(Self {
            enabled: true,
            heaps,
            count,
            base: 0,
            ticks: vec![0; count],
            ns_per_tick: minmetal::timestamp_nanoseconds_per_tick()?,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn begin(&mut self, frame: &Frame) {
        self.base = ((frame.value - 1) % FRAMES_IN_FLIGHT) as usize;
        self.ticks.fill(0);
        let range = minmetal::Range::new(0, self.count);
        let heap = &self.heaps[self.base];
        if let Some(data) = heap.resolve_counter_range(range) {
            for (tick, entry) in self.ticks.iter_mut().zip(data.chunks_exact(8)) {
                *tick = u64::from_ne_bytes(entry.try_into().unwrap());
            }
        }
        heap.invalidate_counter_range(range);
    }

    pub fn written(&self, index: usize) -> bool {
        self.enabled && self.ticks[index] != 0
    }

    pub fn ns(&self, start: usize, end: usize) -> f64 {
        self.checked_ns(start, end).unwrap_or(0.0)
    }

    pub fn ms(&self, start: usize, end: usize) -> f64 {
        self.ns(start, end) / 1e6
    }
}

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

    pub fn samples(&self, index: u32) -> u64 {
        self.samples[index as usize]
    }
}

#[test]
#[ignore = "requires a GPU"]
fn deferred_cube_storage_views() -> Result<(), Error> {
    let device = Device::new(true)?;
    let heap = GpuHeap::new(&device, 1048576)?;
    let texture = heap.alloc_texture(&TextureDesc::texture_cube(Format::Rgba32Float, 4).mips(3).usage(Usage::STORAGE))?;
    let before = heap.usage();
    assert!(before.allocated > 0 && before.allocated <= before.capacity);
    assert_eq!(before.peak, before.allocated);
    heap.free_later(7, Retired::Texture(texture));
    heap.retire(7);
    assert_eq!(heap.usage().allocated, before.allocated);
    assert_eq!(heap.usage().retired, before.allocated);
    {
        let pending = unsafe { &*heap.retired.get() };
        assert_eq!(pending.len(), 1);
        match &pending.front().unwrap().2 {
            PendingResource::Texture(_, views) => assert_eq!(views.len(), 3),
            _ => panic!("expected a deferred texture"),
        }
    }
    heap.retire(8);
    assert!(unsafe { &*heap.retired.get() }.is_empty());
    assert_eq!(heap.usage().allocated, 0);
    assert_eq!(heap.usage().retired, 0);
    assert_eq!(heap.usage().peak, before.allocated);
    Ok(())
}

#[test]
fn checked_timing_intervals() {
    let mut timers = Timers {
        enabled: true,
        heaps: Vec::new(),
        count: 4,
        base: 0,
        ticks: vec![0, 10, 10, 30],
        ns_per_tick: 2.5,
    };
    assert_eq!(timers.checked_ns(0, 1), None);
    assert_eq!(timers.checked_ns(1, 0), None);
    assert_eq!(timers.checked_ns(3, 1), None);
    assert_eq!(timers.checked_ns(1, 2), Some(0.0));
    assert_eq!(timers.checked_ns(1, 3), Some(50.0));
    assert_eq!(timers.checked_ms(1, 3), Some(0.00005));
    assert_eq!(timers.checked_ns(4, 1), None);
    assert_eq!(timers.checked_ns(1, usize::MAX), None);
    assert_eq!(timers.ms(3, 1), 0.0);
    timers.enabled = false;
    assert_eq!(timers.checked_ns(1, 3), None);
    timers.enabled = true;
    assert_eq!(timers.checked_ns(1, 3), Some(50.0));
}
