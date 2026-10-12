# nga-vk: where Vulkan follows Metal, and where it doesn't

Port of `nga` (Metal 4 + MSL) to Vulkan 1.4. `#[gpu_mod]` compiles Rust shaders to SPIR-V and MSL; no offline compiler. Hardware: NVIDIA GTX 1070 Ti (Pascal), driver 582.66, Vulkan 1.4.312, SDK 1.4.357. Zero validation errors.

Sebastian Aaltonen's Metal-vs-Vulkan post is the comparison point.

| Example                  | Ports?       | Proves                                                              |
| ------------------------ | ------------ | ------------------------------------------------------------------- |
| `triangle`               | yes          | BDA root pointer, dynamic rendering, timeline semaphore                |
| `triangle_shader_object` | yes          | same frame, no `VkPipeline`                                         |
| `raster`                 | yes          | `GpuHeap`/TLSF, depth + blend as dynamic state, specialization constants |
| `sync`                   | yes          | compute → compute → raster barriers                                   |
| `bindless`               | yes          | descriptor buffers, GPU-painted storage image                        |
| `threaded`               | yes          | 4 threads, 4 command pools, one target                                |
| `gpu_driven`             | reformulated | GPU-decided draw count via `vkCmdDrawIndirectCount`                 |
| `meshlets`               | no           | no mesh-shader extension on Pascal                                    |
| `tbdr`                   | no           | no tile shading in Vulkan                                            |

## Vulkan is better

- **Pointers.** Aaltonen says BDA stays niche until HLSL/GLSL have native pointers. Rust `Gpu<T>` emits `PhysicalStorageBuffer` and reads like MSL `device const Data*`. Pointer atomics come with it (`GpuRw<AtomicU32>` → `OpAtomicIAdd`). One host struct, not two.
- **Root pointer.** Metal needs an `MTL4ArgumentTable`: sized at creation, allocated per frame/thread, bound per encoder. Vulkan is a 16-byte push constant and one `vkCmdPushConstants`.
- **Residency.** Metal's `MTLResidencySet` is what makes raw pointers work. Vulkan memory is resident by construction, so that whole subsystem is gone.
- **`maintenance4`.** The post's "create the object, then ask which heaps work" is fixed. `vkGetDeviceBufferMemoryRequirements` / `vkGetDeviceImageMemoryRequirements` answer from create-info. `GpuHeap` never makes a throwaway object.
- **Bindless.** The post prefers 32-bit heap indices over Metal's 64-bit `ResourceID`. Metal backs `ResourceHeap` with a contiguous GPU-visible table of texture resource IDs. Texture handles select slots, and shaders load the resource ID from the table. The shared handle wrappers are still 64-bit; this does not yet provide compact 32-bit handles.

## Vulkan is worse

### 1. Image layouts

`nga` barriers are `(after_stages, before_stages, visibility)`. Buffers match. `VkMemoryBarrier2` is stage/access with no resource list. Images still need explicit transitions: `Pass` must know clear (`UNDEFINED`) vs load (`COLOR_ATTACHMENT_OPTIMAL`). Metal load actions have no layout concept. `VK_KHR_unified_image_layouts` (2025) would delete this; this driver doesn't have it. Irreducible gap.

### 2. `VK_EXT_host_image_copy` is advertised but dead

Expected Metal `replace_region`: CPU memory into an image, no staging. Images with `HOST_TRANSFER_EXT` report `memoryTypeBits == 0`, even though `R8G8B8A8_UNORM` advertises `HOST_IMAGE_TRANSFER`. `examples/probe.rs` confirms create-info and object queries agree. Texture upload stages through a buffer.

### 3. No `StorageMode::Shared` on discrete Windows

Metal `Arena` writes structs into GPU memory. Here that only works in a 214 MiB host-visible device-local BAR window (no ReBAR). `Arena` fits; `GpuHeap` is device-local, so `HeapBuffer::write` is a staged copy. `bufferImageGranularity` is 1024: heap allocations pad so buffers and images don't share a page. `MTLHeap` never asks.

`cargo run --example probe`:

```
heap 0:  8060 MiB  DEVICE_LOCAL
heap 1:  8146 MiB  system RAM
heap 2:   214 MiB  DEVICE_LOCAL BAR window  (~205 MiB free; ~9 MiB already used by the desktop)

Bar      host-visible VRAM   type 10 heap 2
Upload   write-combined RAM  type  8 heap 1
Readback cached RAM          type  9 heap 1
```

- **Placement.** `Arena::new` takes `Placement` because "host-visible" is three memories. `Bar` is reserved for per-frame root data. `Upload` is write-combined RAM (staging, costs no BAR). `Readback` is cached RAM. Staging from `Bar` spent the scarcest memory on a buffer no shader reads.
- **BAR budget.** `Context` allocates `FRAMES_IN_FLIGHT * threads` arenas. `Context::with_layout` fails against `VK_EXT_memory_budget` instead of `ERROR_OUT_OF_DEVICE_MEMORY` mid-loop. A full heap and a missing heap are different failures.
- **Uploads batch.** Used to submit + `vkQueueWaitIdle` per resource (8 MiB staging, panic past it). `Uploader` records into a wrapping ring and submits once on `flush`.

|                                           | before                  | now                          |
| ----------------------------------------- | ----------------------- | ---------------------------- |
| cost of one `write`/`upload`/`transition`   | submit + device drain   | memcpy + a few command words |
| 32-texture load                           | 64+ drains               | 1 submission                 |
| write larger than staging                 | panic at 8 MiB          | chunked through the ring    |
| staging                                   | 8 MiB of BAR            | 32 MiB system RAM            |
| write of 64 KiB or less                    | staged copy             | `vkCmdUpdateBuffer`          |

`GpuHeap::flush` blocks once for the batch. `flush_async` returns a timeline value; `Frame::after_uploads` makes the GPU wait, so streaming never blocks the CPU. `tests/upload.rs` covers the `vkCmdUpdateBuffer` cutoff, a write 3× the ring, an image taller than the ring, and one submission per batch.

**Copy engine, not done.** Dedicated transfer family exists (2 queues, transfer-only). Uploader stays on graphics because moving it needs queue-family ownership transfers and split transitions. Worth it only once assets stream during rendering.

### 4. Visibility is stage-constrained

`MTL4VisibilityOptions::DEVICE` is the same for any barrier. Vulkan `SHADER_WRITE` is illegal with `ALL_TRANSFER`, `SHADER_READ` illegal with `DRAW_INDIRECT`. `Visibility::Device` maps to `MEMORY_READ`/`MEMORY_WRITE` so the portable `barrier(after, before, hazard)` stays legal. Correct, more conservative than Metal.

### 5. Struct layout is Rust's

MSL has `static_assert(__builtin_offsetof(...))`. SPIR-V has no offsetof, and std430 makes a 3-vector 16 bytes where `packed_float3` / `[f32; 3]` are 12. `scalarBlockLayout` is required. Even then a matrix wants align 4 vs Rust `f32x4x4` `align(16)`:

|                              | `vertices` | `mvp` | size |
| ---------------------------- | ---------- | ----- | ---- |
| Rust `#[repr(C, align(16))]` | 0          | 16    | 80   |
| inferred scalar layout       | 0          | 8     | 72   |

`#[gpu_mod]` records every field offset from Rust. `spirv.rs` emits `OpMemberDecorate Offset`; `msl.rs` emits explicit `uint` padding. `tests/gpu.rs` parses the decorations back out and checks them against `TypeAbi`.

### 6. One bindless heap becomes three

Metal backs `ResourceHeap` with a GPU-visible table of texture resource IDs, indexed by slot inside MSL intrinsics. Sampled and storage textures use the same table; samplers carry their resource IDs directly rather than loading from the table. Vulkan needs `SAMPLER_DESCRIPTOR_BUFFER_BIT` vs `RESOURCE_DESCRIPTOR_BUFFER_BIT`. One heap becomes sampled images, storage images, and samplers across two buffers. Unified index space so `range.base + slot` still works, at the cost of reserving a slot in all three.

Descriptor size is driver-defined (4 bytes here, not Metal's 8). A texture written and sampled in the same frame stays `GENERAL`, so the sampled descriptor must record that layout. `write_texture_in` exists only because of layouts.

### 7. No GPU-side command encoding in core Vulkan

Metal constructs `render_command` inside a kernel with a per-draw root pointer. Core Vulkan has no GPU encoder.

Portable substitute: GPU-written `VkDrawIndirectCommand` array + count, consumed by `vkCmdDrawIndirectCount`. Every draw shares one pipeline and one push constant, so per-draw data is smuggled through `first_instance` / `gl_BaseInstance`. Reset is `vkCmdFillBuffer`.

#### `VK_NV_device_generated_commands`

Exposed here (rev 3 + compute rev 2). Cross-vendor `VK_EXT_device_generated_commands` is not. Pascal is on a maintenance driver.

| Metal ICB                                | NV token                            | Fit                                      |
| ---------------------------------------- | ----------------------------------- | ---------------------------------------- |
| `set_render_pipeline_state(handle)`       | `SHADER_GROUP_NV`                   | index, not a handle                      |
| `set_vertex_buffer(draw, 1)` (per-draw root) | `PUSH_CONSTANT_NV`              | exact, cleaner                            |
| `draw_primitives(...)`                  | `DRAW_NV`                           | exact except `type`                     |
| reset slot replays as nothing             | `sequencesCountBuffer`             | better: GPU-written count                |
| `IndirectCommands::dispatches`          | `DISPATCH_NV` + `PIPELINE_NV`       | exact                                   |
| `IndirectCommands::meshes`               | `DRAW_TASKS_NV`                     | needs NV mesh shaders, dead on Pascal    |

Push-constant token is the win: each generated sequence can write its own root pointer, which deletes the `first_instance` trick. The kernel writes a plain buffer of structs (`{ groupIndex }`, `{ u64 root }`, `{ vertexCount, ... }`). No encoder object.

Gap: per-command render state. Shader groups share input assembly, raster, and blend. `STATE_FLAGS_NV` is front-face only. `gpu_meshes` switches `Topology::LineList` vs triangles per object, which one stream cannot do. Options: one `vkCmdExecuteGeneratedCommandsNV` per topology, or expand wireframe to triangles. Pipeline set also freezes at creation (`SHADER_GROUP_NV` indexes groups declared with `VK_PIPELINE_CREATE_INDIRECT_BINDABLE_BIT_NV`). Metal's `PipelineHandle` is a free 64-bit id.

`Feature::IndirectCommands` could answer true with the split-by-topology form. Extra machinery: layout object, preprocess buffer, `vkCmdFillBuffer` for the count.

### 8. Presentation is not `wait_for_drawable`

Metal is three queue calls. Vulkan needs a binary semaphore pair per swapchain image through `vkQueueSubmit2` / `vkQueuePresentKHR`, plus out-of-date handling and rebuild. `Frame::submit` takes semaphores, not a drawable.

### 9. No programmable blending

`fragment_screen` reads `float4 dst [[color(0)]]` (framebuffer fetch). Desktop Vulkan has none. Screen blend happens to be `src + dst*(1-src)` = `ONE` / `ONE_MINUS_SRC_COLOR`. A formula that needs the previous value cannot port.

### 10. Extra feature bits

`shaderInt64` (u64 push constant and pointer math), `shaderDrawParameters` (`#[base_instance]` → `BaseInstance`), `drawIndirectCount`, `scalarBlockLayout`. Metal has 64-bit addressing on. Validation caught all of them.

## Equal

- Function constants → specialization constants. `[[vk::constant_id(N)]]` is the field index in `#[derive(Constants)]`. SPIR-V bools are `VkBool32` (4 bytes).
- Timeline semaphores: `MTLSharedEvent` → `VkSemaphore` timeline, 1:1.
- Command allocators: `MTL4CommandAllocator` → `VkCommandPool` + `vkResetCommandPool`, 1:1.
- Load/store actions: `MTLRenderPassDescriptor` → `VkRenderingAttachmentInfo` (dynamic rendering).
- Depth/stencil and blend as state objects via `VK_EXT_extended_dynamic_state` / `extendedDynamicState3`.
- TLSF copied unchanged.
- Multi-threaded recording: one command pool per thread per frame; lock-free `Arena` bump ports.

## `VK_EXT_shader_object`

`triangle_shader_object` is `triangle` with no `VkPipeline`. Identical frame, zero validation errors.

- **Deletes.** Pipeline creation is one call naming two entry points. Gone: `VkGraphicsPipelineCreateInfo` and its ten sub-structs, colour-target formats, dynamic-state array, blend/depth/multisample. Shader objects reuse across different attachment formats. A `VkPipeline` never does. That is Metal 4's model.
- **Costs.** State moves to the command buffer: `vkCmdBindShadersEXT` plus 19 mandatory setters (omitting any is UB). Needs four extra `extendedDynamicState3` features.
- **Verdict.** Better at creation (no PSO permutations, no format coupling). Worse at record time: 19 setters vs 10 on the dynamic-state pipeline. Worth it when permutations are the problem. Not here, so the other examples keep pipelines.

## Not portable on this hardware

- **`meshlets`.** No `VK_EXT_mesh_shader` on Pascal (Turing+). Mapping is otherwise close: object shader → task, `set_threadgroups_per_grid` → `DispatchMesh`, `set_primitive_count` → `SetMeshOutputsEXT`.
- **`gpu_meshes`.** No `VK_EXT_device_generated_commands`. NV form cannot vary topology per command, so the single-stream example has no Vulkan spelling. See §7.
- **`tbdr`.** No tile shading or imageblocks on desktop Vulkan. Nearest: `VK_KHR_dynamic_rendering_local_read` (supported) or an extra full-screen pass.
