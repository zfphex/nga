# Port `nga` (Metal 4) to `nga-vk` (Vulkan 1.3 + Slang)

## Context

`nga` is a "No Graphics API" renderer built on Metal 4 via `minmetal`: raw GPU device addresses instead of
bound buffers, one argument table holding two `u64` roots, a `MTLResourceID` array as a hand-owned bindless
heap, untracked hazard mode with explicit barriers, TLSF over a single `MTLHeap`, and timeline `MTLSharedEvent`s.
`Plan.md` derives all of it from Aaltonen's *No Graphics API* post.

The goal is **not** a portability layer. It is to build the same design a second time on Vulkan and find out
where Vulkan cannot follow Metal — so `nga-vk` stays a standalone crate and the two are deliberately not merged.
Shaders move from MSL to Slang so the shader language is not itself a variable.

Environment probed: Vulkan SDK 1.4.357 with `slangc.exe`; GPU is a **GTX 1070 Ti (Pascal)**, `ash 0.38.0`
already in the cargo cache.

### Decisions taken

`ash` for bindings · `VK_EXT_descriptor_buffer` for the bindless heap · `build.rs` + offline `slangc` ·
port the portable core only.

---

## What the post says about Vulkan, checked against this GPU

The post names specific Vulkan extensions as the things that make the design viable. Every one was checked
against the actual driver rather than assumed:

| Named in the post | On this GPU | Consequence |
| --- | --- | --- |
| `VK_KHR_buffer_device_address` (2019) | ✅ core 1.2 | `Gpu<'a,T>` ports unchanged |
| `VK_EXT_descriptor_buffer` (2022) | ✅ `descriptorBuffer = true` | `ResourceHeap` ports as CPU-written descriptor memory |
| `VK_EXT_host_image_copy` | ✅ `hostImageCopy = true` | texture upload needs **no** staging buffer or command buffer |
| `VK_KHR_unified_image_layouts` (2025) | ❌ **absent** | image layout transitions remain; this is the one unavoidable gap |
| dynamic state (post wants depth/blend out of the PSO) | ✅ `extendedDynamicState` + `extendedDynamicState3ColorBlendEquation` | `depth_state()` stays a state object, exactly as in `nga` |
| `VK_KHR_maintenance4` | ✅ | answers the post's memory-allocation "design flaw" (below) |
| `VK_EXT_shader_object` | ✅ `shaderObject = true` | optional step 8: drop `VkPipeline` entirely |

The post's headline complaint about Vulkan — *"the user has to use raw 64-bit integers instead"* of pointers,
and BDA *"will remain a niche until HLSL and GLSL support pointers natively"* — is precisely what choosing
Slang eliminates. That was verified, not assumed: `Data* data = (Data*)root;` compiles straight to
`PhysicalStorageBuffer` (details below).

### Verified up front

All compiled/queried before writing this plan:

- Slang raw pointers → SPIR-V `PhysicalStorageBuffer` works.
- `-fvk-use-scalar-layout` gives `float3` stride 12 — the exact match for MSL `packed_float3` / Rust `[f32; 3]`.
- `[[vk::push_constant]] uniform uint64_t root` is the direct analog of `table.set_address(addr, 0)`.
- `[SpecializationConstant]` → `OpDecorate %x SpecId 0` — 1:1 with Metal function constant indices.
- Bindless `Texture2D<float4> textures[]` + `NonUniformResourceIndex` compiles on this profile.
- `slangc -reflection-json` emits full member offsets — a usable replacement for MSL `static_assert`.
- Also present: `timelineSemaphore`, `dynamicRendering`, `runtimeDescriptorArray`, `drawIndirectCount`.
- **Absent**: any mesh-shader extension, any tile-shading extension.
- Memory: 7.87 GiB device-local, but only a **214 MiB** host-visible device-local heap (no ReBAR).

---

## The five real differences to be discovered

These are what the port exists to demonstrate. Each is exercised by a working example.

**1. Vulkan's bindless heap is *closer* to the post's design than Metal's — but split in two.**
This is the one place the expected direction reverses. The post explicitly prefers 32-bit heap indices over
Metal's 64-bit `ResourceID`, because Metal *"automatically manages the texture descriptor heap (users can't
allocate contiguous ranges)"*. `nga`'s `ResourceHeap::alloc_range` exists to rebuild that missing capability
by hand; on Vulkan, descriptor buffers give it natively. The genuine Vulkan cost is different: sampler
descriptors need a buffer with `SAMPLER_DESCRIPTOR_BUFFER_BIT` and resource descriptors one with
`RESOURCE_DESCRIPTOR_BUFFER_BIT`, so Metal's *single* heap — which MSL indexes as both `heap[i].t` and
`nga_heap_as(nga_sampler, heap)[i].s` — becomes two. Descriptor size is driver-defined (4 bytes here, not 8)
and must be queried. And the heap is still *bound*, not dereferenced from a `u64` like an `Arena` pointer.

**2. `StorageMode::Shared` does not exist on discrete Windows — but `host_image_copy` covers half of it.**
`nga`'s `Arena` writes structs directly into GPU memory and hands out the address; on this GPU that only works
inside a 214 MiB BAR window. `Arena` fits there comfortably. `GpuHeap` does not, and becomes device-local —
but `VK_EXT_host_image_copy` means `HeapTexture::upload` still writes straight from CPU memory via
`vkCopyMemoryToImageEXT`, with no staging buffer and no command buffer, which is *closer* to Metal's
`replace_region` than a staging path would be. Only large **buffer** uploads need a staging ring.
(The post's caveat that host image copy lacks DCC support applies; irrelevant at this scale.)

**3. Image layouts have no Metal counterpart, and the fix is not available here.**
`nga`'s barriers are `(after_stages, before_stages, visibility)`. Vulkan's `VkMemoryBarrier2` matches that
exactly for buffers — stage + access masks only, no resource lists, which is the barrier design the post asks
for. Images are the exception: they need layout transitions naming individual resources.
`VK_KHR_unified_image_layouts` removes this, and is absent on this driver, so `nga-vk` must carry per-image
layout state that `nga` does not. This is the single irreducible gap.

**4. GPU-side command encoding is gone.** `gpu_driven.metal` builds `render_command cmd(data->commands.cb, slot)`
inside a kernel. `VK_NV_device_generated_commands` exists on this GPU but is vendor-locked; the portable
reformulation is a GPU-written arg buffer plus a GPU-written count consumed by `vkCmdDrawIndirectCount`.

**5. Struct layout must be checked externally.** MSL had `static_assert(__builtin_offsetof(...))` in-language.
Slang has no offsetof, and scalar layout gives `float4x4` alignment 4 where Rust `f32x4x4` is `align(16)` —
a silent mismatch that already appears in a two-field struct. Offsets get verified from SPIR-V reflection at
build time instead.

### Bonus: Vulkan answers a criticism the post levels at it

The post calls out *"a design flaw in Vulkan: you have to create your texture/buffer object first, then you can
ask which heap types are compatible."* `VK_KHR_maintenance4` (supported here) fixes exactly this —
`vkGetDeviceBufferMemoryRequirements` / `vkGetDeviceImageMemoryRequirements` answer from a *create-info*, with
no object created. `GpuHeap` will use these so TLSF sizing needs no throwaway objects, matching Metal's
`heap_texture_size_and_align(desc)` one-for-one. `bufferImageGranularity` still has to be honoured when
buffers and images share the heap, which Metal never requires.

---

## Crate layout

```
nga-vk/
  build.rs            slangc invocation + reflection-based layout verification
  src/lib.rs          the port, mirroring nga/src/lib.rs section for section
  src/abi.rs          vectors + ConstantValue, retargeted at SPIR-V SpecId
  src/slang.rs        Slang prelude + header emission (replaces nga/src/msl.rs)
  src/tlsf.rs         copied verbatim from nga/src/tlsf.rs
  shaders/*.slang     Slang ports of nga/shaders/*.metal
  examples/*.rs       triangle, raster, sync, threaded, bindless, gpu_driven
nga-vk-macros/        fork of nga-macros emitting Slang instead of MSL
```

`nga-vk/Cargo.toml` gains `ash = "0.38"` and `nga-vk-macros`; keeps `minwin` as a dev-dependency.

`nga-vk-macros` is a deliberate fork of `nga-macros`, not a refactor of it — the two stay independent until the
APIs are merged. The parser (`parse_ty`, `scalar`) carries over unchanged; only `declare()` and the generated
`TypeAbi.decl` change to Slang syntax, and the macro emits `::nga_vk::` paths. Its one new job is **padding
synthesis**: it already knows every Rust field offset, so it inserts `uint _padN[k]` members to force the Slang
struct onto the same offsets under scalar layout, which is what makes difference #5 tractable.

`tlsf.rs` is copied rather than shared — pure `u32` arithmetic with no Metal in it, and copying keeps the
"not combined yet" boundary clean.

---

## Mapping, subsystem by subsystem

Working left to right through `nga/src/lib.rs`.

| `nga` | `nga-vk` |
| --- | --- |
| `Gpu<'a,T>` / `GpuMut<'a,T>` | unchanged — a `VkDeviceAddress` is also a `NonZeroU64` |
| `Arena` (shared buffer + `gpu_address`) | `VkBuffer` on `HOST_VISIBLE｜DEVICE_LOCAL`, persistently mapped, `vkGetBufferDeviceAddress` |
| `Pool<T>` | unchanged, it is arena arithmetic |
| `GpuHeap` (`MTLHeap` + TLSF) | one `DEVICE_LOCAL` `VkDeviceMemory` + TLSF, sized via `vkGetDeviceBufferMemoryRequirements`, honouring `bufferImageGranularity` |
| `HeapTexture::upload` | `vkCopyMemoryToImageEXT` — no staging, no command buffer |
| `HeapBuffer::write` | staging ring in the arena + `vkCmdCopyBuffer` |
| `ResourceHeap` (`u64` `ResourceID` array) | **two** descriptor buffers (resource + sampler), `vkGetDescriptorEXT` writes, `u32` slot handles |
| `residency` / `ResidencySet` | deleted — Vulkan memory is resident by construction. Keep a no-op method so examples read the same. |
| `Shaders` + `M4Compiler` | `include_bytes!` of build.rs `.spv` + `vkCreateShaderModule` |
| `FunctionConstantValues` | `VkSpecializationInfo`, index → `SpecId` |
| `RasterDesc` | `VkGraphicsPipelineCreateInfo` + `VkPipelineRenderingCreateInfo` (no render pass object) |
| `depth_state` / `stencil_state` | **stays a state object**: a plain struct replayed as `vkCmdSetDepthTestEnable` / `SetDepthWriteEnable` / `SetDepthCompareOp` / `SetStencilOp` via `VK_EXT_extended_dynamic_state`, keeping `raster.depth(&state)` and avoiding PSO permutations |
| `BlendMode::Fixed` | `vkCmdSetColorBlendEquationEXT` (`extendedDynamicState3`), or baked into the PSO as a fallback |
| `BlendMode::Programmable` | framebuffer fetch — no equivalent on desktop Vulkan; falls back to a second pass |
| `Pass` (load/store actions) | `VkRenderingAttachmentInfo` — near-exact match |
| `Raster` / `Compute` encoders | `VkCommandBuffer` wrappers, same method names |
| `barrier` / `wait` / `signal` | `vkCmdPipelineBarrier2` with `VkMemoryBarrier2`; image barriers added from tracked layout state |
| `Fence` | `VkEvent` (`vkCmdSetEvent2` / `vkCmdWaitEvents2`) — the post's split-barrier pattern |
| `SharedEvent` timeline | `VkSemaphore` with `VK_SEMAPHORE_TYPE_TIMELINE` — 1:1 |
| `M4CommandAllocator` | `VkCommandPool` + `vkResetCommandPool` — 1:1 |
| `Timers` | `VkQueryPool` (`TIMESTAMP`), `timestampPeriod` for ns |
| `Pass::occlusion` | `VkQueryPool` (`OCCLUSION`) + `vkCmdBeginQuery` |
| `IndirectCommands` (ICB) | `VkDrawIndirectCommand` array in the heap + `vkCmdDrawIndirectCount` |
| `Drawable` / `MetalLayer` | `VK_KHR_surface` + `VK_KHR_win32_surface` from `window.hwnd` + swapchain |
| `generate_mipmaps` | hand-rolled `vkCmdBlitImage` loop — no Vulkan equivalent exists |

`Context` / `Frame` / `Recorder` keep their exact shape and method names. The one addition is that
`Frame::submit` must take the acquire and present binary semaphores, because Vulkan's swapchain has no
equivalent of `queue.wait_for_drawable(drawable)`.

### Push-constant root

Metal binds two argument-table addresses (root at 0, heap at 1). Vulkan uses a single push-constant range of
16 bytes: `{ u64 root; u32 resource_offset; u32 sampler_offset; }`. Every pipeline layout in `nga-vk` is
identical — that range plus the two descriptor-buffer set layouts — so pipelines are freely interchangeable,
matching Metal's single-argument-table model and the post's "no root signatures" principle.

### build.rs

For each `shaders/*.slang`: invoke `slangc -target spirv -profile spirv_1_5 -emit-spirv-directly
-fvk-use-scalar-layout -reflection-json`, write `.spv` into `OUT_DIR`, and rerun-if-changed. Then parse the
reflection JSON and compare each `#[gpu_struct]` member offset against the `TypeAbi.fields` table, failing the
build on mismatch. `slangc` is located via `$VULKAN_SDK/Bin/slangc.exe`, falling back to `PATH`, with a clear
error when absent.

---

## Order of work

Each step is a running example, so the design is validated continuously rather than at the end.

1. **`triangle`** — instance/device/queue selection, surface + swapchain from `window.hwnd`, `Arena` on the BAR
   heap, BDA root through push constants, dynamic rendering, timeline semaphore, `Pool<Data>`. This alone
   proves differences #2 and #5 and is the highest-risk step.
2. **build.rs + `nga-vk-macros`** — promote the hand-written triangle shader onto the generated-header path
   and turn on offset verification.
3. **`raster`** — `GpuHeap` via `maintenance4` sizing, depth/blend as *dynamic state*, `Options` specialization
   constants, multiple pipelines.
4. **`sync`** — compute→compute→raster chain, `vkCmdPipelineBarrier2`. Difference #3, on buffers only, where
   Vulkan matches the post's design cleanly.
5. **`bindless`** — the two descriptor buffers, `vkGetDescriptorEXT`, `host_image_copy` upload, storage-image
   writes and sampled reads of the same texture. Differences #1 and #3 together, and the largest divergence.
6. **`threaded`** — per-thread `VkCommandPool`, parallel primary recording, multi-queue submit.
7. **`gpu_driven`** — reformulated onto `vkCmdDrawIndirectCount`. Difference #4.
8. **Optional: `VK_EXT_shader_object`** — re-do `triangle` with no `VkPipeline` at all. This is the closest
   Vulkan gets to Metal 4's model and to the post's "shader IR in, pipeline out" ideal; worth one example to
   measure how much of the PSO machinery it actually deletes.

`meshlets` and `tbdr` are **not** ported: no mesh-shader and no tile-shading extension exists on this GPU.
Both are written up in `nga-vk/NOTES.md` alongside the differences above, with the reformulation each would
need (`VK_EXT_mesh_shader` on Turing+; `VK_KHR_dynamic_rendering_local_read` — which *is* supported here — or
an extra full-screen pass for the imageblock vignette).

---

## Verification

- `cargo build -p nga-vk` — build.rs must compile every `.slang` and pass offset verification. Deliberately
  perturbing a `#[gpu_struct]` field order must fail the build.
- `cargo run -p nga-vk --example triangle` (and each subsequent example) with
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` — **zero validation errors is the acceptance bar**, since
  the whole point is finding where the Metal design violates Vulkan's rules.
- Each example is compared side by side against its `nga` counterpart for visual parity.
- `nga-vk/NOTES.md` is the actual deliverable of the exercise; it grows as each example lands, and closes with
  a per-item verdict against the post's proposed API.
