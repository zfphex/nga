# nga-vk — where Vulkan follows Metal, and where it doesn't

Findings from porting `nga` (Metal 4 + MSL) to Vulkan 1.3 + Slang. This file is the deliverable;
the code exists to produce it.

Hardware: NVIDIA GTX 1070 Ti (Pascal), driver 582.66, Vulkan 1.4.312. Vulkan SDK 1.4.357.
Every example runs with **zero validation errors** under `VK_LAYER_KHRONOS_validation`.

| Example | Ports? | What it proves |
| --- | --- | --- |
| `triangle` | yes | BDA root pointer, dynamic rendering, timeline semaphore |
| `triangle_shader_object` | yes | the same frame with no `VkPipeline` at all |
| `raster` | yes | `GpuHeap`/TLSF, depth + blend as dynamic state, specialization constants |
| `sync` | yes | compute→compute→raster barriers |
| `bindless` | yes | descriptor buffers, GPU-painted storage image |
| `threaded` | yes | 4 threads, 4 command pools, one target |
| `gpu_driven` | reformulated | GPU-decided draw count via `vkCmdDrawIndirectCount` |
| `meshlets` | **no** | no mesh-shader extension on Pascal |
| `tbdr` | **no** | no tile shading in Vulkan at all |

---

## Where Vulkan is better

### Slang erases the post's main complaint about Vulkan

Aaltonen argues BDA *"will remain a niche until HLSL and GLSL support pointers natively"* — the
user is stuck casting raw `uint64_t`. Slang removes this completely:

```slang
struct Data { Vertex* vertices; uint _nga_pad0[2]; float4x4 mvp; };
Data* data = (Data*)push.root;
out.position = mul(data.mvp, float4(data.vertices[vid].position, 1.0));
```

That compiles straight to `PhysicalStorageBuffer` addressing and reads the same as the MSL
`device const Data* data [[buffer(0)]]` it replaces. Slang also gave us pointer atomics
(`Atomic<uint>*` → `OpAtomicIAdd`), which `gpu_driven` depends on.

### The root pointer is simpler than Metal's

Metal needs an `MTL4ArgumentTable`: an object, sized at creation via `set_max_buffer_bind_count`,
allocated per frame-in-flight per thread, bound per encoder. Vulkan needs a 16-byte push-constant
range and one `vkCmdPushConstants`. No object, no per-frame array, nothing to size in advance.

### Residency sets disappear

`nga` must build a `MTLResidencySet`, add every arena and heap, commit it, and attach it to each
queue — the mechanism that makes raw pointers viable on Metal at all. Vulkan memory is resident by
construction. The whole subsystem deletes to nothing.

### `maintenance4` answers the post's own criticism of Vulkan

The post calls out *"a design flaw in Vulkan: you have to create your texture/buffer object first,
then you can ask which heap types are compatible."* `vkGetDeviceBufferMemoryRequirements` and
`vkGetDeviceImageMemoryRequirements` answer from a create-info with no object created, which is
exactly `heap_texture_size_and_align(desc)`. `GpuHeap` uses them and never makes a throwaway object.

### Bindless is closer to the post's ideal than Metal is

The expected direction reverses here. The post prefers 32-bit heap indices to Metal's 64-bit
`ResourceID`, because Metal *"automatically manages the texture descriptor heap (users can't
allocate contiguous ranges)"*. `nga`'s `ResourceHeap::alloc_range` exists to rebuild that missing
capability by hand. On Vulkan the descriptor buffer gives it natively, and the handle is already
a `u32` index.

---

## Where Vulkan is worse

### 1. Image layouts, with the fix out of reach

`nga`'s barriers are `(after_stages, before_stages, visibility)`. For buffers Vulkan matches that
exactly — `VkMemoryBarrier2` is stage and access masks with no resource list, which is precisely
the barrier design the post asks for. Images are the exception: every attachment needs an explicit
transition, and `Pass` has to know whether it clears (discard, `UNDEFINED`) or loads (preserve,
`COLOR_ATTACHMENT_OPTIMAL`) — a distinction Metal's load actions make without any layout concept.

`VK_KHR_unified_image_layouts` (2025) deletes this entirely and is **absent on this driver**.
This is the one irreducible gap.

### 2. `VK_EXT_host_image_copy` is advertised but unusable

The plan expected this to recover half of `StorageMode::Shared`: CPU memory straight into an image
with no staging buffer and no command buffer, close to Metal's `replace_region`. It does not work
here. An image created with `HOST_TRANSFER_EXT` usage reports **`memoryTypeBits == 0`** — no memory
type can back it — even though the extension is enabled and `R8G8B8A8_UNORM` advertises
`HOST_IMAGE_TRANSFER` in its optimal-tiling format features.

`examples/probe.rs` demonstrates it, and rules out `maintenance4` as the cause since the
create-info query and the created-object query agree:

```
SAMPLED|TRANSFER_DST                   maintenance4 bits 0x0082   object bits 0x0082
SAMPLED|TRANSFER_DST|STORAGE           maintenance4 bits 0x0082   object bits 0x0082
SAMPLED|TRANSFER_DST|HOST_TRANSFER     maintenance4 bits 0x0000   object bits 0x0000
SAMPLED|HOST_TRANSFER|STORAGE          maintenance4 bits 0x0000   object bits 0x0000
R8G8B8A8_UNORM optimal supports HOST_IMAGE_TRANSFER: true
```

So texture upload stages through a buffer. A layout transition is then only a barrier, which is
free as long as it is *recorded* rather than submitted — see §3.

### 3. `StorageMode::Shared` does not exist on discrete Windows

`nga`'s `Arena` writes structs directly into GPU memory and hands out the address. On this GPU that
only works inside a **214 MiB** host-visible device-local BAR window (no ReBAR). `Arena` fits
comfortably; `GpuHeap` does not and is device-local, so `HeapBuffer::write` — a memcpy in Metal —
becomes a staged copy. Vulkan also imposes `bufferImageGranularity` (1024 here), so every heap
allocation is aligned to it to keep buffers and images off the same granularity page; `MTLHeap`
never asks for this.

`cargo run --example probe` prints the whole topology. On this machine:

```
heap 0:   8060.00 MiB total,   7287.59 MiB free  DEVICE_LOCAL
heap 1:   8146.38 MiB total,   7472.27 MiB free
heap 2:    214.00 MiB total,    205.38 MiB free  DEVICE_LOCAL   <- the BAR window

Bar      (host-visible VRAM)   -> type 10 heap 2,    205.38 MiB free
Upload   (write-combined RAM)  -> type  8 heap 1,   7472.27 MiB free
Readback (cached RAM)          -> type  9 heap 1,   7472.27 MiB free
```

Note the BAR figure is a *budget*, not a size: the window is shared with every other process on
the desktop, so ~9 MiB of it is already gone before the program starts.

Three consequences the Metal design has no equivalent of:

**Placement is explicit.** `Arena::new` takes a `Placement`, because "host-visible" names three
physically different kinds of memory here and the wrong one is silently slow. `Bar` is the window
above and is reserved for per-frame root data. `Upload` is write-combined system RAM — fast
sequential CPU writes, read once by the DMA engine, and it costs no BAR, which is exactly what a
staging buffer wants. `Readback` is cached system RAM for the other direction. An earlier version
allocated staging from `Bar`, which spent the scarcest memory on the one buffer no shader ever
reads.

**The BAR budget is checked up front.** `Context` allocates `FRAMES_IN_FLIGHT * threads` arenas, so
a large arena and a high thread count silently exceed 214 MiB. `Context::with_layout` compares the
total against `VK_EXT_memory_budget` and fails with the numbers rather than returning
`ERROR_OUT_OF_DEVICE_MEMORY` several arenas into the loop. `Arena::new` also treats a full heap as
a reason to try the next candidate memory type, not a reason to give up — a full heap and an
absent heap are different failures.

**Uploads batch instead of blocking.** This is the one that mattered most. Every staged copy used
to submit its own command buffer and then `vkQueueWaitIdle`, so loading a scene drained the whole
device once per resource, and staging was capped at 8 MiB with a panic past it. `Uploader` records
into an open batch over a wrapping ring and submits once on `flush`, signalling a timeline instead
of idling:

| | before | now |
| --- | --- | --- |
| cost of one `write`/`upload`/`transition` | submit + full device drain | memcpy + a few command words |
| a 32-texture load | 64+ drains | **1 submission** |
| write larger than staging | panic at 8 MiB | chunked through the ring |
| staging memory | 8 MiB of the BAR window | 32 MiB of system RAM |
| ≤ 64 KiB write | staged copy | `vkCmdUpdateBuffer`, ring untouched |

`GpuHeap::flush` blocks once for the whole batch, which is what load-time code wants.
`flush_async` returns the timeline value instead, and `Frame::after_uploads` makes the *GPU* wait
on it, so streaming uploads never block the CPU at all. `tests/upload.rs` covers the boundaries by
reading the bytes back: the `vkCmdUpdateBuffer` cutoff, a write three times the ring size, an
image taller than the ring, and that a whole batch is one submission.

**Still on the table: the copy engine.** `probe` reports a dedicated transfer family (2 queues,
`transfer` without `graphics` or `compute`) that could DMA in parallel with rendering. The uploader
records on the graphics queue instead, because moving it costs queue-family ownership transfers
(or `SharingMode::CONCURRENT`) and a split of transitions — graphics-stage barriers are not legal
on a transfer queue. With all uploads currently at load time there is nothing to overlap, so it is
the right change only once assets stream during rendering.

### 4. Metal's stage-agnostic visibility flag doesn't survive

`MTL4VisibilityOptions::DEVICE` means the same thing whatever stages a barrier joins. Vulkan
access masks are stage-constrained: `SHADER_WRITE` is illegal with `ALL_TRANSFER`, `SHADER_READ`
is illegal with `DRAW_INDIRECT`. Validation caught both. Keeping `barrier(after, before, hazard)`
portable across the whole API means `Visibility::Device` maps to `MEMORY_READ`/`MEMORY_WRITE`,
the only masks legal with every stage — correct, but strictly more conservative than Metal.

### 5. Struct layout has to be checked outside the shader language

MSL checks this in-language with `static_assert(__builtin_offsetof(...))`. Slang has no offsetof.
Worse, under `-fvk-use-scalar-layout` — needed because it makes `float3` 12 bytes, matching
`packed_float3` and `[f32; 3]`, which is what lets the shared structs port at all — `float4x4`
gets alignment 4, where Rust `f32x4x4` is `align(16)`. A two-field struct already diverges:

| | `vertices` | `mvp` | size |
| --- | --- | --- | --- |
| Rust `#[repr(C)]` | 0 | 16 | 80 |
| Slang, scalar layout | 0 | **8** | **72** |

**Offline compilation also breaks the generated-header trick.** `nga` builds its MSL header at
runtime from `TypeAbi` and hands it to Metal's runtime compiler. `nga-vk` compiles SPIR-V in
`build.rs`, which runs *before* rustc expands `#[gpu_struct]` — so the shader cannot be generated
from the Rust types at all. Shaders are hand-written and *verified* instead:

- `nga_vk::slang::declare` synthesises the correct padding from the recorded Rust offsets, so a
  mismatch is fixable, not merely detectable. It emits `uint _nga_pad0[2]` before `mvp`.
- `tests/layout.rs` parses `OpMemberDecorate ... Offset` out of the compiled SPIR-V and compares
  against `TypeAbi`. Note that `slangc -reflection-json` is *not* usable for this: it describes
  only the binding interface, so structs reached through a pointer — every struct in this design —
  never appear in it.

Deleting the padding from `triangle.slang` produces:

```
Data::mvp is at 16 in Rust but 8 in Slang — add padding to the Slang struct
```

### 6. One bindless heap becomes three

Metal writes one `u64` per texture *or* sampler into a single buffer, and MSL reinterprets it per
use (`heap[i].t`, `nga_heap_as(nga_sampler, heap)[i].s`). Vulkan cannot: sampler descriptors need
a buffer created with `SAMPLER_DESCRIPTOR_BUFFER_BIT` and image descriptors one with
`RESOURCE_DESCRIPTOR_BUFFER_BIT`. So one heap becomes three arrays across two buffers — sampled
images, storage images, samplers. `nga-vk` keeps a unified index space so `range.base + slot`
arithmetic ports unchanged, at the cost of reserving a slot in all three.

Descriptor size is driver-defined and must be queried: **4 bytes** here for all three types, not
Metal's fixed 8. And a texture written and sampled in the same frame stays in `GENERAL`, so its
sampled descriptor must record that layout — `write_texture_in` exists only because of layouts.

### 7. GPU-side command encoding is gone

`gpu_driven.metal` constructs `render_command cmd(data->commands.cb, slot)` inside a kernel and
gives each encoded draw *its own root pointer*. Vulkan has no GPU-side encoder.
`VK_NV_device_generated_commands` exists on this GPU but is vendor-locked.

The portable reformulation is a GPU-written `VkDrawIndirectCommand` array plus a GPU-written count
consumed by `vkCmdDrawIndirectCount`. It only works because every draw shares one pipeline *and*
one push constant — so per-draw data has to be smuggled through `first_instance`, which the vertex
stage reads back as `gl_BaseInstance`. Metal needs no such trick. Resetting the count is
`vkCmdFillBuffer` rather than Metal's dedicated ICB reset.

### 8. Presentation is not `wait_for_drawable`

Metal is three queue calls. Vulkan needs a binary semaphore pair per swapchain image threaded
through `vkQueueSubmit2` and `vkQueuePresentKHR`, plus out-of-date/suboptimal handling and
swapchain rebuild, so `Frame::submit` takes semaphores rather than a drawable.

### 9. Programmable blending has no equivalent

`raster.metal`'s `fragment_screen` reads `float4 dst [[color(0)]]` — framebuffer fetch. Desktop
Vulkan has none. This particular formula survives by luck: screen blending is
`src + dst*(1-src)`, which `ONE` / `ONE_MINUS_SRC_COLOR` expresses exactly as fixed function.
A formula that genuinely needs the previous value cannot be ported.

### 10. Two feature bits the Metal design implies

`shaderInt64` (for the `uint64_t` push constant and pointer arithmetic) and `shaderDrawParameters`
(Slang's `SV_VertexID` lowers to `DrawParameters`). Also `drawIndirectCount`. Metal simply has
64-bit addressing on. Validation caught all three.

---

## Where they are equal

- **Function constants → specialization constants.** `[[vk::constant_id(N)]]` pins the `SpecId` to
  the field index in `#[derive(Constants)]`, so the mapping is exact rather than order-dependent.
  Only the payload encoding differs: SPIR-V booleans are `VkBool32`, so `bool` writes 4 bytes.
- **Timeline semaphores.** `MTLSharedEvent` → `VkSemaphore` with `VK_SEMAPHORE_TYPE_TIMELINE`, 1:1.
- **Command allocators.** `MTL4CommandAllocator` → `VkCommandPool` + `vkResetCommandPool`, 1:1.
- **Load/store actions.** `MTLRenderPassDescriptor` → `VkRenderingAttachmentInfo` under dynamic
  rendering, near-exact.
- **Depth/stencil as a state object.** `depth_state()` + `raster.depth(&state)` survives intact via
  `VK_EXT_extended_dynamic_state`, and blend via `extendedDynamicState3` — so the post's principle
  that these belong outside the PSO holds on both APIs.
- **TLSF.** Copied from `nga` unchanged; it is pure `u32` arithmetic with no API in it.
- **Multi-threaded recording.** One command pool per thread per frame; `Arena` uses the same
  lock-free atomic bump as `nga`, so `Frame::recorders(&[n])` ports directly.

---

## `VK_EXT_shader_object`: the closest Vulkan gets to Metal 4

`examples/triangle_shader_object.rs` is `triangle` with no `VkPipeline` anywhere. It renders an
identical frame with zero validation errors, so the diff between the two files is a clean
measurement of what the extension costs and saves.

**What it deletes.** Pipeline creation collapses to one call naming two entry points:

```rust
let shaders = ShaderObjects::raster(&device, SHADERS, "vertex_main", "fragment_main")?;
```

Gone entirely: `VkGraphicsPipelineCreateInfo` and its ten sub-structs, the colour-target format
list, `VkPipelineRenderingCreateInfo`, the dynamic-state array, and the blend/depth/multisample
descriptions. `RasterDesc` has nothing left to describe — the target formats a PSO must be built
against are simply not needed, which means a shader object is genuinely reusable across passes
with different attachment formats. A `VkPipeline` never is.

That is the real prize, and it is exactly Metal 4's model: `MTL4RenderPipelineDescriptor` is
little more than its function descriptors plus target formats, and `nga`'s `Shaders::raster`
already reads that way.

**What it costs.** All that state does not vanish; it moves to the command buffer. `bind` makes
**20 calls** — one `vkCmdBindShadersEXT` plus 19 state setters — and none are optional. Omitting
any one is undefined behaviour, not a default:

```
cmd_set_vertex_input            cmd_set_rasterizer_discard_enable   cmd_set_rasterization_samples
cmd_set_primitive_topology      cmd_set_polygon_mode                cmd_set_sample_mask
cmd_set_primitive_restart_enable cmd_set_cull_mode                  cmd_set_alpha_to_coverage_enable
cmd_set_viewport_with_count     cmd_set_front_face                  cmd_set_logic_op_enable
cmd_set_scissor_with_count      cmd_set_depth_bias_enable           cmd_set_color_blend_enable
                                cmd_set_depth_test_enable           cmd_set_color_write_mask
                                cmd_set_depth_write_enable
                                cmd_set_stencil_test_enable
```

It also needs four more device features on top of `shaderObject` — the `extendedDynamicState3`
bits for polygon mode, rasterization samples, sample mask, alpha-to-coverage and logic-op — since
those setters have no pipeline fallback once there is no pipeline.

**Verdict.** Creation-side, this is a clear win and the better match for the design: no PSO
permutations, no format coupling, no descriptor structs. Record-side it is worse than the
dynamic-state pipeline used everywhere else in `nga-vk`, which sets 10 states rather than 19
because the pipeline still carries the rest. For a renderer that already treats depth and blend
as state objects, shader objects mostly trade a build-time cost for a per-draw one. Worth it when
pipeline permutations are the problem; not worth it here, which is why the other six examples
keep their pipelines.

---

## Not portable on this hardware

- **`meshlets`** — no `VK_EXT_mesh_shader` on Pascal (Turing+ only). The mapping is otherwise
  close: object shader → task shader, `mesh_grid_properties::set_threadgroups_per_grid` →
  `DispatchMesh`, `out.set_primitive_count` → `SetMeshOutputsEXT`, and payloads match.
- **`tbdr`** — Vulkan has no tile shading or imageblocks on desktop at all. `tile_vignette` reads
  and writes the imageblock in place; the nearest reformulation is
  `VK_KHR_dynamic_rendering_local_read` (which *is* supported here) or an extra full-screen pass.

---

## Verdict against the post's design principles

| Principle | Metal 4 | Vulkan 1.3 + Slang |
| --- | --- | --- |
| Single 64-bit root pointer | yes | **yes, simpler** (push constant, no table object) |
| Native pointers in shaders | yes | **yes, via Slang** |
| Bindless via 32-bit indices | no (64-bit ids, no ranges) | **yes** |
| No persistent binding objects | yes | yes (descriptor buffers are plain memory) |
| Stage-only barriers | yes | yes for buffers, **no for images** |
| Dynamic depth/blend state | yes | yes |
| Minimal PSO state | yes | yes (and `VK_EXT_shader_object` removes the PSO entirely) |
| CPU writes directly to GPU memory | yes (UMA) | **partly** (214 MiB BAR; no working host image copy) |
| Upload without a stall | free (memcpy) | batched: one submission per flush, none per resource |
| GPU-side command encoding | yes | **no** |
| Timeline semaphores | yes | yes |

The design survives the port. Six of the eight `nga` examples run unmodified in structure, and two
subsystems — the root pointer and residency — come out simpler. The costs are concentrated in
image layouts, host-visible memory, and GPU-side encoding.
