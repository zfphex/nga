# nga-vk — where Vulkan follows Metal, and where it doesn't

Running notes from porting `nga` (Metal 4) to Vulkan 1.3 + Slang. This file is the deliverable;
the code exists to produce it.

Hardware: NVIDIA GTX 1070 Ti (Pascal), driver 582.66, Vulkan 1.4.312. Vulkan SDK 1.4.357.

---

## Confirmed at M0 (triangle)

The `triangle` example runs with **zero validation errors** and is a structural match for
`nga/examples/triangle.rs`: an `Arena` in GPU-visible memory, a root struct allocated per frame,
its device address handed to the shader, and a timeline semaphore gating frames in flight.

### Slang closes the gap the post complains about

Aaltonen's criticism of Vulkan is that BDA *"will remain a niche until HLSL and GLSL support
pointers natively"* — the user is stuck casting raw `uint64_t`. Slang removes this entirely:

```slang
struct Data { Vertex* vertices; uint2 _pad0; float4x4 mvp; };
[[vk::push_constant]] ConstantBuffer<Push> push;

Data* data = (Data*)push.root;
out.position = mul(data.mvp, float4(data.vertices[vid].position, 1.0));
```

compiles straight to `PhysicalStorageBuffer` addressing. This reads the same as the MSL
`device const Data* data [[buffer(0)]]` it replaces. **Verdict: parity with Metal.**

### The root pointer is simpler than Metal's

Metal needs an `MTL4ArgumentTable` object, sized at creation via `set_max_buffer_bind_count`,
allocated per frame-in-flight per thread, bound per encoder. Vulkan needs a 16-byte push-constant
range in the pipeline layout and one `vkCmdPushConstants`. No object, no per-frame array.

Every `nga-vk` pipeline layout is that single range, so pipelines stay interchangeable the way
they are under one argument table. **Verdict: Vulkan is simpler.**

### Function constants → specialization constants: exact

`[SpecializationConstant] const bool boost = false;` emits `OpDecorate %boost SpecId 0`. The
`SpecId` is the same integer index `nga`'s `set_constant(values, index, ..)` uses for Metal
function constants, so `#[derive(Constants)]` ports with only the payload encoding changed.
**Verdict: parity.**

### Difference #5 is real, and bites immediately

Under `-fvk-use-scalar-layout`, Slang gives `float3` a stride of 12 — matching MSL
`packed_float3` and Rust `[f32; 3]` exactly, which is what makes the whole shared-struct design
survive the port. But it also gives `float4x4` an alignment of 4, where Rust `f32x4x4` is
`align(16)`. In a two-field struct that is already a silent mismatch:

| | `vertices` | `mvp` | size |
| --- | --- | --- | --- |
| Rust `#[repr(C)]` | 0 | 16 | 80 |
| Slang, scalar layout | 0 | **8** | **72** |

MSL caught this class of bug in-language with `static_assert(__builtin_offsetof(...))`. Slang has
no offsetof. For M0 the Slang struct carries a hand-written `uint2 _pad0`; the macro will
synthesise that padding from the known Rust offsets, and `build.rs` will verify the result against
`slangc -reflection-json`. **Verdict: Vulkan/Slang is worse — the check has to move out of the
shader language and into the build.**

### Two feature bits the Metal design implies but Vulkan makes you ask for

Validation caught both on first run:

- `shaderInt64` — required for the `uint64_t` push constant and pointer arithmetic. Metal has no
  equivalent switch; 64-bit addressing is simply on.
- `shaderDrawParameters` — Slang's `SV_VertexID` lowers to `DrawParameters`.

Neither is a limitation, but both are ceremony Metal does not have.

### Image layouts, as predicted

Every colour attachment needs an explicit `UNDEFINED → COLOR_ATTACHMENT_OPTIMAL` barrier on entry
and `→ PRESENT_SRC_KHR` before submit. `nga`'s `Pass` has no counterpart to either.
`VK_KHR_unified_image_layouts` (2025) is the extension that deletes this, and it is **absent on
this driver**, so `nga-vk` carries per-image layout state that `nga` does not.
**Verdict: Vulkan is worse, with a fix that exists but isn't reachable here.**

### Presentation is not `wait_for_drawable`

Metal's `queue.wait_for_drawable(drawable)` / `signal_drawable` / `present` is three calls on the
queue. Vulkan needs a binary semaphore pair per swapchain image, threaded through `vkQueueSubmit2`
and `vkQueuePresentKHR`, plus out-of-date/suboptimal handling and swapchain rebuild. `Frame::submit`
therefore takes the semaphores rather than a drawable. **Verdict: Vulkan is worse.**

### Descriptor sizes are driver-defined

Queried on this driver: sampled image **4** bytes, storage image **4**, sampler **4** — not the
fixed 8 of `MTLResourceID`. The bindless heap stride has to be a runtime value.

---

## Still open

- **M1** — macro-synthesised Slang padding + `build.rs` offset verification.
- **M2 `raster`** — `GpuHeap` via `maintenance4` sizing, depth/blend as dynamic state.
- **M3 `sync`** — `vkCmdPipelineBarrier2` on buffers, where Vulkan should match the post exactly.
- **M4 `bindless`** — the two descriptor buffers; the largest expected divergence.
- **M5 `threaded`**, **M6 `gpu_driven`** (`vkCmdDrawIndirectCount`).
- **Not portable here**: `meshlets` (no mesh-shader extension on Pascal), `tbdr` (no tile shading
  in Vulkan at all).
