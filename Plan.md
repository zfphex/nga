# Plan: `nga` + `nga-slang` — one API over Metal 4 and Vulkan 1.4

Supersedes `nga-mtl4/Plan.md` and `nga-vk/Plan.md`. Those two crates proved the design twice, independently, against the same blog post (`target/No Graphics API.md`). This plan folds them into a single crate with one public surface and two private backend files.

The blog post is the inspiration, not the spec. Deliberate departures are listed in §11.

---

## 1. Crate layout

```
nga-slang/            proc-macro only, no dependencies
  src/lib.rs          #[gpu_struct], #[derive(Constants)]

nga/
  Cargo.toml          ash (private, non-apple) | minmetal (private, apple) | nga-slang
  build.rs            slangc: shaders/*.slang -> OUT_DIR/*.spv | OUT_DIR/*.metal
  src/lib.rs          the entire public API: enums, Gpu<T>, TypeAbi, re-exports
  src/slang.rs        SLANG_PRELUDE + header() + type-layout emission
  src/prelude.slang   the shader prelude; include_str!'d by slang.rs *and* build.rs
  src/vulkan.rs       #[cfg(not(target_vendor = "apple"))]
  src/metal.rs        #[cfg(target_vendor = "apple")]
  src/tlsf.rs         unchanged, already byte-identical in both crates
  shaders/*.slang     one source per example, both targets
  examples/*.rs       one source per example, both backends
```

Six source files — `src/prelude.slang` joined them, because the shader prelude turned out to be the one place the two backends genuinely fork (§4). `lib.rs` ends with:

```rust
#[cfg(not(target_vendor = "apple"))]
mod vulkan;
#[cfg(not(target_vendor = "apple"))]
use vulkan as backend;

#[cfg(target_vendor = "apple")]
mod metal;
#[cfg(target_vendor = "apple")]
use metal as backend;

pub use backend::*;
```

No `dyn`, no trait objects, no wrapper structs forwarding into the backend. The backend module *is* the implementation of the public types; `lib.rs` owns everything that does not differ.

**Why not wrapper types.** `pub struct Device(backend::Device)` plus forwarding methods would enforce the surface at compile time, but doubles the line count and adds exactly the indirection this project exists to remove. The cost is that signature drift between the two files is caught by compiling on both platforms, not by the type system — see §10.

---

## 2. The rule: nothing from `ash` or `minmetal` escapes

Today it leaks badly. `nga-vk` puts `vk::Format`, `vk::CompareOp`, `vk::Pipeline`, `vk::CullModeFlags`, `vk::MultiDrawInfoEXT`, `vk::CommandBuffer` and `vk::Result` in public signatures, and `examples/triangle.rs` calls `device.swapchain_ext.queue_present` by hand. `nga-mtl4` re-exports `minmetal::FunctionConstantValues` and hands out `PixelFormat`, `PrimitiveType`, `CullMode`, `Viewport`, `ResourceID`, `RenderPipelineState` and `M4VisibilityOptions`.

Both dependencies become private:

```toml
[target.'cfg(not(target_vendor = "apple"))'.dependencies]
ash = { git = "...", default-features = false, features = ["loaded"] }

[target.'cfg(target_vendor = "apple")'.dependencies]
minmetal = { git = "https://github.com/zfphex/minmetal.git" }
```

Every backend type crossing the boundary is a newtype declared in the backend file:

```rust
// vulkan.rs
#[derive(Clone, Copy)]
pub struct Pipeline(vk::Pipeline);

// metal.rs
pub struct Pipeline(RenderPipelineState);
```

`Texture`, `Sampler`, `Buffer`, `Pipeline`, `Fence`, `Drawable` all follow. Fields stay private to the module, so `pub use backend::*` cannot leak them.

**Checklist for M7 acceptance:** `grep -n 'vk::\|minmetal::' nga/src/lib.rs` returns nothing, and no example imports `ash` or `minmetal`.

---

## 3. What is already shared

Verified identical or near-identical between the two crates today. These move to `lib.rs` (or `tlsf.rs`) and are written once:

| Item | Status |
| --- | --- |
| `tlsf.rs` (218 lines) | byte-identical — `diff -q` passes |
| `Gpu<'a, T>`, `GpuMut<'a, T>`, `GpuBox<'a, T>` | identical |
| `Pool<'a, T>` | identical apart from `Arena` type |
| `f32x2..f32x4x4`, `u32xN`, `i32xN` vectors | identical layouts, different trait impls (§5) |
| `Draw`, `DrawIndexed`, `Dispatch`, `DispatchThreads` | identical |
| `TypeAbi` / `DeviceAbi` | differ only in whether `fields` carries a size (§5) |
| `BlendMode`, `ColorTarget`, `RasterDesc`, `Pass` | same shape, backend enum types |
| `Context` / `Frame` / `Recorder` / `Raster` / `Compute` | same shape and method names throughout |
| the two macro crates (409 vs 420 lines) | differ only in a scalar-name table and error strings |

The convergence is already high enough that this is mostly a mechanical merge plus an enum-translation layer.

---

## 4. `nga-slang`

Merge of `nga-vk-macros` and `nga-mtl4-macros`. The diff between them is one `fn scalar` table and the word "MSL" vs "Slang" in error messages. Since Slang is now the only shading language, the Slang table wins outright and the MSL spellings (`ushort`, `packed_float3`, `nga_texture2d`) are deleted — slangc emits those itself when targeting Metal.

### The M1 gate — resolved

The question was whether one Slang struct can land on the same bytes for both targets, given that MSL's `float3` is 16 bytes while scalar-layout SPIR-V's is 12. **It can, and no spelling change is needed.** Measured, not assumed:

- slangc emits `packed_float3` for the Metal target automatically, so `float3` is 12 bytes on both.
- `float4x4` lowers to `array<packed_float4, 4>` on Metal and to a stride-16 array on SPIR-V — 64 bytes, alignment 4, on both. It does *not* match Rust's `align(16)`, which is exactly why padding synthesis is load-bearing.
- Explicit padding members survive Metal codegen intact (`uint _nga_pad0[2]` → `array<uint, 2>` / `packed_uint2`).
- SPIR-V reports `OpMemberDecorate Offset` of `0, 8, 16, 80, 92` with `ArrayStride 96` for the reference struct — byte-for-byte the Rust layout.

So the Vulkan crate's padding-insertion approach (`slang.rs::declare`) is the single code path for both targets, and the MSL `static_assert(__builtin_offsetof(...))` emitter is deleted rather than ported.

Two further findings, both of which simplify the design beyond what this plan originally assumed:

**`DescriptorHandle<T>` is the portable bindless handle.** Slang lowers it to a `uint2` descriptor index on SPIR-V (only the low word is used as the index) and to a bare `texture2d<...>` / `sampler` argument-buffer member on Metal. Both are 8 bytes at 8-byte alignment and land on identical offsets, so one Rust `Handle(u64)` covers both. This deletes the entire MSL wrapper-struct prelude — `nga_texture2d` and friends existed only because MSL cannot spell a bare `MTLResourceID`, and Slang can.

**`[[vk::constant_id(N)]]` is portable too.** It lowers to `[[function_constant(N)]]` on the Metal target with the *same* index, so `#[derive(Constants)]` needs no fork at all — only a `Scalar` → `MTLDataType` match in `metal.rs`, which is §5's design already.

**`[[vk::push_constant]]` is portable.** The existing root-binding shader form compiles on the Metal target to `Push constant* push [[buffer(0)]]` — a Metal 4 argument-table address at index 0. So the `Push { root, resources, samplers }` design carries over unchanged and the root pointer reaches the shader the same way on both backends, with no source-level difference.

**Where `DescriptorHandle` stops working, and what replaced it.** On SPIR-V, slangc auto-generates the heap arrays and assigns sampled textures and storage textures the *same* `(set 0, binding 2)` with different descriptor types. A `VK_EXT_descriptor_buffer` layout cannot declare that binding twice with two types, so using `DescriptorHandle` on the Vulkan side would mean taking a dependency on `VK_EXT_mutable_descriptor_type` or the very new `VK_EXT_descriptor_heap`.

Neither is worth a hardware requirement, so **the prelude forks and nothing else does**. `build.rs` defines `NGA_METAL` or `NGA_VULKAN` — Slang predefines no target macro of its own — and `src/prelude.slang` picks:

- **Metal:** `typedef DescriptorHandle<Texture2D<float4> > nga_texture2d;` and friends. Eight bytes, an `MTLResourceID` in an argument buffer.
- **Vulkan:** three explicit arrays at binding 0 of sets 0, 1 and 2 — exactly the layout the device already creates — and `struct nga_texture2d { uint2 slot; }`. Eight bytes, of which the low word is the descriptor index and the high word is padding whose only job is to keep the size equal.

Access goes through `nga_sample` / `nga_load` / `nga_store`, which are the only functions that differ between the two halves. Everything above them — the shader body, the shared struct, the Rust `Handle(u64)` — is identical. `tests/layout.rs::handles_are_eight_bytes_on_both_targets` pins the agreement against the real SPIR-V and the real MSL.

The prelude lives in `src/prelude.slang` and is `include_str!`d by both `slang.rs` and `build.rs`, so there is one copy and drift is impossible.

Generated code targets `nga::` paths (`impl nga::DeviceAbi for Foo`), the serde/serde_derive split. `nga` re-exports the macros so users depend on one crate:

```rust
pub use nga_slang::{Constants, gpu_struct};
```

---

## 5. Constants and specialization

`nga-mtl4` uses Metal function constants (`DataType` + raw bytes); `nga-vk` uses SPIR-V specialization constants (`SpecializationMapEntry` + `SPEC_SIZE`, where `bool` is 4 bytes rather than 1). One shared trait, one shared payload type, backend-side consumption:

```rust
// lib.rs
#[derive(Clone, Copy, PartialEq)]
pub enum Scalar { Bool, F16, F32, U32, I32, U64, F32x2, F32x3, F32x4, U32x2, /* .. */ }

pub trait ConstantValue: Copy {
    const SCALAR: Scalar;
    const SLANG_TYPE: &'static str;
}

pub struct Constant { pub index: u32, pub scalar: Scalar, pub bytes: [u8; 16], pub len: u8 }

pub trait Constants {
    fn slang() -> String where Self: Sized;
    fn values(&self) -> Vec<Constant>;
}
```

`vulkan.rs` turns `&[Constant]` into a `VkSpecializationInfo` (widening `Bool` to 4 bytes); `metal.rs` turns it into `FunctionConstantValues` via a `Scalar -> DataType` match. `minmetal::FunctionConstantValues` stops being re-exported.

`TypeAbi::fields` becomes `&[(&'static str, usize, usize)]` — name, offset, size — the Vulkan form. Metal's header emitter ignores the size and keeps emitting `static_assert(__builtin_offsetof(..))`; Vulkan's padding walker needs it.

---

## 6. Shaders: Slang only

`nga-vk/build.rs` already drives `slangc` for SPIR-V. Extend it to emit both targets from the same source:

```
slangc x.slang -target spirv  -profile spirv_1_5 -emit-spirv-directly \
       -fvk-use-scalar-layout -o $OUT_DIR/x.spv -reflection-json $OUT_DIR/x.json
slangc x.slang -target metal -o $OUT_DIR/x.metal
```

Only the host platform's target is built. `nga-vk/shaders/*.slang` (5 files) are the starting point; `nga-mtl4/shaders/*.metal` (8 files) get ported to Slang, and the three with no Vulkan counterpart — `meshlets`, `gpu_meshes`, `tbdr` — wait for M6.

The two preludes collapse. `MSL_PRELUDE`'s `nga_texture2d`-style wrapper structs exist only because MSL cannot spell a bare `MTLResourceID`; in Slang a texture is `Texture2D<float4>` and the Metal codegen path emits the argument-buffer struct itself. What survives in `slang.rs` is:

```slang
[[vk::binding(0, 0)]] Texture2D<float4>   nga_textures[];
[[vk::binding(0, 1)]] RWTexture2D<float4> nga_storage[];
[[vk::binding(0, 2)]] SamplerState        nga_samplers[];

struct nga_push { uint64_t root; uint resources; uint samplers; };
[[vk::push_constant]] ConstantBuffer<nga_push> push;

struct nga_draw { uint vertex_count; uint instance_count; uint vertex_start; uint base_instance; };
struct nga_draw_indexed { /* .. */ };
struct nga_dispatch { uint threadgroups[3]; };
struct nga_dispatch_threads { uint threads[3]; uint group[3]; };
```

plus the `constexpr sampler` set (`nga_linear_clamp` etc.), which becomes a Slang `SamplerState` table. The `[[vk::binding]]` attributes are inert on the Metal target.

**Root argument binding** is the one place the two disagree structurally: Metal binds a 64-bit address into an `MTL4ArgumentTable`, Vulkan pushes the same address as a push constant alongside two descriptor-buffer offsets. The shader side is identical either way (`push.root` / the argument-table address), so this stays entirely inside the backend files.

---

## 7. The public enums

The translation layer. Each backend file carries a `fn` per enum — plain `match`, no macros, no `From` impls to chase.

| `nga` | Vulkan | Metal |
| --- | --- | --- |
| `Memory { Device, Upload, Readback }` | replaces `Placement` (`Bar`/`Device`/`Host`) | `StorageMode` (all `Shared` on unified memory) |
| `Format` | `vk::Format` | `PixelFormat` |
| `TextureKind { D1, D2, D3, Cube, D2Array, CubeArray }` | `vk::ImageViewType` | `TextureType` |
| `Usage` (bitflags: `SAMPLED`, `STORAGE`, `COLOR`, `DEPTH_STENCIL`) | `vk::ImageUsageFlags` | `TextureUsage` |
| `Topology` | `vk::PrimitiveTopology` | `PrimitiveType` |
| `Cull`, `Winding` | `vk::CullModeFlags`, `vk::FrontFace` | `CullMode`, `Winding` |
| `CompareOp` | `vk::CompareOp` | `CompareFunction` |
| `BlendOp`, `BlendFactor` | `vk::BlendOp`, `vk::BlendFactor` | `BlendOperation`, `BlendFactor` |
| `Stage` (bitflags) | `vk::PipelineStageFlags2` | `Stages` |
| `Hazard` (bitflags) | `Visibility` -> `vk::AccessFlags2` | `M4VisibilityOptions` |
| `Error` | replaces `VkError` | replaces `MetalError` |

`Format` is intersection-only: a variant exists only if both `vk::Format` and `PixelFormat` have it. Start with what the examples need — `Rgba8Unorm`, `Bgra8Unorm`, `Rgba16Float`, `Rg11B10Float`, `Rgb10A2Unorm`, `R32Uint`, `D32Float`, `D32FloatS8` — and grow on demand.

`Viewport` and `ScissorRect` become plain `nga` structs (`{ x, y, width, height, min_depth, max_depth }`), since both backends take the same six floats in a different order.

**Barrier argument order.** Both crates currently spell it `barrier(after: Stages, before: Stages, hazard)`, which reads backwards against the blog's `gpuBarrier(cb, STAGE before, STAGE after, HAZARD_FLAGS)`. Normalize on the blog: `barrier(before: Stage, after: Stage, hazard: Hazard)`. This is a silent semantic flip at every call site in the examples — do it in one pass in M4 and check each one.

---

## 8. Reconciling the divergent types

| Type | Resolution |
| --- | --- |
| `Uploader` (Vulkan only, 200 lines) | Vulkan needs a staging ring because device-local memory is not host-visible; Metal's unified memory does not. Becomes private to `vulkan.rs`. Public surface is `GpuHeap::write` / `upload` / `flush` / `flush_async`, which both already have. |
| `Arena::new(.., Placement)` | `Arena::new(size, Memory)`; Metal ignores the argument beyond picking a `StorageMode`. |
| `ResourceHeap` | Vulkan: three descriptor arrays, handle is a `u32` index. Metal: one argument buffer, handle is a 64-bit `MTLResourceID`. Public handle becomes an opaque `Handle(u64)` written by `write_texture`/`write_storage`/`write_sampler`; the shader side already differs and is generated. |
| `ShaderObjects` (VK_EXT_shader_object) | Vulkan-only. Not in the public surface; keep it as a private fast path inside `vulkan.rs` or drop it. Decide in M3. |
| `Timers` / occlusion / `IndirectCommands` | Metal-only today. Out of v1 (§9). |
| `Swapchain` / `MetalLayer` | Unify as `Surface::new(hwnd_or_nsview, width, height)` + `acquire() -> Option<Drawable>` + `present(drawable)`. The raw `queue_present` call in the Vulkan example moves inside. |
| Sync (`vk` timeline semaphore vs `MTLSharedEvent`) | `Fence` newtype: `signal(value)`, `wait(value)`, `value()`. Both map cleanly. |
| `Retired` / `free_later` (Metal) | Deferred-free is needed on both; promote to the shared surface. |

---

## 9. Feature gating (deferred to M6)

Intersection-first. These stay in the old crates until the core lands, then return behind a query:

```rust
pub enum Feature { MeshShaders, TileShaders, IndirectCommands, Timestamps, Occlusion }
impl Device { pub fn supports(&self, feature: Feature) -> bool; }
```

Metal-only today: mesh/meshlet pipelines, tile shaders and imageblocks (`tbdr.rs`, 554 lines), indirect command buffers, `Timers`, occlusion queries, MSAA resolve helpers, `generate_mipmaps`.
Vulkan-only today: shader objects, `Placement::Bar`, `draw_multi` / `draw_indirect_count`.

Vulkan 1.4 has `VK_EXT_mesh_shader` and `vkCmdDrawIndirectCount`, so most of the Metal-only list is reachable later. Tile shaders and imageblocks are not — they stay `#[cfg(target_vendor = "apple")]` permanently, and that is fine.

---

## 10. Risks

1. **The Metal backend cannot be compiled on this machine.** Development is on Windows; `metal.rs` only builds on macOS. Signature drift between the two files will not surface locally. Mitigations: a literal backend-contract comment at the top of both files listing every required item in the same order; porting each milestone on both platforms before starting the next.

   Partly retired. slangc's Metal backend runs on any host, so `build.rs` now emits the MSL on every platform and `tests/layout.rs::metal_output_packs_the_same_way` asserts the Metal-side struct layout from Windows. That covers the *shared ABI* — the part that would have been most expensive to discover late. The Objective-C half of `metal.rs` is still unverifiable here.
2. **slangc's Metal target quality.** The `float3` packing question (§4) is the known one; entry-point naming, `[[vk::binding]]` handling and function-constant emission are unverified. M1 exists to find out before anything depends on it.
3. **`pub use backend::*` hides what the API is.** No single file states the surface. Mitigation: `lib.rs` carries the contract comment; consider generating a `cargo public-api`-style dump in CI if drift becomes real.
4. **Scope.** ~5,300 + ~5,800 lines merging into one crate. The milestones below are each independently shippable; stopping after M5 leaves a working library.

---

## 11. Departures from the blog post

Recorded so they stay deliberate:

- **Lifetimes instead of `void*`.** `Gpu<'a, T>` / `GpuMut<'a, T>` are 8 bytes and `Copy`, but a frame-arena pointer stored in a persistent structure is a compile error rather than a GPU hang.
- **Owned allocators, not a global `gpuMalloc`.** `Arena`, `GpuHeap` and `Pool` take `&mut self`; exclusivity is proven at compile time, so no lock is ever needed.
- **No `gpuFree(void*)`.** Freeing goes through the allocator that owns the range, and deferred frees are queued per frame.
- **`Context` / `Frame` / `Recorder`** are an addition — the blog stops at `gpuStartCommandRecording`. Frames-in-flight, per-frame arenas and multi-threaded recording need somewhere to live.
- **Typed pipelines, not `ByteSpan`.** Shader IR is selected by entry-point name against a compiled module.

---

## 12. Milestones

Each is a working build. Vulkan and Metal are done together within a milestone, never sequentially across milestones (§10.1).

Status: **M0–M5 are done on both backends; M6 is partly done.** `cargo test -p nga` passes eight tests against a real GPU — on Windows against Vulkan with validation enabled, on macOS against Metal 4 — `cargo build` is warning-free on both, and no backend name appears in any test or example.

`metal.rs` landed as a 1,400-line port against the finished Vulkan contract (M0's deviation, now closed). Three things `vulkan.rs` spends hundreds of lines on are absent, and their methods survive as documented no-ops so the two surfaces still match: staging (every heap is `StorageMode::Shared`), image layouts (there are none), and descriptor memory (a handle *is* an `MTLResourceID`, stored inline in the root struct — `ResourceHeap` allocates nothing and is a slot bookkeeper).

Four things the port had to invent rather than copy:

- **The root block is a pointer, not a value.** slangc lowers `[[vk::push_constant]] ConstantBuffer<nga_push>` to `constant nga_push* [[buffer(0)]]`, so binding a root means allocating the 16-byte block from the frame arena and putting *its* address in the argument table.
- **`wait_idle` had to be built.** Metal has no `vkDeviceWaitIdle`. `Device` owns the queue and a private `MTLSharedEvent`, signals it behind everything already committed, and waits — the queue being in-order is what makes that equivalent.
- **Depth state is an object, not dynamic state.** All sixteen (compare, write) pairs are built once in `Context::new` and looked up by value, because creating one inside a render pass is not acceptable.
- **A barrier is two calls.** Metal 4 splits what one `vkCmdPipelineBarrier2` does: `barrier` orders against earlier commands in the same encoder, `barrierAfterQueueStages` against everything already committed. Only the second sees a preceding render pass. Issuing one and not the other is a race, and the offscreen raster test caught it intermittently — six runs in twenty before the fix, zero in twenty after.
- **`Pipeline` is a borrowed pointer.** The pipeline objects live in the `Shaders` that built them, which is exactly the Vulkan lifetime, and `id` is `Send + Sync` so it still crosses threads.

Also: `slangc` is now a build requirement on macOS as well (`~/.local/slang`, symlinked into `PATH`); `nga::shader!("name")` picks `.spv` or `.metal` so no test or example spells the extension.

**M0 — skeleton.** ✅ Create both crates, wire the workspace, move `tlsf.rs` verbatim, move `Gpu`/`GpuMut`/`GpuBox`/`Pool` into `lib.rs`, merge the two macro crates into `nga-slang`. The old crates stay in-tree and building. *Exit: `cargo build` on the four-crate workspace plus the two new ones.*

> Done. It carried one deviation for a while — `metal.rs` was a one-line stub rather than a port — which is now closed; see the status note above.
>
> The original reasoning, kept because it turned out to be right: `metal.rs` is a one-line stub rather than a port. The Vulkan backend was transformed in place — 3,236 lines copied, the now-shared sections deleted, 153 struct fields demoted to `pub(crate)`, and the leaking signatures translated. Porting Metal against the finished contract is cheaper than porting it against a moving one, but it does mean §10.1's "both platforms per milestone" rule is currently broken, and the debt is real.

**M1 — the ABI gate.** ✅ `Scalar`/`ConstantValue`/`Constants`/`TypeAbi` in `lib.rs`, `slang.rs` prelude and header emission, `build.rs` emitting both targets. *Exit met:* `tests/layout.rs` reads the real SPIR-V and the real MSL and checks both against `TypeAbi`. See §4 for what the gate found.

> `build.rs` also needs `-fvk-use-entrypoint-name`: without it slangc renames a module's only entry point to `main`, which the compute test caught immediately.

**M2 — device and memory.** ✅ (Vulkan) `Error` replaces `VkError`, `Device::new(validation)`, `Memory` replaces the now-internal `Placement`, `Arena`, `GpuHeap`, `Uploader` made `pub(crate)`. *Exit met on Vulkan:* `tests/compute.rs` allocates, dispatches and reads back.

**M3 — textures, heaps, pipelines.** ✅ `Format`/`Usage`/`Filter`/`AddressMode` tables, `TextureDesc` in nga vocabulary, `Device::sampler(&SamplerDesc)`, and `ResourceHeap` writes that *return* the typed handle they created instead of leaving the caller to recompute `base + slot`. *Exit met and then some:* `tests/texture.rs` dispatches into a storage texture, and `tests/sample.rs` uploads three textures plus a sampler and checks the shader reached each through its handle.

> Two decisions landed here. `ShaderObjects` (`VK_EXT_shader_object`) was **deleted**, not kept — it has no Metal counterpart, and intersection-first says it does not belong in the shared surface. That also dropped `VK_EXT_shader_object` and `VK_EXT_extended_dynamic_state3` from the device requirements. `TextureKind` and `BlendOp` were dropped rather than left unused; they come back with the code that needs them.
>
> A portability constraint surfaced: **handle arithmetic is Vulkan-only.** The old `bindless` shader did `texture_base + layer`, which is fine when a handle is an index and meaningless when it is an `MTLResourceID`. An array of handles is the portable spelling, and `sample.slang` uses it.

**M4 — commands.** ✅ on Vulkan. `Stage`/`Hazard` replace `Stages`/`Visibility`, and `barrier` now takes `(before, after, hazards)` in the blog's order rather than the reversed one both crates used. `Viewport`/`Scissor` are nga structs, `Compute::transition` takes a `Usage` rather than a `VkImageLayout`, and `Pass::color_texture` renders offscreen. *Exit met:* `tests/raster.rs` rasterises a triangle into a texture and checks the coverage. *Remaining:* an explicit `Fence` type and deferred frees.

> The offscreen test caught a real bug: `Pass` did not update the texture's tracked layout, so a following `read_texture` transitioned out of `UNDEFINED` and discarded everything the pass had drawn. Textures were also leaking their `VkImage`/`VkImageView`; `GpuHeap::free_texture` and `free_buffer` close that.

**M5 — surface and examples.** ✅ on both, for six of the seven. `Swapchain` is now `Surface`, with `format()`/`width()`/`height()`/`present()` — the raw `queue_present` at the call site is gone. `triangle`, `raster`, `bindless`, `sync`, `gpu_driven` and `threaded` all build against the unified API, and **no example imports `ash`**. They have not been *run* on either backend: each opens a window, and per CLAUDE.md that is not something to do unattended — the headless tests are what actually proves the API.

> On the Metal side the six needed two changes. `Surface::new` takes the platform window handle as an `isize`, which is an `HWND` on Windows and the `NSView` on macOS, so each example picks the field with a two-line `cfg`. And `gpu_driven` used `draw_indirect_count`, which Metal has no equivalent for — the GPU-side count can only be consumed by an indirect command buffer. Until `Feature::IndirectCommands` lands, the Metal backend replays all `max` slots instead, which is correct as long as the unclaimed ones read `instance_count == 0`; the example now clears the draw buffer each frame to guarantee that.

> `probe` was **dropped**, not ported. It is pure `VkFormatProperties3` / `VkMemoryHeapFlags` introspection with no backend-neutral meaning; the two facts worth keeping came back as `Device::max_resources()` and `Device::max_multi_draw()`.
>
> Porting `bindless` forced the read-write texture path back: `GpuHeap::transition(texture, Usage)` and `ResourceHeap::write_texture_in(.., Usage)`, both now taking `Usage` rather than a `VkImageLayout`.

**M6 — feature gates.** Query done, examples not. `Feature` and `Device::supports` exist and answer conservatively: `true` only where the backend has a path today, not merely where the hardware could. On Vulkan that is `MultiDraw`, `Timestamps` and `Occlusion`; on Metal it is `Timestamps` and `Occlusion`. Everything else reports `false` on both — including `IndirectCommands` and `MeshShaders`, which Metal 4 genuinely has and `nga-mtl4` genuinely implemented, but `metal.rs` has not ported back. *Remaining:* those two, plus bringing `meshlets`, `gpu_meshes` and `tbdr` back under `#[cfg(target_vendor = "apple")]`. `Timers`, occlusion queries, ICBs and mesh pipelines are all still in `nga-mtl4` to port from.

**M7 — delete.** **Still blocked, but for a smaller reason.** The acceptance criteria are met — `grep -n 'vk::\|minmetal::' nga/src/lib.rs` returns nothing and no example or test imports `ash` or `minmetal` — and `metal.rs` is now a working backend rather than a stub, so the original blocker is gone. What remains is that `nga-mtl4` is still the only implementation of the M6 leftovers: `Timers`, occlusion queries, indirect command buffers, mesh and tile pipelines. Delete `nga-mtl4` once those are ported or explicitly abandoned; `nga-vk` and the two macro crates can go now.
