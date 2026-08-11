# Plan: `nga` — a zero-dependency Rust Metal library on "No Graphics API" principles

Revised for `minmetal` (Metal 4) as the only binding layer, no third-party crates anywhere in the tree, and no C-facing surface. `#[repr(C)]` still appears, but purely as a _layout_ tool for matching MSL — nothing is `extern "C"`, nothing is exported for C consumers, and there is no header to ship.

---

## 1. Dependency budget

| Crate                      | Dependencies                          |
| -------------------------- | ------------------------------------- |
| `nga`                      | `minmetal`, `nga-macros` (both yours) |
| `nga-macros`               | none — `proc_macro` only              |
| `nga-abi` (optional split) | none, `no_std`                        |

`proc_macro` ships with the compiler and is not a Cargo dependency, so a zero-dep proc-macro crate is legal. Everything else — bitflags, TLSF, MSL emission, shader build invocation — is hand-rolled against `std`. This is realistic: none of it is algorithmically deep, and the parts that usually pull crates (`syn`, `quote`, `bitflags`, `inventory`) are each replaced below with something small and specific.

---

## 2. What `nga` needs from `minmetal`

Since the binding details are yours, here's the contract. This doubles as the M0 checklist — if any of these can't be expressed, the design shifts before code is written.

**Load-bearing (design fails without these):**

- Buffer creation with **shared storage** and **untracked hazard mode**, plus a mapped `&mut [u8]` view.
- `MTLBuffer::gpu_address()` and `MTLHeap` placement (`heap_texture_size_and_align(desc)`, `new_texture(desc, offset)`).
- **Residency sets**: create, add allocation, commit, attach to queue.
- **Argument table address binding** — setting a raw `u64` device address at a binding index, not a buffer object. (Metal 4 `MTL4ArgumentTable`.)
- `MTLTexture::gpu_resource_id() -> MTLResourceID` (a `#[repr(C)]` newtype over `u64`), writable into a plain buffer.
- Encoder-level memory barrier with scope + before/after stage masks.
- `MTLSharedEvent` (signal on queue, wait on CPU) and `MTLFence`.

**Nice to have:** indirect command buffers, `optimize_contents_for_gpu_access`, counter sample buffers.

**Style assumptions:** RAII handles (`Retained<T>` or equivalent) with `Drop` releasing; `Send` where Metal permits; `&mut self` on encoders. `nga` never calls `msg_send!` itself — if a selector is missing, it gets added to `minmetal`.

---

## 3. Device pointers: lifetimes instead of `void*`

The C proposal returns `void*` and trusts you. The Rust adaptation replaces that trust with the borrow checker, at zero runtime cost:

```rust
#[repr(transparent)]
pub struct Gpu<'a, T: ?Sized>(NonZeroU64, PhantomData<&'a T>);       // covariant
#[repr(transparent)]
pub struct GpuMut<'a, T: ?Sized>(NonZeroU64, PhantomData<&'a Cell<T>>); // invariant
```

Eight bytes, `Copy`, embeddable in a struct shared with MSL. A frame-arena pointer cannot be stored in a persistent structure — that's a compile error, not a GPU hang. Persistent allocations hand out `Gpu<'static, T>`, which coerces into any shorter `'f`.

`GpuSlice<'a, T>` adds a `u32` length for the optional bounds-clamping debug mode.

---

## 4. Memory: `gpuMalloc`/`gpuFree` → owned allocators, `&mut self` as the lock

No global malloc, no interior mutability. An allocator taking `&mut self` is provably exclusive at compile time, which is exactly why no mutex is ever needed.

| Type      | Backing                          | Use                                  | Free                              |
| --------- | -------------------------------- | ------------------------------------ | --------------------------------- |
| `Arena`   | one shared, untracked buffer     | root data, indirect args, staging    | bump; whole-arena reset per frame |
| `Pool<T>` | slab within an arena             | instances, materials                 | intrusive `u32` free list         |
| `GpuHeap` | `MTLHeap`, placement + untracked | meshes, textures, persistent buffers | TLSF, O(1), no allocation         |

```rust
let d: GpuBox<'f, Data> = frame.arena.alloc(Data { mvp, verts: mesh.verts() })?;
d.cpu_mut().mvp = m;   // direct write into GPU memory (UMA: cached, not write-combined)
pass.draw_indexed(d.gpu(), px.gpu(), mesh.indices, mesh.index_count);
```

`GpuBox<'a, T>` holds `NonNull<T>` + `Gpu<'a, T>` and deliberately **has no freeing `Drop`** — arena contents die at reset. Persistent frees are explicit: `heap.free(&mut self, alloc)`, or a `RetireList` (`Vec<(u64 timeline, Range<u32>)>`) drained at frame start. Sorted `Vec`, no map.

TLSF is ~250 lines: two-level bitmap over size classes, `u32` block indices into a `Vec<Block>`, boundary tags for coalescing. Uses `u32::leading_zeros` — no crate needed.

---

## 5. Residency: the one thing that makes raw pointers viable on Metal

Raw device addresses mean the driver can't infer what a shader touches, and Metal's usual answer (`use_resource` per resource per encoder) is precisely the per-resource tracking being eliminated.

The escape is that we only ever create a handful of arenas and heaps. Build one residency set at init containing all of them, attach it to the queue, and residency cost is O(~10) instead of O(number of resources). Everything is created untracked, so Metal's automatic dependency tracking is off and barriers are ours alone.

Prove this in M0 before anything else is written.

---

## 5. Bindless textures: build the heap Metal won't give you

Aaltonen's complaint that Metal lacks a global indexable heap is about the *allocator*, not the hardware. `MTLTexture.gpuResourceID` is a 64-bit handle writable into a plain buffer, so we own the array and the indexing scheme.

The only real constraint is how MSL spells the array element. There is no `resource_id` type, and no `texture2d<float>` constructor taking a handle — verified against Apple metal 32023.864 (`-std=metal3.2`), where every spelling fails:

```
device const resource_id* heap        ->  unknown type name 'resource_id'
device const metal::resource_id* heap ->  no type named 'resource_id' in namespace 'metal'
texture2d<float> tex(heap[i])         ->  no matching constructor for 'texture2d<float>'
```

`resource_id` appears nowhere in the MSL headers; `MTLResourceID` exists only inside `metal_raytracing`, for acceleration structures. `texture2d`'s constructor set is copy-constructors across address spaces plus a null-texture default — there is no handle-to-texture constructor to call, whatever the array element is named.

What Apple does expose is the one-member argument-buffer struct, which is exactly 8 bytes — one resource ID:

```metal
// In MSL shared header:
struct nga_texture2d { texture2d<float> t; };
static_assert(sizeof(nga_texture2d) == 8, "not one resource id");

fragment float4 my_fs(
    device const Data*         data         [[buffer(0)]],
    device const nga_texture2d* texture_heap [[buffer(1)]]
) {
    uint albedo_idx = data->material.texture_base + 0;
    return texture_heap[albedo_idx].t.sample(nga_linear_repeat, data->uv);
}
```

The CPU side is therefore exactly the intended thing — a plain array of 64-bit handles in GPU memory:

```rust
pub struct ResourceHeap {
    storage: HeapBuffer,     // u64 array in GPU memory; freed via its Alloc
    ranges: RangeAllocator,  // supports contiguous ranges
}
let range: HeapRange = heap.alloc_range(3)?;   // base index -> one u32 in your material
heap.write(range, 0, albedo.gpu_resource_id()?);
```

Metal won't give us a user-indexable heap, so we build one. Because we own the array layout, we get **contiguous descriptor ranges** — the thing Metal's automatic handles can't express — so a material is one `u32 texture_base`, not five 64-bit handles. A `RangeAllocator` (first-fit over a sorted free-range `Vec`) backs it.

`storage` is a `HeapBuffer`, not the `GpuBox` earlier drafts assumed: `GpuBox` is arena-shaped and deliberately has no freeing `Drop`, whereas a descriptor heap is persistent and needs an explicit free. Allocating it from the `GpuHeap` also places it in the heap already covered by the residency set. The element type is `ResourceID` — a `#[repr(C)]` newtype over `u64`, so the same 8 bytes and the same layout as `[u64]`, but it is what `gpu_resource_id()` returns, so no conversion appears at any call site and a stray integer cannot be written into a descriptor slot.

Samplers default to MSL `constexpr sampler` (embedded, as in the post). A dynamic sampler heap needs no extra code: `struct nga_sampler { sampler s; }` is also 8 bytes, so it is the same `ResourceHeap` holding `SamplerState::gpu_resource_id()` values. Compute shaders can write descriptors too, gated by `Hazard::DESCRIPTORS` on the following barrier.

---

## 6. Root data, pipelines and state

- **Compute / raster PSO creation takes IR + a small `RasterDesc`.** No root signature, no vertex layout, no bind group layouts — there's nothing to describe.
- **Depth-stencil is a separate object** (`MTLDepthStencilState`), matching both Metal and the proposal.
- **Blend is baked into the PSO.** Metal has no dynamic blend state, so we take the post's "mobile" branch: bake it, _or_ expose framebuffer fetch (`[[color(0)]]` fragment input) so users write parameterised blend formulas and collapse the permutations themselves. Expose this as `BlendMode::Fixed(BlendDesc)` vs `BlendMode::Programmable`.
- **Function constants** map to specialization constants. A `#[derive(Constants)]` struct generates the index mapping, so the user writes a struct rather than an index-keyed API. Caveat: MSL function constants are scalars/vectors only — a "constant GPU pointer" must be a `ulong` cast in the shader.
- Vertex buffers do not exist. Ever. Vertex fetch is `data->vertices[vid]`.

**Root data** is one device address per stage in the argument table — index 0 for vertex/kernel, index 0 for fragment. Nothing else is ever bound per draw; the descriptor heap of §5 is a second address set once at init, not per-draw state. MSL receives `device const Data* data [[buffer(0)]]`; `const` is what lets Apple's compiler run uniformity analysis and preload into uniform registers, which is why constant buffers can be deleted outright.

**Pipelines** take IR plus a small `RasterDesc`. No root signature, no vertex layout, no bind group layouts — there is nothing to describe. Depth-stencil is a separate object. Blend is baked into the PSO (Metal has no dynamic blend state), with framebuffer fetch exposed as the permutation-collapsing alternative, per the post's mobile branch. Vertex buffers do not exist: fetch is `data->vertices[vid]`.

Function constants map to specialization constants via `#[derive(Constants)]`. Caveat worth stating up front: MSL function constants are scalars and vectors only, so a "constant GPU pointer" must be a `ulong` cast inside the shader.

---

## 7. Barriers, frames, and hand-rolled bitflags

```rust
pass.barrier(Stage::COMPUTE, Stage::COMPUTE, Hazard::empty());
pass.barrier(Stage::RASTER_COLOR_OUT | Stage::RASTER_DEPTH_OUT, Stage::FRAGMENT, Hazard::empty());
```

No resource lists, no image layouts. `Stage` and `Hazard` come from a ~40-line `macro_rules! bitflags` generating a `#[repr(transparent)] struct(u32)` with `BitOr`/`BitAnd`/`Not`/`contains`/`empty` — the 5% of the `bitflags` crate anyone actually uses.

Frames are transient by construction, with encoder scoping enforced by borrows:

```rust
let mut frame = ctx.begin_frame();        // waits event >= n - FRAMES_IN_FLIGHT; resets arenas; drains retire list
{
    let mut rec = frame.recorder();
    let mut pass = rec.raster(&desc, viewport); // Drop ends encoding — nesting errors are borrow errors
    pass.pipeline(&pso);
    pass.draw_indexed(/* … */);
}
frame.submit(None);                        // signals shared event, n += 1
```

Multi-threaded recording: `Context::new` takes a `threads` count and builds `FRAMES_IN_FLIGHT × threads` of everything a recording thread touches exclusively — arena, command allocator, command buffer, argument table. A shared argument table would be a real race, since every draw writes the root address into slot 0.

`frame.recorders(n)` vends `n` `Recorder`s up front, each owning one disjoint set. `Recorder` is `Send + !Sync` (a `PhantomData<Cell<()>>` field), so it moves into a scoped thread but cannot be aliased; `Drop` ends its command buffer. Because recorders borrow the `Frame`, `submit` cannot run until every thread has joined — that's a borrow error, not a runtime check.

No atomic ticket. Vending all recorders in one `&mut self` call makes submission order the vending order, so `submit` commits `command_buffers[..n]` as a contiguous slice in a single `commit:count:`. A ticket would only be needed if threads pulled recorders dynamically from a shared `&Frame`, which would also force the `Frame` to be `Sync`.

The arena's bump head is an `AtomicUsize` with a CAS loop rather than a `Cell`, so `&Arena` is `Sync` and `Recorder` can hold one. Per-thread arenas mean the CAS is uncontended in practice; the atomic is what makes sharing an arena *sound* if a caller ever wants to, and an atomic is not a lock.

`wait`/`signal` (`barrierAfterQueueStages:` / `barrierBeforeQueueStages:`) order an encoder against the *queue's own timeline* — work committed earlier on the same queue — not against a second queue. One queue is their normal case, and they are what makes a multi-recorder frame correct, since two command buffers in one `commit` cannot use an intra-buffer `MTLFence`.

**Honest gap:** Metal has no encoder-level wait-on-memory-address, so the post's `gpuSignalAfter`/`gpuWaitBefore` atomic-max-on-a-pointer scheme can't be reproduced exactly. Split barriers use `MTLFence` intra-buffer (`signal_fence`/`wait_fence`, mapping to `updateFence:afterEncoderStages:` / `waitForFence:beforeEncoderStages:`) and `MTLSharedEvent` across submissions. MTL4 fences carry no `visibilityOptions`, so they order execution only — a device-visibility `barrier` still has to precede the signal when the consumer reads what the producer wrote.

---

## 8. `nga-macros`: proc macros with zero dependencies

This is the part that changes most from the previous plan, so it's worth detail. Replacing `syn`/`quote` is entirely tractable **if the accepted grammar is deliberately narrow**, which it should be anyway — a GPU-shared struct has no business having generics.

**Parsing.** Work directly on `TokenStream`/`TokenTree`. A hand-written recursive-descent pass over `#[gpu_struct] pub struct Name { pub field: Type, … }`:

- Reject generics, lifetimes (except `'static` inside `Gpu<>`), tuple structs, enums, and unions immediately with a targeted `compile_error!`.
- Field types come from a fixed allowlist: `nga-abi` scalars/vectors, `[T; N]`, `Gpu<'static, T>`, `GpuMut<'static, T>`, and other `#[gpu_struct]` types. Anything else is an error naming the field.

Field-type parsing is the only fiddly bit, since `Gpu<'static, T>` spans several `TokenTree`s. Keep it as a small state machine over a flat token vector with an index cursor — roughly 300 lines, and it stays that size because the grammar can't grow.

**Emission.** `quote!` is ergonomics over string building. The dep-free equivalent:

```rust
let out = format!(
    "impl ::nga::DeviceAbi for {name} {{ \
       const MSL_NAME: &'static str = \"{name}\"; \
       const MSL_DECL: &'static str = {msl:?}; \
       const MSL_ASSERTS: &'static str = {asserts:?}; }}"
);
out.parse::<TokenStream>().unwrap()
```

Fully sufficient, and easier to debug than nested `quote!` (`cargo expand` isn't available dependency-free, but the generated string can be `eprintln!`'d behind a feature).

**Diagnostics.** No `syn::Error`, but spans are still reachable: capture the offending `TokenTree`'s `Span`, build `Literal::string(msg)`, call `set_span`, and wrap in `compile_error!`. Errors land on the right field. Nothing else `syn` offers here is missed.

**Hygiene.** Generated code references `::nga::…`; inside `nga` itself, `extern crate self as nga;` makes that resolve.

---

## 9. MSL generation without `inventory` or `build.rs` magic

The macro must not write files — proc macros run on demand and are cached, so file side effects are unreliable. Instead each type carries its MSL as associated consts (above), and a **golden-file test** the user owns does the emission:

```rust
#[test]
fn shaders_are_current() {
    let header = nga::msl::header(&[
        Data::DECL, Vertex::DECL, Meshlet::DECL, Material::DECL,
    ]);
    nga::msl::assert_matches("shaders/generated.h", &header); // rewrites if NGA_BLESS=1, else fails
}
```

`cargo test` fails whenever a Rust struct drifts from the shader header. No registry, no link-time collection, no hidden ordering, no dependency — and the explicit type list matches the library's "you own your data" philosophy rather than fighting it.

The header also carries `static_assert(offsetof(Data, verts) == 16, "…")` lines generated from `core::mem::offset_of!` (stable since 1.77). A layout divergence therefore becomes a **shader compile error**, not silent corruption. This matters more than usual here, because `packed_float3` vs `float3` is the classic way to lose an afternoon.

Shader compilation is `std::process::Command` invoking `metal` and `metallib` from a small `xtask`-style bin target or `build.rs` — no crate required for spawning a process.

---

## 10. Crate layout

```
nga-abi      no_std, zero deps — DeviceAbi trait, MSL-layout scalar/vector types
             (f32x4, packed_f32x3, f16x4, u8x4, u16x2), Gpu/GpuMut
nga-macros   proc-macro, zero deps — gpu_struct, derive(Constants)
nga-alloc    no_std, zero deps — Bump, TLSF, Pool, RangeAllocator
nga          the API surface (~300 lines of public signatures); deps: minmetal + above
nga-examples triangle, gpu-driven meshlets, bindless deferred
```

`nga-abi` and `nga-alloc` can start as modules inside `nga` and split later if you want them reusable from a shader-side or CPU-side crate that shouldn't pull `minmetal`. `nga-macros` must be its own crate — that's a compiler requirement, not a choice.

---

## 11. Milestones

- **M0 — Feasibility spike.** One triangle: untracked shared buffer → `gpu_address` → argument table → residency set → MSL `device const Data*`.
- **M1 — Memory.** Bump/TLSF/Pool, `Gpu<'a,T>`, `GpuBox`, placement textures, staging upload.
- **M2 — Bindless.** Texture heap array, contiguous ranges, `constexpr` samplers. No `NonUniformResourceIndex` marker: that is an HLSL qualifier with no MSL equivalent, and Metal 3 already treats dynamic indexing into a descriptor array as potentially divergent, so a Rust-side marker would be decorative.
- **M3 — Raster.** `RasterDesc`, separate depth-stencil state, render passes, framebuffer-fetch blending.
- **M4 — Sync.** Stage/hazard barriers, fences, shared-event frame loop, frames-in-flight ring.
- **M5 — Macros.** `gpu_struct` parser, MSL emission, offset asserts, golden-file test harness. Deliberately after M1–M4, so the macro is written against real structs rather than guessed ones.
- **M6 — GPU-driven.** Indirect dispatch/draw, ICB multi-draw. Metal is genuinely ahead here: MSL `render_command::set_vertex_buffer` takes a device pointer, so GPU-generated per-draw root data works natively — the thing DX12 and Vulkan can't do.

---

## 12. Risks

- **No dynamic blend state; no memory-address encoder waits.** Both accepted, both mitigated above.
