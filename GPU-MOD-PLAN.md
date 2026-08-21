# `#[gpu_mod]`: one tag per shader module, not one per item

**Status:** done. `#[gpu]` and `#[gpu_struct]` are gone, every shader lives in a `#[gpu_mod]`,
and no `ENTRIES` list is written by hand. Phase 3 shipped in a smaller form than planned; the
table at the end says how.

## The question

`rust-gpu` needs no `#[gpu_struct]` because it is not a proc macro: it is a rustc codegen
backend, so it reads MIR and `rustc`'s own layout for every type it meets. It still tags what
it cannot infer — `#[spirv(vertex)]` on entry points, `#[spirv(position)]` on builtins — which
is exactly the set we would keep too.

We cannot copy that. A proc macro sees tokens for one item and can look nothing up, so
`Vertex`'s field names, offsets and types have to be written into a `DeviceAbi` impl *at the
struct's definition*, and only a macro that sees the definition can do it. Nothing about
`Ty::Struct(&<Vertex as DeviceAbi>::ABI)` changes.

What can change is *which* macro sees it. An attribute macro on an inline `mod` receives every
item inside it. One tag on the module then does what today needs a tag per struct and per
function, and it can also see the whole entry list, which is the `ENTRIES` half of the problem.

## The shape

```rust
#[gpu_mod]
mod triangle {
    use nga::*;

    struct Vertex { position: f32x3, color: f32x3 }

    #[repr(C, align(16))]
    struct Data { vertices: Gpu<'static, [Vertex]>, mvp: f32x4x4 }

    struct VertexOut { position: f32x4, color: f32x3 }

    #[vertex]
    fn vertex(root: Gpu<'static, Data>, #[vertex_id] vid: u32) -> VertexOut { .. }

    #[fragment]
    fn fragment(input: VertexOut) -> f32x4 { .. }

    fn saturate(x: f32) -> f32 { .. }        // no stage: a helper, as `#[gpu]` alone is today
}
```

Rules, all of them:

- **Every `struct` is a GPU struct.** `#[repr(C)]` + `#[derive(Clone, Copy)]` + `impl DeviceAbi`,
  the current `#[gpu_struct]` expansion verbatim. If the struct already carries a `#[repr(..)]`
  the macro leaves it alone and emits only the derive and the impl — and rejects it unless the
  `repr` names `C`, because a Rust-layout struct is free to reorder its fields and both backends
  write the offsets out as facts — which is how
  `#[gpu_struct(align(16))]` disappears: it becomes the `#[repr(C, align(16))]` Rust already
  has a spelling for, and one less thing for the macro to parse.
- **Every `fn` is a GPU function.** Stage comes from an inert attribute the macro strips:
  `#[vertex]`, `#[fragment]`, `#[compute(8, 8, 1)]`. None means `Stage::Helper`. This is the
  information `#[gpu(vertex)]` carried; only the `gpu` wrapper goes.
- **`ENTRIES` is generated** from every `fn` that had a stage attribute, in source order.
- **`use`, `const`, `type`, `static` pass through untouched.**
- **`#[host]` on any item passes it through untouched**, for the host-side struct or helper that
  wants to live next to the shader it belongs to.
- **An inner `#![..]` attribute is the module's**, not the following item's, and passes through.
- **`unsafe`, `const`, `async` and `extern` functions are rejected**, with the span on the
  qualifier. Left alone they read as an item kind of their own and pass through untagged, which
  fails much further from the cause.

Everything else stays: the body is still re-emitted so `rustc` type-checks it, the `GpuFn`
consts are still named `VERTEX`/`FRAGMENT`, `calls` still links by const reference, and both
backends are untouched.

## Why the attribute on a `mod`, not a `gpu!{ .. }` bang

`rustfmt` formats inside an attribute-macro module and skips the body of a function-like macro,
and `cargo expand` is not something we want to need to read a shader. The cost is that
`#[gpu_mod] mod sync;` — the out-of-line form — cannot work: the macro receives `mod sync;` and
has no stable way to read the file (`Span::source_file` is unstable). A shared shader file
therefore holds one `#[gpu_mod] pub mod shader { .. }` and a `pub use shader::*;` under it, so
`#[path]` pulls it in as it does today and `sync::Vertex` still names the struct. The nesting is
one level and the `#[path]` line is already written.

The same nesting is what the three Metal-only examples use for the opposite reason:
`gpu_meshes`, `meshlets` and `tbdr` are host modules that share a struct with a hand-written
`.metal` shader, so the structs sit in a nested `#[gpu_mod] mod data` and the host code around
them is never asked to be shader code. `#[host]` per item would have been the alternative and
is much noisier when the host half is most of the module.

## The import half

Three things make a shader awkward to move today, and the first two go with the module macro:

1. `pub const ENTRIES: &[&nga::ir::GpuFn] = &[&SIMULATE, &BUILD_VERTICES, &VERTEX, &FRAGMENT];`
   hand-maintained in seven places. Generated.
2. A new shader function silently missing from `ENTRIES` fails at pipeline creation as a
   missing function name, not at the definition. Cannot happen once the list is derived from
   the definitions.
3. `shaders.compute("simulate", None)` and `RasterDesc::new("vertex", "fragment", ..)` name
   entry points with strings, so renaming a shader function breaks at runtime and the host code
   cannot be moved without hand-editing the names.

For (3): the call sites name the const instead of the string —
`shaders.compute(sync::SIMULATE.name, None)`, `RasterDesc::new(VERTEX.name, FRAGMENT.name, ..)`.
Renaming or moving a shader now breaks at compile time, at the call site, which is the whole
point of the change.

Taking `&'static ir::GpuFn` in those signatures instead was the first plan and is worse: the
descriptors hold `&'a str`, so a `GpuFn` argument means either a lifetime-carrying trait to
borrow the name out of, or a second `*_named` constructor beside every existing one for the six
hand-written MSL shaders that have no `GpuFn`. Both are more API and more indirection than
`.name` at the call site, and neither buys a check that `.name` does not already give.

Composing two modules into one library needs nothing new:
`nga::compile(&[a::ENTRIES, b::ENTRIES].concat())`.

## Phases

| # | Deliverable | Done when |
|---|---|---|
| 0 | `#[gpu_mod]` in `nga-gpu/src/lib.rs`: walk items, dispatch to `expand_struct` for structs and `gpu::expand` for fns, strip stage attributes, emit `ENTRIES`. | `tests/gpu.rs` reads the IR, both emitters and the entry list back |
| 1 | Migrate the seven examples and `tests/gpu.rs`: drop the per-item tags, `#[gpu_struct(align(16))]` → `#[repr(C, align(16))]`, delete every hand-written `ENTRIES`. | every example still draws; `cargo test` clean |
| 2 | Delete `#[gpu]` and `#[gpu_struct]` and their `pub use`. `align(N)` attribute parsing goes with them. | nothing outside `gpu_mod` calls the two expansions |
| 3 | Entry points named through their `GpuFn` const at the call site. | no string literal names a shader function |

Phase 0 is a dispatcher over an item list: both expansions already take token slices and are
reused unchanged, so it is roughly 150 new lines against ~40 deleted, and phases 1–2 are
mechanical.

## Not doing

Inferring the stage from the parameter list (a `#[vertex_id]` implies `vertex`). It reads as
magic, and a compute shader with no builtin parameter has nothing to infer from.

Reading an out-of-line `mod` from disk. Unstable, and `#[path]` already works.

Making `compile` variadic or const-concatenating `ENTRIES`. `.concat()` at pipeline creation
costs nothing and one call site wants it.
