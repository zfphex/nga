# nga-gpu: Rust functions as shaders, straight to SPIR-V and MSL

Replaces the `.slang` sources and `slangc` with `#[gpu]`-tagged Rust functions compiled by us.
`#[gpu_struct]` stays and grows a typed field list; `slang.rs` goes when the last `.slang` file
does.

No dependencies: `proc_macro` only in the macro crate, plain `std` in `nga`.

**Status:** done. All five phases are built and tested (§8), every shader is a `#[gpu]`
function, and `slangc`, `slang.rs`, `prelude.slang` and the `.slang` sources are gone. What
remains is the list of subset gaps at the end of §8 and the deliberate omissions in §9.

## 1. The split

The macro crate stays a parser, exactly like `#[gpu_struct]` does today: it turns tokens into a
`const` and never thinks about a backend. Both backends live in `nga`, as ordinary code that can
be read, stepped through and unit-tested.

```
nga-gpu/src/gpu.rs      #[gpu] : tokens -> GpuFn const  (+ the body re-emitted as normal Rust)
nga-gpu/src/lib.rs      #[gpu_struct] and #[derive(Constants)], sharing the same type parser
nga/src/ir.rs           GpuFn / Node / Ty, and the one type-inference pass both backends run
nga/src/spirv.rs        &GpuFn... -> Vec<u8>
nga/src/msl.rs          &GpuFn... -> String
nga/src/vector.rs       f32x4 and friends, with host bodies, so a shader body compiles
```

`nga-slang` was renamed `nga-gpu`: it holds both macros now, and they share one Rust-type
parser rather than two.

Nothing about SPIR-V or Metal is reachable from a proc macro, so a backend bug is a normal
breakpoint instead of a `cargo expand` session.

## 2. What `#[gpu]` emits

```rust
#[gpu(compute(8, 8, 1))]
fn clear(root: Gpu<'static, Scene>, #[thread_id] id: u32x3) {
    let px = id.x + id.y * root.width;
    root.pixels[px] = f32x4::splat(0.0);
}
```

expands to *two* things:

1. **The function, unchanged.** It still compiles as Rust, so `rustc` type-checks the shader for
   free. That is the single largest win over Slang: the errors we get today from `slangc` at
   build time, or from the driver at pipeline creation, become ordinary Rust type errors with
   spans. `f32x4`, `Gpu<T>` indexing and the intrinsic set must therefore all have real CPU
   bodies in `nga` (a `sample()` that returns zero is enough - it only has to compile).
2. **`pub const CLEAR: GpuFn`**, the parsed body.

### Cross-function and cross-type resolution without a registry

A proc macro sees one item and cannot look anything up. Rust's own name resolution does it
instead: a call `helper(x)` lowers to `Node::Call { func: 0, .. }` plus an entry in

```rust
calls: &[&HELPER],                          // the other fn's own const
```

and a field of type `Foo` lowers to `Ty::Struct(&<Foo as DeviceAbi>::ABI)`. Both are const
expressions, so linking is the linker's job, works across crates, and an unknown name is a
normal "cannot find value" error rather than something we have to diagnose.

## 3. The subset

Accepted: `let` (annotated or inferred), assignment and compound assignment, `if`/`else`,
`while`, `loop`, `for i in a..b`, `break`, `continue`, `return`, unary/binary operators, field
access, swizzles, indexing, `as` casts between scalars, calls to `#[gpu]` functions and to
intrinsics, struct literals, array literals.

Rejected with a span: closures, generics, traits, `match`, `impl` blocks, references, `Box`/`Vec`/`String`, `?`, iterators other than a bare range,
recursion (SPIR-V forbids it; detected by walking `calls`), and anything else.

Types: `f16 f32 i32 u32 bool`, the `f32x2..4 / u32x2..4 / i32x2..4` vectors and `f32x3x3
f32x4x4` already spelled in `#[gpu_struct]`, `[T; N]`, `#[gpu_struct]` structs, and
`Gpu<'static, T>` / `GpuMut<'static, T>` pointers. Identical vocabulary to the struct macro, so
one type table serves both.

Three things the host half had to grow so a shader body is ordinary Rust:

- **Components are fields, swizzles are methods.** `v.x` is a real field; `v.xyz()` is a method,
  because Rust has no multi-field access and inventing one would break the compile that is the
  whole point. The vector types became named-field structs to allow it, keeping a function of
  the same name (`f32x4([..])`) so existing host code reads unchanged.
- **A three-vector is twelve bytes.** `f32x3` was `align(16)` and therefore sixteen, which
  matched neither `packed_float3` nor scalar block layout. It is `align(4)` now.
- **`Gpu<T>` derefs and indexes.** Reading `root.width` and writing `root.pixels[i]` are the
  ordinary operators, so a parameter written through is `mut root: GpuMut<'static, T>` exactly
  as Rust requires. The host bodies are never called; they only have to type-check.

## 4. The IR

Flat arenas, `Copy`, const-constructible, no `Box`, indices instead of pointers.

```rust
pub struct GpuFn {
    pub name:   &'static str,
    pub stage:  Stage,                 // Helper | Vertex | Fragment | Compute([u32;3]) | Mesh
    pub params: &'static [Param],      // name, Ty, Binding (Builtin/Location/Root)
    pub ret:    Ty,
    pub locals: &'static [Local],      // name, Ty (Ty::Infer when unannotated)
    pub nodes:  &'static [Node],
    pub body:   u32,                   // the body block; the parser numbers it last
    pub calls:  &'static [&'static GpuFn],
}

pub enum Node {
    Block(&'static [u32]),
    Let { local: u32, value: u32 },
    Store { place: u32, value: u32 },
    If { cond: u32, then: u32, els: u32 },      // u32::MAX for none
    While { cond: u32, body: u32 },
    Loop { body: u32 },
    Range { local: u32, start: u32, end: u32, body: u32 },
    Break, Continue, Return(u32),
    Local(u32), Param(u32), Const(Const), Discard,
    // Component, swizzle or struct member: the base's type decides which, because the macro
    // cannot know a struct's field names.
    Field { base: u32, name: &'static str },
    Index { base: u32, index: u32 },
    Unary(UnOp, u32), Binary(BinOp, u32, u32), Cast { value: u32, to: Ty },
    Construct { ty: Ty, args: &'static [u32] },
    StructLit { ty: Ty, fields: &'static [(&'static str, u32)] },
    Call { func: u32, args: &'static [u32] },
    Intrinsic { op: Intrinsic, args: &'static [u32] },
}
```

Nodes carry no types. One pass, `ir::infer(&GpuFn) -> Types`, fills a parallel array bottom-up:
params and locals seed it, operators propagate (with the usual scalar-broadcast rule for
`vec * scalar`), `Field`/`Index` read through `TypeAbi`. Both backends call it first, so there
is exactly one place that knows what an expression means. Errors here are the only ones that
escape `rustc`, and they are reported at pipeline build with the function and node name.

Unsuffixed literals are `IntLit`/`FloatLit` until they meet a typed operand, and `i32`/`f32` if
they never do. Because the arena is a *tree* — every node has exactly one parent — a literal's
type can be rewritten in place with nothing else to disturb. That approximates Rust's own
inference rather than reproducing it, so both backends write out a conversion wherever the
recorded types of an operation's two sides differ. MSL would convert silently and warn; SPIR-V
would reject the module outright.

`TypeAbi.fields` becomes a `FieldAbi` carrying `Ty` alongside name, offset and size. SPIR-V
takes those byte offsets directly as `OpMemberDecorate Offset`, and MSL gets explicit padding
members generated from the same numbers. Rust layout stays authoritative, but nothing has to be
*made* to match any more; the offsets are stated outright — which is `-fvk-use-scalar-layout`
and `slang.rs`'s `padding()` guesswork both, for free.

## 5. SPIR-V backend

Emit binary words directly. No SSA construction: every local becomes an `OpVariable` in the
`Function` storage class declared in the entry block, reads are `OpLoad`, writes are `OpStore`.
This is what glslang, DXC and rust-gpu all do; the driver's mem2reg cleans it up, and it makes
`if`/`while` need no phi nodes at all.

Module order, one `Vec<u32>` per section then concatenated:

1. Header: magic `0x07230203`, version 1.5, our generator id, id bound, 0.
2. `OpCapability Shader`, `PhysicalStorageBufferAddresses`, `Int64`, plus `Int16`/`Float16`
   on demand; `OpExtension "SPV_KHR_physical_storage_buffer"`.
3. `OpMemoryModel Logical PhysicalStorageBuffer64`.
4. `OpEntryPoint` per `Stage != Helper`, with the interface id list, then `OpExecutionMode`
   (`LocalSize` for compute, `OriginUpperLeft` for fragment).
5. `OpName` for every function, param and local, so RenderDoc reads like the Rust source.
6. Decorations: member `Offset` from `TypeAbi`, `ArrayStride`, `Block`, `BuiltIn`, `Location`.
7. Types and constants, deduped by linear scan over what has been emitted (modules are small;
   a hash map buys nothing).
8. Function bodies.

Structured control flow falls out of the AST, because Rust's `if`/`while`/`loop` are already
structured: `If` emits `OpSelectionMerge` + `OpBranchConditional`, `While`/`Loop`/`Range` emit
the four-block `OpLoopMerge` shape, `break` branches to the merge block, `continue` to the
continue block. Nothing needs to be reconstructed from a CFG.

Two things bite, and both are now handled:

- **Every load and store through a `Gpu<T>` needs the `Aligned` memory operand**, and every
  pointer type used with `OpPtrAccessChain` — which is what `p[i]` on a device pointer becomes,
  the only pointer arithmetic SPIR-V has — needs `ArrayStride`. Without either the module is
  invalid, not merely slow. Both numbers come from the IR's own `size`/`align`.
- **Scalar layout.** `-fvk-use-scalar-layout` disappears with slangc; we get it by construction,
  since we write the `Offset` decorations ourselves from `offset_of!`.

Two more the bring-up turned up:

- **An entry point has no parameters.** Its builtins are `Input` variables, its root pointer is
  `OpConvertUToPtr` of member zero of the push-constant block, and its varyings are `Input` or
  `Output` variables — all loaded at the top of the entry block, so the same `Node::Param` means
  a function parameter in a helper and a loaded global in an entry. No wrapper function.
- **Vulkan renames two builtins.** `VertexIndex`/`InstanceIndex`, not `VertexId`/`InstanceId`;
  and the member a vertex stage writes as `Position` is the one a fragment stage reads as
  `FragCoord`, so the builtin's number depends on the storage class. Drivers accept the wrong
  one; `spirv-val` does not, which is why it runs in the test.

## 6. Metal backend

MSL text, printed by a recursive walk of the same arena. Much less to say: pointers become
`device T*`, `Gpu` vs `GpuMut` becomes `device const T*` vs `device T*`, builtins become
`[[thread_position_in_grid]]` and friends, and the padding members come from the same offsets
SPIR-V decorates with. The existing `METAL_PRELUDE` in `build.rs` survives roughly as-is; the
Slang prelude does not.

Metal keeps the features Slang cannot spell (`icb.metal`, `tile.metal`): those stay hand-written
`.metal` and the `#[gpu]` path is not asked to reach them.

## 7. Where the bytes come from

Both emitters are plain functions, so both scheduling options stay open:

- **Runtime.** `nga::compile(&[&CLEAR])` gives `Shaders::new` its bytes at pipeline creation:
  a SPIR-V module on Vulkan, MSL source on Metal. SPIR-V costs microseconds. Metal costs a
  `newLibraryWithSource`, which is not free.
- **AOT.** `nga::dump(dir, name, &[..])` writes `name.spv` and `name.metal` from any host —
  neither emitter needs a toolchain. A bin target that links the crate its shaders live in
  calls it, and the `build.rs` next door compiles the `.metal` to a `.metallib` exactly as it
  does for the hand-written ones. Emitting the Metal source on a non-Apple host is also what
  lets `tests/gpu.rs` check the Metal-side struct layout from a machine that cannot build for
  Apple at all.

## 8. Phases

| # | Deliverable | Done when | Status |
|---|---|---|---|

| 0 | `ir.rs`, `#[gpu]` parses to `GpuFn`, `infer()` | `tests/gpu.rs` reads a parsed body back | done |
| 1 | MSL backend | `tests/gpu.rs` checks the emitted MSL | done |
| 2 | SPIR-V backend, compute | `examples/sync` simulates and draws; `spirv-val` clean | done |
| 3 | vertex + fragment, stage IO, builtins | `examples/triangle` draws | done |
| 4 | textures, samplers, atomics through bindless handles | `examples/bindless`, `examples/gpu_driven` | done |
| 5 | `#[constants]` -> `OpSpecConstant` / `function_constant`; AOT dump | `examples/raster`, `dump_writes_both_targets` | done |

The evidence column used to name headless tests. Those are gone: the shaders now live in the
example that draws with them, and running an example is what says the pipeline works. What is
left under `nga/tests` is what a window cannot show, which is `gpu.rs` (the IR, both emitters
and the struct layout), `fence.rs`, and the three Apple-gated MSL ones.

### The migration

The `.slang` sources, `slang.rs`, `prelude.slang`, `TypeAbi::decl` and the Slang half of
`#[derive(Constants)]` are deleted. So is the `nga-shaders` crate that briefly held the ported
shaders: each one now lives in the example that draws with it, as a `mod` at the top of the
file, so the struct the host writes and the shader that reads it are forty lines apart in one
file. `sync` is the exception, shared by `examples/sync.rs` and `examples/threaded.rs` through
`examples/shaders/sync.rs` and a `#[path]` mod, since those two run the same simulation on a
different number of threads.

The examples are the testbed. A shader with no example was a shader with only a test to prove
it, and those went with the tests: `fill`, `gradient`, `sample` and `material` are gone.

The only remaining `nga::shader!` callers are the six hand-written MSL shaders that stay that
way (§9), all behind `#[cfg(target_vendor = "apple")]`. So `shader!` lost its Vulkan arm, and
`build.rs` is now fifty lines that copy `shaders/*.metal` into `OUT_DIR` with the prelude in
front.

Two things the validator caught during the port, both now fixed and worth knowing:

- **A laid-out struct cannot live in the `Function` class.** `let v = root.vertices[vid]` needs
  a *second* type id for the same struct, without the `Offset` decorations, and a rebuild of
  the value at the memory boundary. `M::type_of(ty, laid)` and `relayout` are that.
- **Scalar block layout has to be asked for.** `spirv-val` rejects a `f32x3` at offset 12
  without `--scalar-block-layout`, which is the same feature the device already enables.

The whole existing suite still passes unchanged, which is what says the shared-struct half of
the ABI did not move under it.

### Known gaps in the subset

- `match`, and `if`/`else` as an expression. Use `select`.
- An array literal needs its `let` annotation; nothing else can type it.
- `f16` has no host type, so nothing can write an `f16` literal in a shader body. `Prim::F16`
  and the half converter in `spirv.rs` are ready for one and unreachable until Rust's own `f16`
  stabilises or a stand-in is written. Nothing wants half precision yet, so neither has happened.

Recursion is no longer on this list, and the cycle walk in `ir.rs` turns out to be the second
line of defence rather than the first. `calls` is a `const` holding `&OTHER`, so two functions
that call each other are two consts that reference each other, and rustc rejects the pair with
`cycle detected when simplifying constant for the type system`. The macro cannot see a cycle and
never needed to: const evaluation sees it, by name, at the definition.

## 9. Deliberately out

Mesh shaders (phase 3's stage IO generalises to them, but Slang's mesh lowering bug is what made
them hand-written MSL, and we would be inheriting nothing). Subgroup ops. Ray tracing. Any form
of generic or trait-based shader code - monomorphisation in a proc macro is not a thing, and
YAGNI. `match`, until something wants a switch.
