# nga — what's missing

The plan that built this is gone; the design it describes is in the code and in the backend
contract comment at the top of `vulkan.rs` and `metal.rs`. What follows is only the list of
things deliberately left out, and why, so none of it looks like an oversight later.

## Render targets

- **Stencil.** No stencil attachment, no stencil state, no reference value. `Format::D32FloatS8`
  exists and `DepthState` is depth-only. Metal wants a `MTLStencilDescriptor` pair on the
  depth-stencil state — which is built once per (compare, write) pair in `Context::new`, so a
  stencil face would multiply that table.
- **Layered rendering.** `vulkan.rs`'s `Pass` carries a `layers` field internally and nothing
  sets it above 1. Needs `Pass::layers` publicly plus `SV_RenderTargetArrayIndex`.
- **Mip levels.** `TextureDesc` builds single-level textures only, which is why there is no
  `generate_mipmaps` — it would have nothing to generate. Mips first, then the blit chain on
  Vulkan and `generateMipmapsForTexture:` on Metal.

## Backend gaps

- **Vulkan mesh shaders.** `Feature::MeshShaders` answers `false` because `Device::new` does not
  request `VK_EXT_mesh_shader`. The pipeline path, `Shaders::mesh` and `Raster::draw_mesh` all
  exist and return an error; wiring the extension is the whole job.
- **`execute_indirect`.** Metal can take an indirect command buffer's *execution range* from
  GPU memory (`executeCommandsInBuffer:indirectBuffer:`), so the count never round-trips through
  the host. `IndirectCommands` replays a host-known range instead; `gpu_meshes` resets every slot
  each frame so the unclaimed ones replay as nothing, which costs a command per empty slot.
- **`draw_multi_indexed` base vertex on Metal.** Refused rather than ignored:
  `MTL4RenderCommandEncoder`'s indexed draw takes no base vertex, and only the indirect-command
  encoder has the full form. Bake the offset into the indices, or extend `minmetal`.
- **Indexed indirect commands.** `IndirectCommands::draws(.., indexed: true)` builds the buffer,
  but no shader in the tree encodes `draw_indexed_primitives` into one.

## Vocabulary that comes back with the code that needs it

- **`TextureKind`** (1D, 3D, cube, arrays) and **`BlendOp`** were dropped rather than left unused.
  The bindless prelude's array/3D/cube handle types exist so a shared struct can hold one, but
  sampling through them needs the matching declaration in `prelude.slang`.
- **`Format`** is intersection-only and grows on demand: a variant exists only if both
  `vk::Format` and `MTLPixelFormat` have it. Adding one means adding a row to the `match` in each
  backend file, and the compiler points at both.

## Shading language

Three shaders are hand-written MSL under `shaders/*.metal` rather than Slang, and `build.rs`
copies them through on Apple:

- **Tile stage** — slangc answers `unknown capability`. Imageblocks are Apple-only for good, so
  this one is permanent.
- **`render_command`** — encoding an indirect command buffer from a shader has no Slang
  spelling. `__target_intrinsic(metal, ..)` passes an expression through, which is not enough.
- **Mesh stage** — this one is a slangc *bug*, not a language gap, so it should come back. The
  mesh stage lowers to Metal correctly; the fragment does not. slangc emits its `stage_in` as
  flat `[[user(...)]]` varyings, the vertex-pipeline convention, where a mesh pipeline's fragment
  has to take the mesh output structs themselves — Metal rejects the mismatch at pipeline
  creation. Retest on a newer slangc and delete `shaders/mesh.metal` and `shaders/meshlets.metal`
  when it passes.

## Verification

`cargo check -p nga --all-targets --target x86_64-pc-windows-msvc` type-checks the entire Vulkan
half from macOS — `check` never links, so it needs the `rust-std` component and `ash` and no
Windows toolchain. That is what keeps the two backend files in step, and it should be run on
every change to either.

The reverse holds from Windows: `cargo check -p nga --lib --target aarch64-apple-darwin`
type-checks the Metal half without an Apple toolchain, because `check` never links. `--all-targets`
does not work there — `minwin`'s macOS window does not implement the current `PlatformWindow` — so
it is `--lib` only, which covers `metal.rs` and is the file that matters.

It proves a backend half *compiles*, not that it *runs*. The headless tests under `nga/tests` are
what prove the API; the four Apple-gated ones (`mesh`, `icb`, `tile`, and the mesh half of
`queries`) prove nothing on Vulkan by construction.

All nine examples have been run on Vulkan (GTX 1070 Ti, driver 582.66) under
`VK_LAYER_KHRONOS_validation` with zero errors, including resize, minimise and restore. None has
been run on Metal. The three Apple-gated ones print a line and exit on Vulkan.
