# nga: what's missing

The plan that built this is gone; the design it describes is in the code and in the backend
contract comment at the top of `vulkan.rs` and `metal.rs`. What follows is only the list of
things deliberately left out, and why, so none of it looks like an oversight later.

## Render targets

- **Stencil.** No stencil attachment, no stencil state, no reference value. `Format::D32FloatS8`
  exists and `DepthState` is depth-only. Metal wants a `MTLStencilDescriptor` pair on the
  depth-stencil state, which is built once per (compare, write) pair in `Context::new`, so a
  stencil face would multiply that table.
- **Layered rendering.** `vulkan.rs`'s `Pass` carries a `layers` field internally and nothing
  sets it above 1. Needs `Pass::layers` publicly plus `SV_RenderTargetArrayIndex`.
- **Mip levels.** `TextureDesc` builds single-level textures only, which is why there is no
  `generate_mipmaps`. It would have nothing to generate. Mips first, then the blit chain on
  Vulkan and `generateMipmapsForTexture:` on Metal.

## Backend gaps

- **Vulkan mesh shaders.** `Feature::MeshShaders` answers `false` because `Device::new` does not
  request `VK_EXT_mesh_shader`. The pipeline path, `Shaders::mesh` and `Raster::draw_mesh` all
  exist and return an error; wiring the extension is the whole job.
- **`execute_indirect`.** Metal can take an indirect command buffer's execution range from
  GPU memory (`executeCommandsInBuffer:indirectBuffer:`), so the count never round-trips through
  the host. `IndirectCommands` replays a host-known range instead; `gpu_meshes` resets every slot
  each frame so the unclaimed ones replay as nothing, which costs a command per empty slot.
- **`draw_multi_indexed` base vertex on Metal.** Refused rather than ignored:
  `MTL4RenderCommandEncoder`'s indexed draw takes no base vertex, and only the indirect-command
  encoder has the full form. Bake the offset into the indices, or extend `minmetal`.
- **Indexed indirect commands.** `IndirectCommands::draws(.., indexed: true)` builds the buffer,
  but no shader in the tree encodes `draw_indexed_primitives` into one.

## Vocabulary that comes back with the code that needs it

- `TextureKind` (1D, 3D, cube, arrays) and `BlendOp` were dropped rather than left unused.
  The array, 3D and cube handle types are in `#[gpu_mod]`'s vocabulary so a shared struct can
  hold one, but sampling through them needs the matching intrinsic in `msl.rs` and `spirv.rs`.
- `Format` is intersection-only and grows on demand: a variant exists only if both
  `vk::Format` and `MTLPixelFormat` have it. Adding one means adding a row to the `match` in each
  backend file, and the compiler points at both.

## Shading language

Every shader is a Rust function in a `#[gpu_mod]` in the example that draws with it, and `nga::compile`
emits SPIR-V or MSL from it.

Six shaders under `shaders/*.metal` stay hand-written, and `build.rs` copies them through on
Apple:

- **Tile stage** (`tile.metal`, `tbdr.metal`). An imageblock is tile-based deferred hardware
  showing through. There is nothing in Vulkan to grow into, so this one is permanent.
- **Encoding a command buffer** (`icb.metal`, `gpu_meshes.metal`). `render_command` is Metal's
  own vocabulary and indirect command buffers are Metal-only, so this one is permanent too.
- **Mesh stage** (`mesh.metal`, `meshlets.metal`). The IR generalises to a mesh stage and the
  emitters would not have far to go. What stops it is that mesh pipelines run on Metal only, so
  the stage would serve one backend. It is worth writing the day Vulkan mesh shaders are wired
  up, and not before.

## Verification

`cargo check -p nga --all-targets --target x86_64-pc-windows-msvc` type-checks the entire Vulkan
half from macOS. `check` never links, so it needs the `rust-std` component and `ash` and no
Windows toolchain. That is what keeps the two backend files in step, and it should be run on
every change to either.

The reverse holds from Windows: `cargo check -p nga --lib --target aarch64-apple-darwin`
type-checks the Metal half without an Apple toolchain, because `check` never links.
`--all-targets` does not work there, since `minwin`'s macOS window does not implement the current
`PlatformWindow`, so it is `--lib` only. That covers `metal.rs`, which is the file that matters.

It proves a backend half compiles, not that it runs. The examples are what prove the API: each
one owns the shader it draws with, so running it exercises the whole path from a shader body to
pixels. `nga/tests` keeps only what a window cannot show, which is `gpu.rs` (the IR, both
emitters and the struct layout), `fence.rs`, and the three Apple-gated MSL ones (`mesh`, `icb`,
`tile`), which prove nothing on Vulkan by construction.

All nine examples have been run on Vulkan (GTX 1070 Ti, driver 582.66) under
`VK_LAYER_KHRONOS_validation` with zero errors, including resize, minimise and restore. None has
been run on Metal.
