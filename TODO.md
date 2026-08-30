# nga: what's missing

The plan that built this is gone; the design it describes is in the code and in the backend
contract comment at the top of `vulkan.rs` and `metal.rs`. What follows is only the list of
things deliberately left out, and why, so none of it looks like an oversight later.

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
