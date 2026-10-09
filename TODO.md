# nga: what's missing

Deliberately left out, not oversights. Design is in the code and the backend contract at the top of `vulkan.rs` and `metal.rs`.

## Backend gaps

- **Vulkan mesh shaders.** `Feature::MeshShaders` is false: `Device::new` does not request `VK_EXT_mesh_shader`. Pipeline path, `Shaders::mesh`, and `Raster::draw_mesh` already exist and error. Wiring the extension is the job.
- **`execute_indirect`.** Metal can take an ICB execution range from GPU memory (`executeCommandsInBuffer:indirectBuffer:`). `IndirectCommands` replays a host-known range instead. `gpu_meshes` resets every slot so empty ones replay as nothing, one command per empty slot.
- **Indexed indirect commands.** `IndirectCommands::draws(.., indexed: true)` builds the buffer. No shader in the tree encodes `draw_indexed_primitives`.

## Vocabulary that comes back with the code that needs it

- `TextureKind` (1D, 3D, cube, arrays) and `BlendOp` were dropped unused. Array/3D/cube handle types exist in `#[gpu_mod]` so a shared struct can hold one; sampling needs the matching intrinsic in `msl.rs` and `spirv.rs`.
- `Format` is intersection-only: a variant exists only if both `vk::Format` and `MTLPixelFormat` have it. Adding one is a row in each backend `match`.
