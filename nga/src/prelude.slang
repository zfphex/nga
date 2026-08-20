// -- bindless handles ---------------------------------------------------------------------
//
// A handle is eight bytes on both targets, which is what lets one Rust `Handle(u64)` and one
// shared struct layout serve both. How those eight bytes are interpreted is the one place
// the targets genuinely disagree, so it is the one place the prelude forks.
//
// Metal: `DescriptorHandle<T>` is a bare argument-buffer member holding an MTLResourceID.
//
// Vulkan: slangc would happily lower `DescriptorHandle<T>` here too, but it assigns sampled
// and storage textures the same (set 0, binding 2) with different descriptor types, which a
// descriptor-buffer layout cannot declare without VK_EXT_mutable_descriptor_type. Rather than
// take on that requirement, Vulkan keeps three explicit arrays and a `uint2` index, padded to
// the same eight bytes so the struct layout does not move.

#ifdef NGA_METAL

typedef DescriptorHandle<Texture2D<float4> >      nga_texture2d;
typedef DescriptorHandle<RWTexture2D<float4> >    nga_texture2d_rw;
typedef DescriptorHandle<Texture2DArray<float4> > nga_texture2d_array;
typedef DescriptorHandle<Texture3D<float4> >      nga_texture3d;
typedef DescriptorHandle<RWTexture3D<float4> >    nga_texture3d_rw;
typedef DescriptorHandle<TextureCube<float4> >    nga_texturecube;
typedef DescriptorHandle<Texture2D<float> >       nga_depth2d;
typedef DescriptorHandle<SamplerState>            nga_sampler;

float4 nga_sample(nga_texture2d t, nga_sampler s, float2 uv) { return t.Sample(s, uv); }
float4 nga_sample_level(nga_texture2d t, nga_sampler s, float2 uv, float lod) { return t.SampleLevel(s, uv, lod); }
float4 nga_load(nga_texture2d_rw t, uint2 p) { return t[p]; }
void nga_store(nga_texture2d_rw t, uint2 p, float4 v) { t[p] = v; }

#else

// Binding 0 of sets 0, 1 and 2, the three descriptor-buffer layouts the device creates.
// Only the 2D types are declared: aliasing the array/3D/cube views onto binding 0 is legal
// (they are all VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) but slangc warns on every build, so they
// arrive with the code that needs them.
[[vk::binding(0, 0)]] Texture2D<float4>   nga_textures[];
[[vk::binding(0, 1)]] RWTexture2D<float4> nga_storage[];
[[vk::binding(0, 2)]] SamplerState        nga_samplers[];

// The high word is unused. It exists so the handle is eight bytes here too.
struct nga_texture2d       { uint2 slot; };
struct nga_texture2d_rw    { uint2 slot; };
struct nga_texture2d_array { uint2 slot; };
struct nga_texture3d       { uint2 slot; };
struct nga_texture3d_rw    { uint2 slot; };
struct nga_texturecube     { uint2 slot; };
struct nga_depth2d         { uint2 slot; };
struct nga_sampler         { uint2 slot; };
// The array/3D/cube handles exist so a shared struct can hold one; sampling through them
// needs the matching declaration above.

float4 nga_sample(nga_texture2d t, nga_sampler s, float2 uv) { return nga_textures[t.slot.x].Sample(nga_samplers[s.slot.x], uv); }
float4 nga_sample_level(nga_texture2d t, nga_sampler s, float2 uv, float lod) { return nga_textures[t.slot.x].SampleLevel(nga_samplers[s.slot.x], uv, lod); }
float4 nga_load(nga_texture2d_rw t, uint2 p) { return nga_storage[t.slot.x][p]; }
void nga_store(nga_texture2d_rw t, uint2 p, float4 v) { nga_storage[t.slot.x][p] = v; }

#endif

// -- root binding -------------------------------------------------------------------------
//
// One push constant on Vulkan, one argument-table address at index 0 on Metal. slangc lowers
// the same declaration to both, so this needs no fork.

struct nga_push {
    uint64_t root;
    uint resources;
    uint samplers;
};

[[vk::push_constant]] ConstantBuffer<nga_push> push;

// -- indirect arguments -------------------------------------------------------------------

struct nga_draw { uint vertex_count; uint instance_count; uint vertex_start; uint base_instance; };
struct nga_draw_indexed { uint index_count; uint instance_count; uint index_start; int base_vertex; uint base_instance; };
struct nga_dispatch { uint threadgroups[3]; };
struct nga_dispatch_threads { uint threads[3]; uint group[3]; };
