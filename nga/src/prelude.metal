#include <metal_stdlib>
using namespace metal;

// The root binding: one argument-table address at index 0. Every entry point takes it, and
// `#[gpu]`'s root parameter is a cast of `root`.
struct nga_push {
    ulong root;
    ulong resources;
    uint resources_stride;
    uint samplers;
};

template<typename T>
struct nga_desc { T t; };

// A bindless handle is a 64-bit value: a 32-bit slot index or direct MTLResourceID.
struct nga_texture2d { ulong slot; };
struct nga_texture2d_rw { ulong slot; };
struct nga_texture2d_array { ulong slot; };
struct nga_texture3d { ulong slot; };
struct nga_texture3d_rw { ulong slot; };
struct nga_texturecube { ulong slot; };
struct nga_depth2d { ulong slot; };
struct nga_sampler { sampler s; };
struct nga_opaque { ulong bits; };

static_assert(sizeof(nga_texture2d) == 8, "nga_texture2d is not 8 bytes");
static_assert(sizeof(nga_texture2d_rw) == 8, "nga_texture2d_rw is not 8 bytes");
static_assert(sizeof(nga_sampler) == 8, "nga_sampler is not one resource id");

template<typename H>
inline H nga_offset(H h, uint off) {
    return H { h.slot + (ulong)off };
}

template<typename T, typename H>
inline T nga_deref(constant nga_push* push, H h) {
    ulong addr = push->resources_stride != 0 ? (push->resources + (ulong)h.slot * push->resources_stride) : h.slot;
    return (*(thread const nga_desc<T>*)&addr).t;
}

inline texture2d<float> nga_deref_tex2d(constant nga_push* p, nga_texture2d h) { return nga_deref<texture2d<float>>(p, h); }
inline texture2d<float, access::read_write> nga_deref_tex2d_rw(constant nga_push* p, nga_texture2d_rw h) { return nga_deref<texture2d<float, access::read_write>>(p, h); }
inline texture2d_array<float> nga_deref_tex2d_array(constant nga_push* p, nga_texture2d_array h) { return nga_deref<texture2d_array<float>>(p, h); }
inline texture3d<float> nga_deref_tex3d(constant nga_push* p, nga_texture3d h) { return nga_deref<texture3d<float>>(p, h); }
inline texture3d<float, access::read_write> nga_deref_tex3d_rw(constant nga_push* p, nga_texture3d_rw h) { return nga_deref<texture3d<float, access::read_write>>(p, h); }
inline texturecube<float> nga_deref_texcube(constant nga_push* p, nga_texturecube h) { return nga_deref<texturecube<float>>(p, h); }
inline depth2d<float> nga_deref_depth2d(constant nga_push* p, nga_depth2d h) { return nga_deref<depth2d<float>>(p, h); }
