#include <metal_stdlib>
using namespace metal;

// The root binding: one argument-table address at index 0. Every entry point takes it, and
// `#[gpu]`'s root parameter is a cast of `root`.
struct nga_push {
    ulong root;
    uint resources;
    uint samplers;
};

// A bindless handle is one MTLResourceID either way, which is what lets a Rust `Handle(u64)`
// be written straight into a shared struct. On Metal that is a bare argument-buffer member.
struct nga_texture2d { texture2d<float> t; };
struct nga_texture2d_rw { texture2d<float, access::read_write> t; };
struct nga_texture2d_array { texture2d_array<float> t; };
struct nga_texture3d { texture3d<float> t; };
struct nga_texture3d_rw { texture3d<float, access::read_write> t; };
struct nga_texturecube { texturecube<float> t; };
struct nga_depth2d { depth2d<float> t; };
struct nga_sampler { sampler s; };
struct nga_opaque { ulong bits; };

static_assert(sizeof(nga_texture2d) == 8, "nga_texture2d is not one resource id");
static_assert(sizeof(nga_sampler) == 8, "nga_sampler is not one resource id");
