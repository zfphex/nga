
#include <metal_stdlib>
using namespace metal;

[[maybe_unused]] constexpr sampler nga_nearest_clamp(filter::nearest, address::clamp_to_edge);
[[maybe_unused]] constexpr sampler nga_linear_clamp(filter::linear, mip_filter::linear, address::clamp_to_edge);
[[maybe_unused]] constexpr sampler nga_linear_repeat(filter::linear, mip_filter::linear, address::repeat);
[[maybe_unused]] constexpr sampler nga_aniso_repeat(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(16));
[[maybe_unused]] constexpr sampler nga_shadow(filter::linear, address::clamp_to_edge, compare_func::less);

#define nga_heap(type, name) device const type* name [[buffer(1)]]
#define nga_heap_as(type, heap) ((device const type*)(heap))

struct nga_texture2d { texture2d<float> t; };
struct nga_texture2d_w { texture2d<float, access::write> t; };
struct nga_texture2d_rw { texture2d<float, access::read_write> t; };
struct nga_texture2d_array { texture2d_array<float> t; };
struct nga_texture3d { texture3d<float> t; };
struct nga_texture3d_w { texture3d<float, access::write> t; };
struct nga_texture3d_rw { texture3d<float, access::read_write> t; };
struct nga_texturecube { texturecube<float> t; };
struct nga_depth2d { depth2d<float> t; };
struct nga_sampler { sampler s; };

static_assert(sizeof(nga_texture2d) == 8, "nga_texture2d is not one resource id");
static_assert(sizeof(nga_texture2d_w) == 8, "nga_texture2d_w is not one resource id");
static_assert(sizeof(nga_texture2d_rw) == 8, "nga_texture2d_rw is not one resource id");
static_assert(sizeof(nga_texture2d_array) == 8, "nga_texture2d_array is not one resource id");
static_assert(sizeof(nga_texture3d) == 8, "nga_texture3d is not one resource id");
static_assert(sizeof(nga_texture3d_w) == 8, "nga_texture3d_w is not one resource id");
static_assert(sizeof(nga_texture3d_rw) == 8, "nga_texture3d_rw is not one resource id");
static_assert(sizeof(nga_texturecube) == 8, "nga_texturecube is not one resource id");
static_assert(sizeof(nga_depth2d) == 8, "nga_depth2d is not one resource id");
static_assert(sizeof(nga_sampler) == 8, "nga_sampler is not one resource id");

struct nga_commands { command_buffer cb; };
struct nga_pipeline { render_pipeline_state p; };
struct nga_exec_range { uint location; atomic_uint length; };
struct nga_draw { uint vertex_count; uint instance_count; uint vertex_start; uint base_instance; };
struct nga_draw_indexed { uint index_count; uint instance_count; uint index_start; int base_vertex; uint base_instance; };
struct nga_dispatch { uint threadgroups[3]; };
struct nga_dispatch_threads { uint threads[3]; uint group[3]; };

static_assert(sizeof(nga_commands) == 8, "nga_commands is not one resource id");
static_assert(sizeof(nga_pipeline) == 8, "nga_pipeline is not one resource id");
static_assert(sizeof(nga_exec_range) == 8, "nga_exec_range does not match MTLIndirectCommandBufferExecutionRange");

struct Vertex {
    packed_float2 position;
    packed_float2 uv;
};

struct Material {
    uint texture_base;
    uint sampler;
    packed_float3 tint;
};

struct Data {
    device const Vertex* vertices;
    device const Material* materials;
    float4x4 mvp;
};

struct Paint {
    uint target;
    uint size;
    float time;
};

static_assert(sizeof(Vertex) == 16, "Vertex has the wrong size");
static_assert(alignof(Vertex) == 4, "Vertex has the wrong alignment");
static_assert(__builtin_offsetof(Vertex, position) == 0, "Vertex::position is at the wrong offset");
static_assert(__builtin_offsetof(Vertex, uv) == 8, "Vertex::uv is at the wrong offset");

static_assert(sizeof(Material) == 20, "Material has the wrong size");
static_assert(alignof(Material) == 4, "Material has the wrong alignment");
static_assert(__builtin_offsetof(Material, texture_base) == 0, "Material::texture_base is at the wrong offset");
static_assert(__builtin_offsetof(Material, sampler) == 4, "Material::sampler is at the wrong offset");
static_assert(__builtin_offsetof(Material, tint) == 8, "Material::tint is at the wrong offset");

static_assert(sizeof(Data) == 80, "Data has the wrong size");
static_assert(alignof(Data) == 16, "Data has the wrong alignment");
static_assert(__builtin_offsetof(Data, vertices) == 0, "Data::vertices is at the wrong offset");
static_assert(__builtin_offsetof(Data, materials) == 8, "Data::materials is at the wrong offset");
static_assert(__builtin_offsetof(Data, mvp) == 16, "Data::mvp is at the wrong offset");

static_assert(sizeof(Paint) == 12, "Paint has the wrong size");
static_assert(alignof(Paint) == 4, "Paint has the wrong alignment");
static_assert(__builtin_offsetof(Paint, target) == 0, "Paint::target is at the wrong offset");
static_assert(__builtin_offsetof(Paint, size) == 4, "Paint::size is at the wrong offset");
static_assert(__builtin_offsetof(Paint, time) == 8, "Paint::time is at the wrong offset");

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    uint material [[flat]];
};

kernel void paint(
    device const Paint* data [[buffer(0)]],
    nga_heap(nga_texture2d_rw, heap),
    uint2 tid [[thread_position_in_grid]]
) {
    if (tid.x >= data->size || tid.y >= data->size) {
        return;
    }
    float2 p = (float2(tid) + 0.5) / float(data->size) * 2.0 - 1.0;
    float r = length(p);
    float wave = sin(r * 18.0 - data->time * 6.0) * 0.5 + 0.5;
    texture2d<float, access::read_write> dst = heap[data->target].t;
    float4 prev = dst.read(tid);
    float4 lit = float4(wave, wave * (1.0 - r), prev.b * 0.9 + 0.1 * r, 1.0);
    dst.write(lit, tid);
}

vertex VertexOut vertex_main(device const Data* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 0.0, 1.0);
    out.uv = v.uv;
    out.material = vid / 6;
    return out;
}

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    device const Data* data [[buffer(0)]],
    nga_heap(nga_texture2d, heap)
) {
    Material m = data->materials[in.material];
    uint layer = min(uint(in.uv.x * 3.0), 2u);
    sampler s = nga_heap_as(nga_sampler, heap)[m.sampler].s;
    float3 c = heap[m.texture_base + layer].t.sample(s, in.uv).rgb;
    return float4(c * m.tint, 1.0);
}
