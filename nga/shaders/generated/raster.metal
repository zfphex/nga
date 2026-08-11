
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
    packed_float3 position;
    packed_float3 color;
};

struct Data {
    device const Vertex* vertices;
    float4x4 mvp;
    float4 tint;
};

static_assert(sizeof(Vertex) == 24, "Vertex has the wrong size");
static_assert(alignof(Vertex) == 4, "Vertex has the wrong alignment");
static_assert(__builtin_offsetof(Vertex, position) == 0, "Vertex::position is at the wrong offset");
static_assert(__builtin_offsetof(Vertex, color) == 12, "Vertex::color is at the wrong offset");

static_assert(sizeof(Data) == 96, "Data has the wrong size");
static_assert(alignof(Data) == 16, "Data has the wrong alignment");
static_assert(__builtin_offsetof(Data, vertices) == 0, "Data::vertices is at the wrong offset");
static_assert(__builtin_offsetof(Data, mvp) == 16, "Data::mvp is at the wrong offset");
static_assert(__builtin_offsetof(Data, tint) == 80, "Data::tint is at the wrong offset");
constant bool boost [[function_constant(0)]];
constant float gamma [[function_constant(1)]];

struct VertexOut {
    float4 position [[position]];
    float3 color;
    float alpha;
};

vertex VertexOut vertex_main(device const Data* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 1.0);
    out.color = v.color * data->tint.rgb;
    out.alpha = data->tint.a;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    float3 c = pow(in.color, float3(gamma));
    if (boost) {
        c = min(c * 1.6, 1.0);
    }
    return float4(c, in.alpha);
}

fragment float4 fragment_screen(VertexOut in [[stage_in]], float4 dst [[color(0)]]) {
    float3 src = in.color * in.alpha;
    return float4(1.0 - (1.0 - dst.rgb) * (1.0 - src), 1.0);
}
