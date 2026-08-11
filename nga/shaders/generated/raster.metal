
#include <metal_stdlib>
using namespace metal;

[[maybe_unused]] constexpr sampler nga_nearest_clamp(filter::nearest, address::clamp_to_edge);
[[maybe_unused]] constexpr sampler nga_linear_clamp(filter::linear, mip_filter::linear, address::clamp_to_edge);
[[maybe_unused]] constexpr sampler nga_linear_repeat(filter::linear, mip_filter::linear, address::repeat);
[[maybe_unused]] constexpr sampler nga_aniso_repeat(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(16));

struct nga_texture2d { texture2d<float> t; };
struct nga_texture2d_array { texture2d_array<float> t; };
struct nga_texturecube { texturecube<float> t; };
struct nga_sampler { sampler s; };

static_assert(sizeof(nga_texture2d) == 8, "nga_texture2d is not one resource id");
static_assert(sizeof(nga_texture2d_array) == 8, "nga_texture2d_array is not one resource id");
static_assert(sizeof(nga_texturecube) == 8, "nga_texturecube is not one resource id");
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

struct VertexData {
    device const Vertex* vertices;
    float4x4 mvp;
};

struct PixelData {
    float4 tint;
};

static_assert(sizeof(Vertex) == 24, "Vertex has the wrong size");
static_assert(alignof(Vertex) == 4, "Vertex has the wrong alignment");
static_assert(__builtin_offsetof(Vertex, position) == 0, "Vertex::position is at the wrong offset");
static_assert(__builtin_offsetof(Vertex, color) == 12, "Vertex::color is at the wrong offset");

static_assert(sizeof(VertexData) == 80, "VertexData has the wrong size");
static_assert(alignof(VertexData) == 16, "VertexData has the wrong alignment");
static_assert(__builtin_offsetof(VertexData, vertices) == 0, "VertexData::vertices is at the wrong offset");
static_assert(__builtin_offsetof(VertexData, mvp) == 16, "VertexData::mvp is at the wrong offset");

static_assert(sizeof(PixelData) == 16, "PixelData has the wrong size");
static_assert(alignof(PixelData) == 16, "PixelData has the wrong alignment");
static_assert(__builtin_offsetof(PixelData, tint) == 0, "PixelData::tint is at the wrong offset");
constant bool boost [[function_constant(0)]];
constant float gamma [[function_constant(1)]];

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut vertex_main(device const VertexData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 1.0);
    out.color = v.color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]], device const PixelData* data [[buffer(0)]]) {
    float3 c = pow(in.color * data->tint.rgb, float3(gamma));
    if (boost) {
        c = min(c * 1.6, 1.0);
    }
    return float4(c, data->tint.a);
}

fragment float4 fragment_screen(VertexOut in [[stage_in]], device const PixelData* data [[buffer(0)]], float4 dst [[color(0)]]) {
    float3 src = in.color * data->tint.rgb * data->tint.a;
    return float4(1.0 - (1.0 - dst.rgb) * (1.0 - src), 1.0);
}
