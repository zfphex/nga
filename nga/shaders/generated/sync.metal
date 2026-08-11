
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

struct Instance {
    float2 center;
    float angle;
    float scale;
    float4 color;
};

struct Vertex {
    packed_float3 position;
    packed_float3 color;
};

struct SimData {
    device Instance* instances;
    float time;
    uint count;
};

struct BuildData {
    device const Instance* instances;
    device Vertex* vertices;
    uint count;
    uint pad;
};

struct alignas(16) DrawData {
    device const Vertex* vertices;
    float4x4 mvp;
};

static_assert(sizeof(Instance) == 32, "Instance has the wrong size");
static_assert(alignof(Instance) == 16, "Instance has the wrong alignment");
static_assert(__builtin_offsetof(Instance, center) == 0, "Instance::center is at the wrong offset");
static_assert(__builtin_offsetof(Instance, angle) == 8, "Instance::angle is at the wrong offset");
static_assert(__builtin_offsetof(Instance, scale) == 12, "Instance::scale is at the wrong offset");
static_assert(__builtin_offsetof(Instance, color) == 16, "Instance::color is at the wrong offset");

static_assert(sizeof(Vertex) == 24, "Vertex has the wrong size");
static_assert(alignof(Vertex) == 4, "Vertex has the wrong alignment");
static_assert(__builtin_offsetof(Vertex, position) == 0, "Vertex::position is at the wrong offset");
static_assert(__builtin_offsetof(Vertex, color) == 12, "Vertex::color is at the wrong offset");

static_assert(sizeof(SimData) == 16, "SimData has the wrong size");
static_assert(alignof(SimData) == 8, "SimData has the wrong alignment");
static_assert(__builtin_offsetof(SimData, instances) == 0, "SimData::instances is at the wrong offset");
static_assert(__builtin_offsetof(SimData, time) == 8, "SimData::time is at the wrong offset");
static_assert(__builtin_offsetof(SimData, count) == 12, "SimData::count is at the wrong offset");

static_assert(sizeof(BuildData) == 24, "BuildData has the wrong size");
static_assert(alignof(BuildData) == 8, "BuildData has the wrong alignment");
static_assert(__builtin_offsetof(BuildData, instances) == 0, "BuildData::instances is at the wrong offset");
static_assert(__builtin_offsetof(BuildData, vertices) == 8, "BuildData::vertices is at the wrong offset");
static_assert(__builtin_offsetof(BuildData, count) == 16, "BuildData::count is at the wrong offset");
static_assert(__builtin_offsetof(BuildData, pad) == 20, "BuildData::pad is at the wrong offset");

static_assert(sizeof(DrawData) == 80, "DrawData has the wrong size");
static_assert(alignof(DrawData) == 16, "DrawData has the wrong alignment");
static_assert(__builtin_offsetof(DrawData, vertices) == 0, "DrawData::vertices is at the wrong offset");
static_assert(__builtin_offsetof(DrawData, mvp) == 16, "DrawData::mvp is at the wrong offset");

kernel void simulate(device const SimData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    float f = float(i) / float(data->count);
    float orbit = 0.25 + 0.65 * fract(f * 7.0);
    float phase = f * 6.2831853 + data->time * (0.2 + 0.9 * fract(f * 5.0));

    Instance inst;
    inst.center = float2(cos(phase), sin(phase)) * orbit;
    inst.angle = phase * 2.0 + data->time;
    inst.scale = 0.015 + 0.045 * fract(f * 3.0);
    inst.color = float4(0.5 + 0.5 * cos(6.2831853 * f + float3(0.0, 2.0, 4.0)), 1.0);
    data->instances[i] = inst;
}

kernel void build_vertices(device const BuildData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    Instance inst = data->instances[i];
    device Vertex* out = data->vertices + i * 3;
    for (uint k = 0; k < 3; ++k) {
        float a = inst.angle + float(k) * 2.0943951;
        float2 p = inst.center + float2(cos(a), sin(a)) * inst.scale;
        out[k].position = float3(p, 0.5);
        out[k].color = inst.color.rgb;
    }
}

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut vertex_main(device const DrawData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 1.0);
    out.color = v.color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
