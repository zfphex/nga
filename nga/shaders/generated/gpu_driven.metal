
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

struct SimData {
    device Instance* instances;
    float time;
    uint count;
};

struct alignas(16) DrawData {
    device const Instance* instance;
    float4x4 mvp;
};

struct PrepareData {
    device nga_dispatch* encode;
    device nga_draw* ring;
    uint count;
    uint group;
    uint segments;
    uint pad;
};

struct alignas(16) BoundaryData {
    float4x4 mvp;
    float radius;
    uint segments;
};

struct alignas(16) EncodeData {
    nga_commands commands;
    device const Instance* instances;
    device DrawData* draws;
    device nga_exec_range* range;
    float4x4 mvp;
    float cull_radius;
    uint count;
};

static_assert(sizeof(Instance) == 32, "Instance has the wrong size");
static_assert(alignof(Instance) == 16, "Instance has the wrong alignment");
static_assert(__builtin_offsetof(Instance, center) == 0, "Instance::center is at the wrong offset");
static_assert(__builtin_offsetof(Instance, angle) == 8, "Instance::angle is at the wrong offset");
static_assert(__builtin_offsetof(Instance, scale) == 12, "Instance::scale is at the wrong offset");
static_assert(__builtin_offsetof(Instance, color) == 16, "Instance::color is at the wrong offset");

static_assert(sizeof(SimData) == 16, "SimData has the wrong size");
static_assert(alignof(SimData) == 8, "SimData has the wrong alignment");
static_assert(__builtin_offsetof(SimData, instances) == 0, "SimData::instances is at the wrong offset");
static_assert(__builtin_offsetof(SimData, time) == 8, "SimData::time is at the wrong offset");
static_assert(__builtin_offsetof(SimData, count) == 12, "SimData::count is at the wrong offset");

static_assert(sizeof(DrawData) == 80, "DrawData has the wrong size");
static_assert(alignof(DrawData) == 16, "DrawData has the wrong alignment");
static_assert(__builtin_offsetof(DrawData, instance) == 0, "DrawData::instance is at the wrong offset");
static_assert(__builtin_offsetof(DrawData, mvp) == 16, "DrawData::mvp is at the wrong offset");

static_assert(sizeof(PrepareData) == 32, "PrepareData has the wrong size");
static_assert(alignof(PrepareData) == 8, "PrepareData has the wrong alignment");
static_assert(__builtin_offsetof(PrepareData, encode) == 0, "PrepareData::encode is at the wrong offset");
static_assert(__builtin_offsetof(PrepareData, ring) == 8, "PrepareData::ring is at the wrong offset");
static_assert(__builtin_offsetof(PrepareData, count) == 16, "PrepareData::count is at the wrong offset");
static_assert(__builtin_offsetof(PrepareData, group) == 20, "PrepareData::group is at the wrong offset");
static_assert(__builtin_offsetof(PrepareData, segments) == 24, "PrepareData::segments is at the wrong offset");
static_assert(__builtin_offsetof(PrepareData, pad) == 28, "PrepareData::pad is at the wrong offset");

static_assert(sizeof(BoundaryData) == 80, "BoundaryData has the wrong size");
static_assert(alignof(BoundaryData) == 16, "BoundaryData has the wrong alignment");
static_assert(__builtin_offsetof(BoundaryData, mvp) == 0, "BoundaryData::mvp is at the wrong offset");
static_assert(__builtin_offsetof(BoundaryData, radius) == 64, "BoundaryData::radius is at the wrong offset");
static_assert(__builtin_offsetof(BoundaryData, segments) == 68, "BoundaryData::segments is at the wrong offset");

static_assert(sizeof(EncodeData) == 112, "EncodeData has the wrong size");
static_assert(alignof(EncodeData) == 16, "EncodeData has the wrong alignment");
static_assert(__builtin_offsetof(EncodeData, commands) == 0, "EncodeData::commands is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, instances) == 8, "EncodeData::instances is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, draws) == 16, "EncodeData::draws is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, range) == 24, "EncodeData::range is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, mvp) == 32, "EncodeData::mvp is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, cull_radius) == 96, "EncodeData::cull_radius is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, count) == 100, "EncodeData::count is at the wrong offset");

kernel void prepare(device const PrepareData* data [[buffer(0)]]) {
    data->encode->threadgroups[0] = (data->count + data->group - 1) / data->group;
    data->encode->threadgroups[1] = 1;
    data->encode->threadgroups[2] = 1;

    data->ring->vertex_count = data->segments * 2 + 2;
    data->ring->instance_count = 1;
    data->ring->vertex_start = 0;
    data->ring->base_instance = 0;
}

kernel void simulate(device const SimData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    uint side = 64;
    float2 cell = (float2(i % side, i / side) + 0.5) / float(side) * 2.0 - 1.0;
    float n = fract(sin(float(i) * 12.9898) * 43758.5453);
    float phase = data->time * (0.4 + n) + n * 6.2831853;

    Instance inst;
    inst.center = cell + float2(cos(phase), sin(phase)) * 0.008;
    inst.angle = phase;
    inst.scale = 0.009 + 0.004 * n;
    inst.color = float4(0.5 + 0.5 * cos(6.2831853 * n + float3(0.0, 2.0, 4.0)), 1.0);
    data->instances[i] = inst;
}

kernel void encode(device const EncodeData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    device const Instance* inst = data->instances + i;
    if (length(inst->center) > data->cull_radius) {
        return;
    }

    uint slot = atomic_fetch_add_explicit(&data->range->length, 1, memory_order_relaxed);
    device DrawData* draw = data->draws + slot;
    draw->instance = inst;
    draw->mvp = data->mvp;

    render_command cmd(data->commands.cb, slot);
    cmd.set_vertex_buffer(draw, 0);
    cmd.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);
}

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut vertex_main(device const DrawData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Instance inst = *data->instance;
    float a = inst.angle + float(vid) * 2.0943951;
    float2 p = inst.center + float2(cos(a), sin(a)) * inst.scale;

    VertexOut out;
    out.position = data->mvp * float4(p, 0.5, 1.0);
    out.color = inst.color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}

vertex float4 boundary_main(device const BoundaryData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    float a = float(vid / 2) / float(data->segments) * 6.2831853;
    float r = data->radius + (vid % 2 == 0 ? -0.004 : 0.004);
    return data->mvp * float4(cos(a) * r, sin(a) * r, 0.25, 1.0);
}

fragment float4 boundary_fragment() {
    return float4(1.0, 1.0, 1.0, 0.6);
}
