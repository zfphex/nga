
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
    float3 position;
};

struct Mesh {
    uint tri_start;
    uint tri_count;
    uint line_start;
    uint line_count;
    uint base_vertex;
};

struct alignas(16) Object {
    float4 color;
    float3 cell;
    float scale;
    float orbit;
    float phase;
    uint mesh;
    uint style;
};

struct alignas(16) DrawData {
    device const Vertex* vertices;
    float4x4 mvp;
    float4x4 model;
    float4 color;
};

struct alignas(16) EncodeData {
    nga_commands commands;
    device const nga_pipeline* pipelines;
    device const Object* objects;
    device const Mesh* meshes;
    device const Vertex* vertices;
    device const uint* indices;
    device DrawData* draws;
    device nga_exec_range* range;
    float4x4 viewproj;
    float time;
    float cull_radius;
    uint count;
    uint pad;
};

static_assert(sizeof(Vertex) == 16, "Vertex has the wrong size");
static_assert(alignof(Vertex) == 16, "Vertex has the wrong alignment");
static_assert(__builtin_offsetof(Vertex, position) == 0, "Vertex::position is at the wrong offset");

static_assert(sizeof(Mesh) == 20, "Mesh has the wrong size");
static_assert(alignof(Mesh) == 4, "Mesh has the wrong alignment");
static_assert(__builtin_offsetof(Mesh, tri_start) == 0, "Mesh::tri_start is at the wrong offset");
static_assert(__builtin_offsetof(Mesh, tri_count) == 4, "Mesh::tri_count is at the wrong offset");
static_assert(__builtin_offsetof(Mesh, line_start) == 8, "Mesh::line_start is at the wrong offset");
static_assert(__builtin_offsetof(Mesh, line_count) == 12, "Mesh::line_count is at the wrong offset");
static_assert(__builtin_offsetof(Mesh, base_vertex) == 16, "Mesh::base_vertex is at the wrong offset");

static_assert(sizeof(Object) == 64, "Object has the wrong size");
static_assert(alignof(Object) == 16, "Object has the wrong alignment");
static_assert(__builtin_offsetof(Object, color) == 0, "Object::color is at the wrong offset");
static_assert(__builtin_offsetof(Object, cell) == 16, "Object::cell is at the wrong offset");
static_assert(__builtin_offsetof(Object, scale) == 32, "Object::scale is at the wrong offset");
static_assert(__builtin_offsetof(Object, orbit) == 36, "Object::orbit is at the wrong offset");
static_assert(__builtin_offsetof(Object, phase) == 40, "Object::phase is at the wrong offset");
static_assert(__builtin_offsetof(Object, mesh) == 44, "Object::mesh is at the wrong offset");
static_assert(__builtin_offsetof(Object, style) == 48, "Object::style is at the wrong offset");

static_assert(sizeof(DrawData) == 160, "DrawData has the wrong size");
static_assert(alignof(DrawData) == 16, "DrawData has the wrong alignment");
static_assert(__builtin_offsetof(DrawData, vertices) == 0, "DrawData::vertices is at the wrong offset");
static_assert(__builtin_offsetof(DrawData, mvp) == 16, "DrawData::mvp is at the wrong offset");
static_assert(__builtin_offsetof(DrawData, model) == 80, "DrawData::model is at the wrong offset");
static_assert(__builtin_offsetof(DrawData, color) == 144, "DrawData::color is at the wrong offset");

static_assert(sizeof(EncodeData) == 144, "EncodeData has the wrong size");
static_assert(alignof(EncodeData) == 16, "EncodeData has the wrong alignment");
static_assert(__builtin_offsetof(EncodeData, commands) == 0, "EncodeData::commands is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, pipelines) == 8, "EncodeData::pipelines is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, objects) == 16, "EncodeData::objects is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, meshes) == 24, "EncodeData::meshes is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, vertices) == 32, "EncodeData::vertices is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, indices) == 40, "EncodeData::indices is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, draws) == 48, "EncodeData::draws is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, range) == 56, "EncodeData::range is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, viewproj) == 64, "EncodeData::viewproj is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, time) == 128, "EncodeData::time is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, cull_radius) == 132, "EncodeData::cull_radius is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, count) == 136, "EncodeData::count is at the wrong offset");
static_assert(__builtin_offsetof(EncodeData, pad) == 140, "EncodeData::pad is at the wrong offset");

kernel void encode(device const EncodeData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    device const Object* obj = data->objects + i;
    float t = data->time * obj->orbit + obj->phase;
    float3 center = obj->cell + float3(cos(t), sin(t * 1.3), sin(t)) * 0.12;
    if (length(center) > data->cull_radius) {
        return;
    }

    device const Mesh* mesh = data->meshes + obj->mesh;
    uint slot = atomic_fetch_add_explicit(&data->range->length, 1, memory_order_relaxed);

    float c = cos(t * 1.7);
    float s = sin(t * 1.7);
    float k = obj->scale;
    float4x4 model = float4x4(
        float4(c * k, 0.0, -s * k, 0.0),
        float4(0.0, k, 0.0, 0.0),
        float4(s * k, 0.0, c * k, 0.0),
        float4(center, 1.0));

    device DrawData* draw = data->draws + slot;
    draw->vertices = data->vertices;
    draw->mvp = data->viewproj * model;
    draw->model = model;
    draw->color = obj->color;

    bool wire = obj->style != 0;
    uint start = wire ? mesh->line_start : mesh->tri_start;
    uint count = wire ? mesh->line_count : mesh->tri_count;

    render_command cmd(data->commands.cb, slot);
    cmd.set_render_pipeline_state(data->pipelines[obj->style].p);
    cmd.set_vertex_buffer(draw, 0);
    cmd.draw_indexed_primitives(
        wire ? primitive_type::line : primitive_type::triangle,
        count,
        data->indices + start,
        1,
        mesh->base_vertex,
        0);
}

struct VertexOut {
    float4 position [[position]];
    float3 world;
    float4 color;
};

vertex VertexOut solid_vertex(device const DrawData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    float3 p = data->vertices[vid].position;
    VertexOut out;
    out.position = data->mvp * float4(p, 1.0);
    out.world = (data->model * float4(p, 1.0)).xyz;
    out.color = data->color;
    return out;
}

fragment float4 solid_fragment(VertexOut in [[stage_in]]) {
    float3 n = normalize(cross(dfdx(in.world), dfdy(in.world)));
    float light = 0.2 + 0.8 * abs(dot(n, normalize(float3(0.4, 0.7, 0.55))));
    return float4(in.color.rgb * light, 1.0);
}

vertex VertexOut wire_vertex(device const DrawData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    float3 p = data->vertices[vid].position * 1.08;
    VertexOut out;
    out.position = data->mvp * float4(p, 1.0);
    out.world = p;
    out.color = data->color;
    return out;
}

fragment float4 wire_fragment(VertexOut in [[stage_in]]) {
    return float4(in.color.rgb * 1.4, 0.85);
}
