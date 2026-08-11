
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

struct alignas(16) Meshlet {
    float4 bounds;
    float4 cone;
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
};

struct Model {
    uint meshlet_start;
    uint meshlet_count;
};

struct alignas(16) Object {
    float4 color;
    float3 cell;
    float scale;
    float orbit;
    float phase;
    uint model;
};

struct alignas(16) Payload {
    float4x4 mvp;
    float4x4 model;
    float4 color;
    uint meshlets[32];
};

struct alignas(16) Scene {
    device const Object* objects;
    device const Model* models;
    device const Meshlet* meshlets;
    device const uint* meshlet_vertices;
    device const uint* meshlet_triangles;
    device const Vertex* vertices;
    float4x4 viewproj;
    float4 planes[6];
    float3 camera;
    float time;
    float debug;
    uint count;
    uint pad;
};

static_assert(sizeof(Vertex) == 16, "Vertex has the wrong size");
static_assert(alignof(Vertex) == 16, "Vertex has the wrong alignment");
static_assert(__builtin_offsetof(Vertex, position) == 0, "Vertex::position is at the wrong offset");

static_assert(sizeof(Meshlet) == 48, "Meshlet has the wrong size");
static_assert(alignof(Meshlet) == 16, "Meshlet has the wrong alignment");
static_assert(__builtin_offsetof(Meshlet, bounds) == 0, "Meshlet::bounds is at the wrong offset");
static_assert(__builtin_offsetof(Meshlet, cone) == 16, "Meshlet::cone is at the wrong offset");
static_assert(__builtin_offsetof(Meshlet, vertex_offset) == 32, "Meshlet::vertex_offset is at the wrong offset");
static_assert(__builtin_offsetof(Meshlet, triangle_offset) == 36, "Meshlet::triangle_offset is at the wrong offset");
static_assert(__builtin_offsetof(Meshlet, vertex_count) == 40, "Meshlet::vertex_count is at the wrong offset");
static_assert(__builtin_offsetof(Meshlet, triangle_count) == 44, "Meshlet::triangle_count is at the wrong offset");

static_assert(sizeof(Model) == 8, "Model has the wrong size");
static_assert(alignof(Model) == 4, "Model has the wrong alignment");
static_assert(__builtin_offsetof(Model, meshlet_start) == 0, "Model::meshlet_start is at the wrong offset");
static_assert(__builtin_offsetof(Model, meshlet_count) == 4, "Model::meshlet_count is at the wrong offset");

static_assert(sizeof(Object) == 48, "Object has the wrong size");
static_assert(alignof(Object) == 16, "Object has the wrong alignment");
static_assert(__builtin_offsetof(Object, color) == 0, "Object::color is at the wrong offset");
static_assert(__builtin_offsetof(Object, cell) == 16, "Object::cell is at the wrong offset");
static_assert(__builtin_offsetof(Object, scale) == 32, "Object::scale is at the wrong offset");
static_assert(__builtin_offsetof(Object, orbit) == 36, "Object::orbit is at the wrong offset");
static_assert(__builtin_offsetof(Object, phase) == 40, "Object::phase is at the wrong offset");
static_assert(__builtin_offsetof(Object, model) == 44, "Object::model is at the wrong offset");

static_assert(sizeof(Payload) == 272, "Payload has the wrong size");
static_assert(alignof(Payload) == 16, "Payload has the wrong alignment");
static_assert(__builtin_offsetof(Payload, mvp) == 0, "Payload::mvp is at the wrong offset");
static_assert(__builtin_offsetof(Payload, model) == 64, "Payload::model is at the wrong offset");
static_assert(__builtin_offsetof(Payload, color) == 128, "Payload::color is at the wrong offset");
static_assert(__builtin_offsetof(Payload, meshlets) == 144, "Payload::meshlets is at the wrong offset");

static_assert(sizeof(Scene) == 240, "Scene has the wrong size");
static_assert(alignof(Scene) == 16, "Scene has the wrong alignment");
static_assert(__builtin_offsetof(Scene, objects) == 0, "Scene::objects is at the wrong offset");
static_assert(__builtin_offsetof(Scene, models) == 8, "Scene::models is at the wrong offset");
static_assert(__builtin_offsetof(Scene, meshlets) == 16, "Scene::meshlets is at the wrong offset");
static_assert(__builtin_offsetof(Scene, meshlet_vertices) == 24, "Scene::meshlet_vertices is at the wrong offset");
static_assert(__builtin_offsetof(Scene, meshlet_triangles) == 32, "Scene::meshlet_triangles is at the wrong offset");
static_assert(__builtin_offsetof(Scene, vertices) == 40, "Scene::vertices is at the wrong offset");
static_assert(__builtin_offsetof(Scene, viewproj) == 48, "Scene::viewproj is at the wrong offset");
static_assert(__builtin_offsetof(Scene, planes) == 112, "Scene::planes is at the wrong offset");
static_assert(__builtin_offsetof(Scene, camera) == 208, "Scene::camera is at the wrong offset");
static_assert(__builtin_offsetof(Scene, time) == 224, "Scene::time is at the wrong offset");
static_assert(__builtin_offsetof(Scene, debug) == 228, "Scene::debug is at the wrong offset");
static_assert(__builtin_offsetof(Scene, count) == 232, "Scene::count is at the wrong offset");
static_assert(__builtin_offsetof(Scene, pad) == 236, "Scene::pad is at the wrong offset");

struct VertexOut {
    float4 position [[position]];
    float3 normal;
};

struct PrimitiveOut {
    float4 color;
};

struct FragmentIn {
    VertexOut vertex_out;
    PrimitiveOut primitive_out;
};

using MeshOut = metal::mesh<VertexOut, PrimitiveOut, 64, 126, metal::topology::triangle>;

[[object]] void meshlet_object(
    device const Scene* scene [[buffer(0)]],
    object_data Payload& payload [[payload]],
    mesh_grid_properties props,
    uint object_id [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    threadgroup atomic_uint visible;
    if (lane == 0) {
        atomic_store_explicit(&visible, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const Object* obj = scene->objects + object_id;
    float t = scene->time * obj->orbit + obj->phase;
    float3 center = obj->cell + float3(cos(t), sin(t * 1.3), sin(t)) * 0.12;
    float c = cos(t * 1.7);
    float s = sin(t * 1.7);
    float k = obj->scale;
    float4x4 model = float4x4(
        float4(c * k, 0.0, -s * k, 0.0),
        float4(0.0, k, 0.0, 0.0),
        float4(s * k, 0.0, c * k, 0.0),
        float4(center, 1.0));

    device const Model* m = scene->models + obj->model;
    if (lane < m->meshlet_count) {
        uint index = m->meshlet_start + lane;
        device const Meshlet* meshlet = scene->meshlets + index;
        float3 bounds_center = (model * float4(meshlet->bounds.xyz, 1.0)).xyz;
        float bounds_radius = meshlet->bounds.w * k;

        bool inside = true;
        for (uint i = 0; i < 6; i++) {
            float4 plane = scene->planes[i];
            inside = inside && dot(plane.xyz, bounds_center) + plane.w >= -bounds_radius;
        }

        float3 axis = normalize((model * float4(meshlet->cone.xyz, 0.0)).xyz);
        float3 view = bounds_center - scene->camera;
        bool away = dot(view, axis) >= meshlet->cone.w * length(view) + bounds_radius;

        if (inside && !away) {
            uint slot = atomic_fetch_add_explicit(&visible, 1, memory_order_relaxed);
            payload.meshlets[slot] = index;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        payload.mvp = scene->viewproj * model;
        payload.model = model;
        payload.color = obj->color;
        props.set_threadgroups_per_grid(
            uint3(atomic_load_explicit(&visible, memory_order_relaxed), 1, 1));
    }
}

[[mesh]] void meshlet_mesh(
    MeshOut out,
    device const Scene* scene [[buffer(0)]],
    const object_data Payload& payload [[payload]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    uint index = payload.meshlets[group];
    device const Meshlet* meshlet = scene->meshlets + index;

    if (lane == 0) {
        out.set_primitive_count(meshlet->triangle_count);
    }

    if (lane < meshlet->vertex_count) {
        float3 p = scene->vertices[scene->meshlet_vertices[meshlet->vertex_offset + lane]].position;
        VertexOut v;
        v.position = payload.mvp * float4(p, 1.0);
        v.normal = normalize((payload.model * float4(p, 0.0)).xyz);
        out.set_vertex(lane, v);
    }

    if (lane < meshlet->triangle_count) {
        uint packed = scene->meshlet_triangles[meshlet->triangle_offset + lane];
        out.set_index(lane * 3 + 0, uchar(packed & 0xff));
        out.set_index(lane * 3 + 1, uchar((packed >> 8) & 0xff));
        out.set_index(lane * 3 + 2, uchar((packed >> 16) & 0xff));

        uint h = index * 2654435761u;
        float3 tint = float3((h >> 16) & 255, (h >> 8) & 255, h & 255) / 255.0;
        PrimitiveOut prim;
        prim.color = float4(mix(payload.color.rgb, tint, scene->debug), 1.0);
        out.set_primitive(lane, prim);
    }
}

fragment float4 meshlet_fragment(FragmentIn in [[stage_in]]) {
    float3 n = normalize(in.vertex_out.normal);
    float light = 0.25 + 0.75 * saturate(dot(n, normalize(float3(0.4, 0.7, 0.55))));
    return float4(in.primitive_out.color.rgb * light, 1.0);
}
