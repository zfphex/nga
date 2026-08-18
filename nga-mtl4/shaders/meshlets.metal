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
