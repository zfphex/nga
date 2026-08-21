// The `meshlets` example: an object stage culls meshlets and amplifies into a mesh stage that
// emits their geometry directly. No vertex buffer, no index buffer, no draw per meshlet. The
// GPU decides how much work there is after the CPU has stopped looking.
//
// Hand-written MSL, for the reason in `shaders/mesh.metal`: `#[gpu_mod]` has no mesh stage, and a
// mesh pipeline only ever runs on Metal.

#define MESHLETS_PER_GROUP 32
#define RING 8

struct Meshlet {
    // xyz is the centre, w the radius, so this is the sphere the object stage culls against.
    float4 bounds;
    packed_float3 color;
    uint _pad0;
};

struct Scene {
    device const Meshlet* meshlets;
    uint2 _pad0;
    float4x4 mvp;
    float cull_radius;
    float time;
    uint meshlet_count;
    uint _pad1;
};

struct Payload {
    uint meshlets[MESHLETS_PER_GROUP];
    float4x4 mvp;
    float time;
};

struct VertexOut {
    float4 position [[position]];
    float3 world;
};

struct PrimitiveOut {
    float4 color;
};

struct FragmentIn {
    VertexOut vertex_out;
    PrimitiveOut primitive_out;
};

using MeshOut = metal::mesh<VertexOut, PrimitiveOut, RING + 1, RING, metal::topology::triangle>;

// One thread per candidate meshlet. Survivors are compacted into the payload and the grid is
// sized to exactly that many mesh threadgroups, the amplification the stage is named for.
[[object]] void meshlet_object(
    constant nga_push* push [[buffer(0)]],
    object_data Payload& payload [[payload]],
    mesh_grid_properties props,
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    device const Scene* scene = (device const Scene*)push->root;

    threadgroup atomic_uint visible;
    if (lane == 0) {
        atomic_store_explicit(&visible, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint index = group * MESHLETS_PER_GROUP + lane;
    if (index < scene->meshlet_count) {
        float4 bounds = scene->meshlets[index].bounds;
        // A sphere that breathes, so the culled set changes every frame and the amplification
        // is visibly doing something.
        if (length(bounds.xyz) - bounds.w <= scene->cull_radius) {
            uint slot = atomic_fetch_add_explicit(&visible, 1, memory_order_relaxed);
            payload.meshlets[slot] = index;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        payload.mvp = scene->mvp;
        payload.time = scene->time;
        props.set_threadgroups_per_grid(
            uint3(atomic_load_explicit(&visible, memory_order_relaxed), 1, 1));
    }
}

// One threadgroup per surviving meshlet, emitting a fan: a centre vertex and a ring.
[[mesh]] void meshlet_mesh(
    MeshOut out,
    constant nga_push* push [[buffer(0)]],
    const object_data Payload& payload [[payload]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    device const Scene* scene = (device const Scene*)push->root;
    uint index = payload.meshlets[group];
    device const Meshlet* meshlet = scene->meshlets + index;
    float3 centre = meshlet->bounds.xyz;
    float radius = meshlet->bounds.w;

    if (lane == 0) {
        out.set_primitive_count(RING);
    }

    if (lane <= RING) {
        float3 position = centre;
        if (lane > 0) {
            float angle = float(lane - 1) / float(RING) * 6.2831853 + payload.time * 0.5;
            position += float3(cos(angle), sin(angle), 0.0) * radius;
        }
        VertexOut v;
        v.position = payload.mvp * float4(position, 1.0);
        v.world = position;
        out.set_vertex(lane, v);
    }

    if (lane < RING) {
        out.set_index(lane * 3 + 0, 0);
        out.set_index(lane * 3 + 1, uchar(1 + lane));
        out.set_index(lane * 3 + 2, uchar(1 + (lane + 1) % RING));

        // Tint by meshlet index so the per-primitive output is visibly per-meshlet.
        uint hash = index * 2654435761u;
        float3 tint = float3((hash >> 16) & 255, (hash >> 8) & 255, hash & 255) / 255.0;
        PrimitiveOut p;
        p.color = float4(mix(float3(meshlet->color), tint, 0.35), 1.0);
        out.set_primitive(lane, p);
    }
}

[[fragment]] float4 meshlet_fragment(FragmentIn in [[stage_in]]) {
    float shade = 0.55 + 0.45 * cos(length(in.vertex_out.world) * 6.0);
    return float4(in.primitive_out.color.rgb * shade, 1.0);
}
