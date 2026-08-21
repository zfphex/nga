// A mesh pipeline emitting one triangle, so `tests/mesh.rs` can check the same coverage
// `tests/raster.rs` checks for the vertex path.
//
// Hand-written MSL. `#[gpu]` has no mesh stage: the IR generalises to one, but mesh pipelines
// are Metal-only, so writing the emitter would buy a stage that only ever has one backend. See
// section 9 of `GPU-PLAN.md`.

struct Vertex {
    packed_float3 position;
    packed_float3 color;
};

struct Data {
    device const Vertex* vertices;
    uint2 _pad0;
    float4x4 mvp;
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

struct FragmentIn {
    VertexOut vertex_out;
};

using MeshOut = metal::mesh<VertexOut, void, 3, 1, metal::topology::triangle>;

[[mesh]] void mesh_main(
    MeshOut out,
    constant nga_push* push [[buffer(0)]],
    uint lane [[thread_position_in_threadgroup]]) {
    device const Data* data = (device const Data*)push->root;

    if (lane == 0) {
        out.set_primitive_count(1);
        out.set_index(0, 0);
        out.set_index(1, 1);
        out.set_index(2, 2);
    }

    VertexOut v;
    v.position = data->mvp * float4(float3(data->vertices[lane].position), 1.0);
    v.color = float3(data->vertices[lane].color);
    out.set_vertex(lane, v);
}

[[fragment]] float4 mesh_fragment(FragmentIn in [[stage_in]]) {
    return float4(in.vertex_out.color, 1.0);
}
