// A mesh pipeline emitting one triangle, so `tests/mesh.rs` can check the same coverage
// `tests/raster.rs` checks for the vertex path.
//
// MSL rather than Slang, and not by preference. slangc lowers a mesh stage to Metal correctly
// but emits the *fragment* stage_in as a flat struct of `[[user(...)]]` varyings, which is the
// vertex-pipeline convention; a mesh pipeline's fragment has to take the mesh output structs
// themselves, and Metal rejects the mismatch at pipeline creation. Mesh pipelines are Metal-only
// anyway, so an MSL source costs nothing that was portable to begin with.

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
