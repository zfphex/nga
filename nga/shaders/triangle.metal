#include <metal_stdlib>
using namespace metal;

struct Vertex {
    packed_float3 position;
    packed_float3 color;
};

struct Data {
    device const Vertex* vertices;
    float4x4 mvp;
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut vertex_main(device const Data* data [[buffer(0)]], uint vid [[vertex_id]]) {
    VertexOut out;
    out.position = data->mvp * float4(data->vertices[vid].position, 1.0);
    out.color = data->vertices[vid].color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
