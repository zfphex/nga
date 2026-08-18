struct Vertex {
    float3 position;
    float3 color;
};

struct Data {
    Vertex* vertices;
    uint2 _pad0;
    float4x4 mvp;
};

struct Push {
    uint64_t root;
    uint resources;
    uint samplers;
};

[[vk::push_constant]] ConstantBuffer<Push> push;

struct VertexOut {
    float4 position : SV_Position;
    float3 color : COLOR;
};

[shader("vertex")]
VertexOut vertex_main(uint vid : SV_VertexID) {
    Data* data = (Data*)push.root;
    VertexOut out;
    out.position = mul(data.mvp, float4(data.vertices[vid].position, 1.0));
    out.color = data.vertices[vid].color;
    return out;
}

[shader("fragment")]
float4 fragment_main(VertexOut in) : SV_Target {
    return float4(in.color, 1.0);
}
