// Metal's heap is one array indexed three ways. Vulkan needs three arrays across two
// descriptor buffers, because sampler descriptors and image descriptors cannot share one.
[[vk::binding(0, 0)]] Texture2D<float4> nga_textures[];
[[vk::binding(0, 1)]] RWTexture2D<float4> nga_storage[];
[[vk::binding(0, 2)]] SamplerState nga_samplers[];

struct Push {
    uint64_t root;
    uint resources;
    uint samplers;
};

[[vk::push_constant]] ConstantBuffer<Push> push;

struct Vertex {
    float2 position;
    float2 uv;
};

struct Material {
    uint texture_base;
    uint sampler;
    float3 tint;
};

struct Data {
    Vertex* vertices;
    Material* materials;
    float4x4 mvp;
};

struct Paint {
    uint target;
    uint size;
    float time;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void paint(uint3 tid: SV_DispatchThreadID) {
    Paint* data = (Paint*)push.root;
    if (tid.x >= data.size || tid.y >= data.size) {
        return;
    }
    float2 p = (float2(tid.xy) + 0.5) / float(data.size) * 2.0 - 1.0;
    float r = length(p);
    float wave = sin(r * 18.0 - data.time * 6.0) * 0.5 + 0.5;
    float4 prev = nga_storage[NonUniformResourceIndex(data.target)][tid.xy];
    float4 lit = float4(wave, wave * (1.0 - r), prev.b * 0.9 + 0.1 * r, 1.0);
    nga_storage[NonUniformResourceIndex(data.target)][tid.xy] = lit;
}

struct VertexOut {
    float4 position : SV_Position;
    float2 uv : UV;
    nointerpolation uint material : MATERIAL;
};

[shader("vertex")]
VertexOut vertex_main(uint vid : SV_VertexID) {
    Data* data = (Data*)push.root;
    Vertex v = data.vertices[vid];
    VertexOut out;
    out.position = mul(data.mvp, float4(v.position, 0.0, 1.0));
    out.uv = v.uv;
    out.material = vid / 6;
    return out;
}

[shader("fragment")]
float4 fragment_main(VertexOut in) : SV_Target {
    Data* data = (Data*)push.root;
    Material m = data.materials[in.material];
    uint layer = min(uint(in.uv.x * 3.0), 2u);
    SamplerState s = nga_samplers[NonUniformResourceIndex(m.sampler)];
    float3 c = nga_textures[NonUniformResourceIndex(m.texture_base + layer)]
                   .Sample(s, in.uv)
                   .rgb;
    return float4(c * m.tint, 1.0);
}
