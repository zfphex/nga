// Every texture and sampler is reached through a handle the host wrote into the resource
// heap. Note what is absent: arithmetic on a handle. The Vulkan-only `texture_base + layer`
// this example used to do is meaningless on Metal, where a handle is an opaque MTLResourceID,
// so the layers are an array of handles instead.

struct Vertex {
    float2 position;
    float2 uv;
};

struct Material {
    nga_texture2d layers[3];
    nga_sampler samp;
    float3 tint;
};

struct Data {
    Vertex* vertices;
    Material* materials;
    float4x4 mvp;
};

struct Paint {
    nga_texture2d_rw target;
    uint size;
    float time;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void paint(uint3 tid : SV_DispatchThreadID) {
    Paint* data = (Paint*)push.root;
    if (tid.x >= data->size || tid.y >= data->size) {
        return;
    }
    float2 p = (float2(tid.xy) + 0.5) / float(data->size) * 2.0 - 1.0;
    float r = length(p);
    float wave = sin(r * 18.0 - data->time * 6.0) * 0.5 + 0.5;
    float4 prev = nga_load(data->target, tid.xy);
    nga_store(data->target, tid.xy, float4(wave, wave * (1.0 - r), prev.b * 0.9 + 0.1 * r, 1.0));
}

struct VertexOut {
    float4 position : SV_Position;
    float2 uv : UV;
    nointerpolation uint material : MATERIAL;
};

[shader("vertex")]
VertexOut vertex_main(uint vid : SV_VertexID) {
    Data* data = (Data*)push.root;
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = mul(data->mvp, float4(v.position, 0.0, 1.0));
    out.uv = v.uv;
    out.material = vid / 6;
    return out;
}

[shader("fragment")]
float4 fragment_main(VertexOut in) : SV_Target {
    Data* data = (Data*)push.root;
    Material m = data->materials[in.material];
    uint layer = min(uint(in.uv.x * 3.0), 2u);
    float3 c = nga_sample(m.layers[layer], m.samp, in.uv).rgb;
    return float4(c * m.tint, 1.0);
}
