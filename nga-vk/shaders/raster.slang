struct Vertex {
    float3 position;
    float3 color;
};

struct Data {
    Vertex* vertices;
    uint _nga_pad0[2];
    float4x4 mvp;
    float4 tint;
};

struct Push {
    uint64_t root;
    uint resources;
    uint samplers;
};

[[vk::push_constant]] ConstantBuffer<Push> push;

// Metal function constants, one for one: the id is the field index in `#[derive(Constants)]`.
[[vk::constant_id(0)]] const bool boost = false;
[[vk::constant_id(1)]] const float gamma = 1.0;

struct VertexOut {
    float4 position : SV_Position;
    float3 color : COLOR;
    float alpha : ALPHA;
};

[shader("vertex")]
VertexOut vertex_main(uint vid : SV_VertexID) {
    Data* data = (Data*)push.root;
    Vertex v = data.vertices[vid];
    VertexOut out;
    out.position = mul(data.mvp, float4(v.position, 1.0));
    out.color = v.color * data.tint.rgb;
    out.alpha = data.tint.a;
    return out;
}

[shader("fragment")]
float4 fragment_main(VertexOut in) : SV_Target {
    float3 c = pow(in.color, float3(gamma, gamma, gamma));
    if (boost) {
        c = min(c * 1.6, 1.0);
    }
    return float4(c, in.alpha);
}

// The Metal original reads `float4 dst [[color(0)]]` — framebuffer fetch — and computes
// screen blending in the shader. Desktop Vulkan has no framebuffer fetch, so the fragment
// only emits `src` and the screen formula is expressed as a fixed blend equation instead:
// ONE, ONE_MINUS_SRC_COLOR gives src + dst*(1-src), which is exactly screen.
[shader("fragment")]
float4 fragment_screen(VertexOut in) : SV_Target {
    return float4(in.color * in.alpha, 1.0);
}
