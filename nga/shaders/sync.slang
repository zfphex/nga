struct Instance {
    float2 center;
    float angle;
    float scale;
    float4 color;
};

struct Vertex {
    float3 position;
    float3 color;
};

struct SimData {
    Instance* instances;
    float time;
    uint count;
};

struct BuildData {
    Instance* instances;
    Vertex* vertices;
    uint count;
    uint pad;
};

struct DrawData {
    Vertex* vertices;
    uint _nga_pad0[2];
    float4x4 mvp;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void simulate(uint3 tid: SV_DispatchThreadID) {
    SimData* data = (SimData*)push.root;
    uint i = tid.x;
    if (i >= data.count) {
        return;
    }
    float f = float(i) / float(data.count);
    float orbit = 0.25 + 0.65 * frac(f * 7.0);
    float phase = f * 6.2831853 + data.time * (0.2 + 0.9 * frac(f * 5.0));

    Instance inst;
    inst.center = float2(cos(phase), sin(phase)) * orbit;
    inst.angle = phase * 2.0 + data.time;
    inst.scale = 0.015 + 0.045 * frac(f * 3.0);
    inst.color = float4(0.5 + 0.5 * cos(6.2831853 * f + float3(0.0, 2.0, 4.0)), 1.0);
    data.instances[i] = inst;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_vertices(uint3 tid: SV_DispatchThreadID) {
    BuildData* data = (BuildData*)push.root;
    uint i = tid.x;
    if (i >= data.count) {
        return;
    }
    Instance inst = data.instances[i];
    Vertex* out = data.vertices + i * 3;
    for (uint k = 0; k < 3; ++k) {
        float a = inst.angle + float(k) * 2.0943951;
        float2 p = inst.center + float2(cos(a), sin(a)) * inst.scale;
        out[k].position = float3(p, 0.5);
        out[k].color = inst.color.rgb;
    }
}

struct VertexOut {
    float4 position : SV_Position;
    float3 color : COLOR;
};

[shader("vertex")]
VertexOut vertex_main(uint vid : SV_VertexID) {
    DrawData* data = (DrawData*)push.root;
    Vertex v = data.vertices[vid];
    VertexOut out;
    out.position = mul(data.mvp, float4(v.position, 1.0));
    out.color = v.color;
    return out;
}

[shader("fragment")]
float4 fragment_main(VertexOut in) : SV_Target {
    return float4(in.color, 1.0);
}
