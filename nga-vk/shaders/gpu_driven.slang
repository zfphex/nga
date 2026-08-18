struct Instance {
    float2 center;
    float angle;
    float scale;
    float4 color;
};

// VkDrawIndirectCommand, laid out by the spec rather than by us.
struct DrawArgs {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

struct SimData {
    Instance* instances;
    float time;
    uint count;
};

struct EncodeData {
    Instance* instances;
    DrawArgs* draws;
    Atomic<uint>* count;
    float cull_radius;
    uint total;
};

struct DrawData {
    Instance* instances;
    uint _nga_pad0[2];
    float4x4 mvp;
};

struct BoundaryData {
    float4x4 mvp;
    float radius;
    uint segments;
    uint _nga_pad0[2];
};

struct Push {
    uint64_t root;
    uint resources;
    uint samplers;
};

[[vk::push_constant]] ConstantBuffer<Push> push;

[shader("compute")]
[numthreads(64, 1, 1)]
void simulate(uint3 tid: SV_DispatchThreadID) {
    SimData* data = (SimData*)push.root;
    uint i = tid.x;
    if (i >= data.count) {
        return;
    }
    uint side = 64;
    float2 cell = (float2(i % side, i / side) + 0.5) / float(side) * 2.0 - 1.0;
    float n = frac(sin(float(i) * 12.9898) * 43758.5453);
    float phase = data.time * (0.4 + n) + n * 6.2831853;

    Instance inst;
    inst.center = cell + float2(cos(phase), sin(phase)) * 0.008;
    inst.angle = phase;
    inst.scale = 0.009 + 0.004 * n;
    inst.color = float4(0.5 + 0.5 * cos(6.2831853 * n + float3(0.0, 2.0, 4.0)), 1.0);
    data.instances[i] = inst;
}

// Metal builds a real `render_command` here and binds a per-draw root pointer to it. Vulkan
// has no GPU-side encoder, so this writes a VkDrawIndirectCommand instead and smuggles the
// instance index through `first_instance`, which the vertex stage reads back as gl_BaseInstance.
[shader("compute")]
[numthreads(64, 1, 1)]
void encode(uint3 tid: SV_DispatchThreadID) {
    EncodeData* data = (EncodeData*)push.root;
    uint i = tid.x;
    if (i >= data.total) {
        return;
    }
    Instance inst = data.instances[i];
    if (length(inst.center) > data.cull_radius) {
        return;
    }
    uint slot = data.count[0].add(1);
    data.draws[slot].vertex_count = 3;
    data.draws[slot].instance_count = 1;
    data.draws[slot].first_vertex = 0;
    data.draws[slot].first_instance = i;
}

struct VertexOut {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOut vertex_main(uint vid: SV_VertexID, uint base: SV_StartInstanceLocation) {
    DrawData* data = (DrawData*)push.root;
    Instance inst = data.instances[base];
    float a = inst.angle + float(vid) * 2.0943951;
    float2 p = inst.center + float2(cos(a), sin(a)) * inst.scale;

    VertexOut out;
    out.position = mul(data.mvp, float4(p, 0.5, 1.0));
    out.color = inst.color;
    return out;
}

[shader("fragment")]
float4 fragment_main(VertexOut in) : SV_Target {
    return in.color;
}

[shader("vertex")]
float4 boundary_main(uint vid: SV_VertexID) : SV_Position {
    BoundaryData* data = (BoundaryData*)push.root;
    float a = float(vid / 2) / float(data.segments) * 6.2831853;
    float r = data.radius + (vid % 2 == 0 ? -0.004 : 0.004);
    return mul(data.mvp, float4(cos(a) * r, sin(a) * r, 0.25, 1.0));
}

[shader("fragment")]
float4 boundary_fragment() : SV_Target {
    return float4(1.0, 1.0, 1.0, 0.6);
}
