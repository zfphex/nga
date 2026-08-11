kernel void simulate(device const SimData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    float f = float(i) / float(data->count);
    float orbit = 0.25 + 0.65 * fract(f * 7.0);
    float phase = f * 6.2831853 + data->time * (0.2 + 0.9 * fract(f * 5.0));

    Instance inst;
    inst.center = float2(cos(phase), sin(phase)) * orbit;
    inst.angle = phase * 2.0 + data->time;
    inst.scale = 0.015 + 0.045 * fract(f * 3.0);
    inst.color = float4(0.5 + 0.5 * cos(6.2831853 * f + float3(0.0, 2.0, 4.0)), 1.0);
    data->instances[i] = inst;
}

kernel void build_vertices(device const BuildData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    Instance inst = data->instances[i];
    device Vertex* out = data->vertices + i * 3;
    for (uint k = 0; k < 3; ++k) {
        float a = inst.angle + float(k) * 2.0943951;
        float2 p = inst.center + float2(cos(a), sin(a)) * inst.scale;
        out[k].position = float3(p, 0.5);
        out[k].color = inst.color.rgb;
    }
}

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut vertex_main(device const DrawData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 1.0);
    out.color = v.color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
