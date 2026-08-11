kernel void prepare(device const PrepareData* data [[buffer(0)]]) {
    data->encode->threadgroups[0] = (data->count + data->group - 1) / data->group;
    data->encode->threadgroups[1] = 1;
    data->encode->threadgroups[2] = 1;

    data->ring->vertex_count = data->segments * 2 + 2;
    data->ring->instance_count = 1;
    data->ring->vertex_start = 0;
    data->ring->base_instance = 0;
}

kernel void simulate(device const SimData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    uint side = 64;
    float2 cell = (float2(i % side, i / side) + 0.5) / float(side) * 2.0 - 1.0;
    float n = fract(sin(float(i) * 12.9898) * 43758.5453);
    float phase = data->time * (0.4 + n) + n * 6.2831853;

    Instance inst;
    inst.center = cell + float2(cos(phase), sin(phase)) * 0.008;
    inst.angle = phase;
    inst.scale = 0.009 + 0.004 * n;
    inst.color = float4(0.5 + 0.5 * cos(6.2831853 * n + float3(0.0, 2.0, 4.0)), 1.0);
    data->instances[i] = inst;
}

kernel void encode(device const EncodeData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    device const Instance* inst = data->instances + i;
    if (length(inst->center) > data->cull_radius) {
        return;
    }

    uint slot = atomic_fetch_add_explicit(&data->range->length, 1, memory_order_relaxed);
    device DrawData* draw = data->draws + slot;
    draw->instance = inst;
    draw->mvp = data->mvp;

    render_command cmd(data->commands.cb, slot);
    cmd.set_vertex_buffer(draw, 0);
    cmd.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);
}

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut vertex_main(device const DrawData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Instance inst = *data->instance;
    float a = inst.angle + float(vid) * 2.0943951;
    float2 p = inst.center + float2(cos(a), sin(a)) * inst.scale;

    VertexOut out;
    out.position = data->mvp * float4(p, 0.5, 1.0);
    out.color = inst.color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}

vertex float4 boundary_main(device const BoundaryData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    float a = float(vid / 2) / float(data->segments) * 6.2831853;
    float r = data->radius + (vid % 2 == 0 ? -0.004 : 0.004);
    return data->mvp * float4(cos(a) * r, sin(a) * r, 0.25, 1.0);
}

fragment float4 boundary_fragment() {
    return float4(1.0, 1.0, 1.0, 0.6);
}
