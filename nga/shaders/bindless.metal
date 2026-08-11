struct VertexOut {
    float4 position [[position]];
    float2 uv;
    uint material [[flat]];
};

kernel void paint(
    device const Paint* data [[buffer(0)]],
    nga_heap(nga_texture2d_rw, heap),
    uint2 tid [[thread_position_in_grid]]
) {
    if (tid.x >= data->size || tid.y >= data->size) {
        return;
    }
    float2 p = (float2(tid) + 0.5) / float(data->size) * 2.0 - 1.0;
    float r = length(p);
    float wave = sin(r * 18.0 - data->time * 6.0) * 0.5 + 0.5;
    texture2d<float, access::read_write> dst = heap[data->target].t;
    float4 prev = dst.read(tid);
    float4 lit = float4(wave, wave * (1.0 - r), prev.b * 0.9 + 0.1 * r, 1.0);
    dst.write(lit, tid);
}

vertex VertexOut vertex_main(device const Data* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 0.0, 1.0);
    out.uv = v.uv;
    out.material = vid / 6;
    return out;
}

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    device const Data* data [[buffer(0)]],
    nga_heap(nga_texture2d, heap)
) {
    Material m = data->materials[in.material];
    uint layer = min(uint(in.uv.x * 3.0), 2u);
    sampler s = nga_heap_as(nga_sampler, heap)[m.sampler].s;
    float3 c = heap[m.texture_base + layer].t.sample(s, in.uv).rgb;
    return float4(c * m.tint, 1.0);
}
