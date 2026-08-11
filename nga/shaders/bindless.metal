struct VertexOut {
    float4 position [[position]];
    float2 uv;
    uint material [[flat]];
};

vertex VertexOut vertex_main(device const Data* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 0.0, 1.0);
    out.uv = v.uv;
    out.material = vid / 6;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]], device const Data* data [[buffer(0)]]) {
    Material m = data->materials[in.material];
    uint layer = min(uint(in.uv.x * 3.0), 2u);
    float3 c = data->textures[m.texture_base + layer].t.sample(nga_linear_clamp, in.uv).rgb;
    return float4(c * m.tint, 1.0);
}
