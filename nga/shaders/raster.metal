struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut vertex_main(device const VertexData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 1.0);
    out.color = v.color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]], device const PixelData* data [[buffer(0)]]) {
    float3 c = pow(in.color * data->tint.rgb, float3(gamma));
    if (boost) {
        c = min(c * 1.6, 1.0);
    }
    return float4(c, data->tint.a);
}

fragment float4 fragment_screen(VertexOut in [[stage_in]], device const PixelData* data [[buffer(0)]], float4 dst [[color(0)]]) {
    float3 src = in.color * data->tint.rgb * data->tint.a;
    return float4(1.0 - (1.0 - dst.rgb) * (1.0 - src), 1.0);
}
