struct VertexOut {
    float4 position [[position]];
    float3 color;
    float alpha;
};

vertex VertexOut vertex_main(device const Data* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 1.0);
    out.color = v.color * data->tint.rgb;
    out.alpha = data->tint.a;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    float3 c = pow(in.color, float3(gamma));
    if (boost) {
        c = min(c * 1.6, 1.0);
    }
    return float4(c, in.alpha);
}

fragment float4 fragment_screen(VertexOut in [[stage_in]], float4 dst [[color(0)]]) {
    float3 src = in.color * in.alpha;
    return float4(1.0 - (1.0 - dst.rgb) * (1.0 - src), 1.0);
}
