struct VertexOut {
    float4 position [[position]];
    float3 color;
    float alpha;
};

struct LayeredOut {
    float4 position [[position]];
    float3 color;
    float alpha;
    uint layer [[render_target_array_index]];
};

struct FullscreenOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut vertex_main(device const Data* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    VertexOut out;
    out.position = data->mvp * float4(v.position, 1.0);
    out.color = v.color * data->tint.rgb;
    out.alpha = data->tint.a;
    return out;
}

vertex LayeredOut vertex_layered(device const Data* data [[buffer(0)]], uint vid [[vertex_id]]) {
    Vertex v = data->vertices[vid];
    LayeredOut out;
    out.position = data->mvp * float4(v.position, 1.0);
    out.color = v.color * data->tint.rgb;
    out.alpha = data->tint.a;
    out.layer = data->layer;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(in.color, in.alpha);
}

fragment float4 fragment_layered(LayeredOut in [[stage_in]]) {
    return float4(in.color, in.alpha);
}

vertex FullscreenOut vertex_fullscreen(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    FullscreenOut out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = float2(p.x, 1.0 - p.y);
    return out;
}

fragment float4 fragment_backdrop(FullscreenOut in [[stage_in]],
                                  device const Data* data [[buffer(0)]]) {
    float band = 0.5 + 0.5 * sin((in.uv.x + in.uv.y) * 14.0 + data->tint.a * 6.2831853);
    return float4(data->tint.rgb * mix(0.25, 0.9, band), 1.0);
}

fragment float4 fragment_composite(FullscreenOut in [[stage_in]],
                                   device const Composite* data [[buffer(0)]],
                                   nga_heap(nga_texture2d, heap)) {
    float4 scene = heap[data->color].t.sample(nga_linear_clamp, in.uv);
    float depth = nga_heap_as(nga_depth2d, heap)[data->depth].t.sample(nga_nearest_clamp, in.uv);
    scene.rgb *= mix(1.0, 0.45, smoothstep(0.88, 1.0, depth));

    if (in.uv.y > 0.75 && in.uv.x < 0.5) {
        uint slice = in.uv.x < 0.25 ? 0 : 1;
        float2 uv = float2(fract(in.uv.x * 4.0), (in.uv.y - 0.75) * 4.0);
        scene = nga_heap_as(nga_texture2d_array, heap)[data->layers].t.sample(
            nga_linear_clamp, uv, slice);
    }
    return float4(scene.rgb, 1.0);
}

struct TileColor {
    half4 color [[color(0)]];
};

kernel void tile_vignette(imageblock<TileColor, imageblock_layout_implicit> block,
                          device const Composite* data [[buffer(0)]],
                          ushort2 tid [[thread_position_in_threadgroup]],
                          uint2 group [[threadgroup_position_in_grid]]) {
    float2 pixel = float2(group * data->tile.xy + uint2(tid));
    float2 d = pixel / float2(data->size.xy) - 0.5;
    TileColor texel = block.read(tid);
    texel.color.rgb *= half(1.0 - 1.1 * dot(d, d));
    block.write(texel, tid);
}
