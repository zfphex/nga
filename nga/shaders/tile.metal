// A tile pipeline, so `tests/tile.rs` can check that a compute function running inside the
// render pass sees the imageblock the draws just wrote. No store, no reload, no second pass.
//
// This is the one feature in the crate that is Apple-only for good rather than for now: an
// imageblock is tile-based deferred hardware, and there is nothing in Vulkan to grow into.
// Slang has no tile stage either, which is why this is MSL.

struct Vertex {
    packed_float3 position;
    packed_float3 color;
};

struct Data {
    device const Vertex* vertices;
    uint2 _pad0;
    float4x4 mvp;
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut tile_vertex(constant nga_push* push [[buffer(0)]], uint vid [[vertex_id]]) {
    device const Data* data = (device const Data*)push->root;
    VertexOut out;
    out.position = data->mvp * float4(float3(data->vertices[vid].position), 1.0);
    out.color = float3(data->vertices[vid].color);
    return out;
}

fragment float4 tile_fragment(VertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}

struct TileColor {
    half4 color [[color(0)]];
};

// Halves whatever the draws left in the tile. A plain post-pass would have to store the render
// target and read it back; this never leaves tile memory.
kernel void tile_halve(
    imageblock<TileColor, imageblock_layout_implicit> block,
    ushort2 tid [[thread_position_in_threadgroup]]) {
    TileColor texel = block.read(tid);
    texel.color.rgb *= half(0.5);
    block.write(texel, tid);
}
