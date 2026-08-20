// The `tbdr` example: draws a field of triangles, then runs a tile function over the imageblock
// to vignette and posterise it without ever storing the render target and reading it back.
//
// MSL, not Slang: there is no tile stage in the language, and imageblocks are Apple-only for
// good. `shaders/tile.metal` carries the same note and the headless version of the same test.

struct Vertex {
    packed_float3 position;
    packed_float3 color;
};

struct Data {
    device const Vertex* vertices;
    uint2 _pad0;
    float4x4 mvp;
};

struct Vignette {
    float2 size;
    float2 tile;
    float strength;
    float levels;
    uint2 _pad0;
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut tbdr_vertex(constant nga_push* push [[buffer(0)]], uint vid [[vertex_id]]) {
    device const Data* data = (device const Data*)push->root;
    VertexOut out;
    out.position = data->mvp * float4(float3(data->vertices[vid].position), 1.0);
    out.color = float3(data->vertices[vid].color);
    return out;
}

fragment float4 tbdr_fragment(VertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}

struct TileColor {
    half4 color [[color(0)]];
};

// One thread per pixel of the tile. `block.read`/`block.write` never leave tile memory, which
// is the entire point: the same effect as a full-screen post pass, with no bandwidth spent.
kernel void tbdr_vignette(
    imageblock<TileColor, imageblock_layout_implicit> block,
    constant nga_push* push [[buffer(0)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    device const Vignette* data = (device const Vignette*)push->root;

    float2 pixel = float2(group) * data->tile + float2(tid);
    float2 offset = pixel / data->size - 0.5;

    TileColor texel = block.read(tid);
    float3 color = float3(texel.color.rgb);
    color = floor(color * data->levels) / data->levels;
    color *= 1.0 - data->strength * dot(offset, offset);
    texel.color.rgb = half3(color);
    block.write(texel, tid);
}
