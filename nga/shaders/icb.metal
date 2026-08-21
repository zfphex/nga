// A compute kernel that encodes draws into an indirect command buffer, and the pipeline those
// draws replay. `tests/icb.rs` checks that the render pass drew what the GPU asked for.
//
// Hand-written MSL because encoding a command buffer from a shader has no `#[gpu]` spelling;
// indirect command buffers are Metal-only anyway. This is the thing `draw_indirect_count`
// cannot do: the pipeline and the per-draw root pointer are chosen on the GPU, not on the
// host.

struct Vertex {
    packed_float3 position;
    packed_float3 color;
};

// One of these per encoded draw. The encoding kernel fills it in and hands its address to the
// command as the vertex stage's root pointer, so every draw gets its own data.
struct DrawData {
    device const Vertex* vertices;
    float2 offset;
    float4 color;
};

struct EncodeData {
    device const Vertex* vertices;
    device DrawData* draws;
    nga_commands commands;
    nga_pipeline pipeline;
    uint count;
};

kernel void encode(constant nga_push* push [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    device const EncodeData* data = (device const EncodeData*)push->root;
    if (i >= data->count) {
        return;
    }

    // A row of shrinking triangles, one per slot, so the coverage the test measures depends on
    // every command having been encoded.
    device DrawData* draw = data->draws + i;
    draw->vertices = data->vertices;
    draw->offset = float2(-0.6 + 0.4 * float(i), 0.0);
    draw->color = float4(1.0, 1.0, 1.0, 1.0);

    render_command cmd(data->commands.cb, i);
    cmd.set_render_pipeline_state(data->pipeline.p);
    cmd.set_vertex_buffer(draw, 1);
    cmd.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);
}

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

// Buffer 1, not 0: index 0 is the argument table's root block, which the encoder binds for the
// whole pass and an encoded command does not displace. Per-draw data goes one slot along.
vertex VertexOut icb_vertex(device const DrawData* data [[buffer(1)]], uint vid [[vertex_id]]) {
    float3 p = float3(data->vertices[vid].position);
    VertexOut out;
    out.position = float4(p.xy * 0.25 + data->offset, p.z, 1.0);
    out.color = data->color;
    return out;
}

fragment float4 icb_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}
