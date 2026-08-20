// The `gpu_meshes` example: a compute kernel encodes a whole render pass into an indirect
// command buffer, choosing the *pipeline* per object as well as the arguments.
//
// That last part is what separates this from `gpu_driven.slang`, which is the portable version
// of the same idea: `draw_indirect_count` lets the GPU pick how many draws and what arguments,
// but every one of them shares the pipeline and the root pointer the host bound. Here each
// command carries its own.
//
// MSL because `render_command` has no Slang spelling. Indirect command buffers are Metal-only.

struct Vertex {
    packed_float3 position;
    packed_float3 color;
};

struct Object {
    packed_float3 cell;
    float scale;
    float orbit;
    float phase;
    // 0 solid, 1 wireframe — the index into the pipeline pair below.
    uint style;
    uint _pad0;
    float4 color;
};

// One per encoded draw, written by the kernel and bound by the command as buffer 1.
struct DrawData {
    device const Vertex* vertices;
    uint2 _pad0;
    float4x4 mvp;
    float4x4 model;
    float4 color;
};

struct EncodeData {
    device const Object* objects;
    device const Vertex* vertices;
    device DrawData* draws;
    nga_commands commands;
    nga_pipeline pipelines[2];
    float4x4 viewproj;
    float time;
    float cull_radius;
    uint count;
    uint tri_start;
    uint tri_count;
    uint line_start;
    uint line_count;
    uint _pad0;
};

kernel void encode(constant nga_push* push [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    device const EncodeData* data = (device const EncodeData*)push->root;
    if (i >= data->count) {
        return;
    }
    device const Object* object = data->objects + i;

    float t = data->time * object->orbit + object->phase;
    float3 centre = float3(object->cell) + float3(cos(t), sin(t * 1.3), sin(t)) * 0.12;
    // Culled objects simply leave their slot reset, which replays as nothing.
    if (length(centre) > data->cull_radius) {
        return;
    }

    float c = cos(t * 1.7);
    float s = sin(t * 1.7);
    float k = object->scale;
    float4x4 model = float4x4(
        float4(c * k, 0.0, -s * k, 0.0),
        float4(0.0, k, 0.0, 0.0),
        float4(s * k, 0.0, c * k, 0.0),
        float4(centre, 1.0));

    device DrawData* draw = data->draws + i;
    draw->vertices = data->vertices;
    draw->mvp = data->viewproj * model;
    draw->model = model;
    draw->color = object->color;

    bool wire = object->style != 0;
    render_command cmd(data->commands.cb, i);
    cmd.set_render_pipeline_state(data->pipelines[object->style].p);
    // Buffer 1, not 0: index 0 is the argument table's root block, which belongs to the pass.
    cmd.set_vertex_buffer(draw, 1);
    cmd.draw_primitives(
        wire ? primitive_type::line : primitive_type::triangle,
        wire ? data->line_start : data->tri_start,
        wire ? data->line_count : data->tri_count,
        1,
        0);
}

struct VertexOut {
    float4 position [[position]];
    float3 world;
    float4 color;
};

vertex VertexOut solid_vertex(device const DrawData* data [[buffer(1)]], uint vid [[vertex_id]]) {
    float3 p = float3(data->vertices[vid].position);
    VertexOut out;
    out.position = data->mvp * float4(p, 1.0);
    out.world = (data->model * float4(p, 1.0)).xyz;
    out.color = data->color;
    return out;
}

fragment float4 solid_fragment(VertexOut in [[stage_in]]) {
    float3 n = normalize(cross(dfdx(in.world), dfdy(in.world)));
    float light = 0.2 + 0.8 * abs(dot(n, normalize(float3(0.4, 0.7, 0.55))));
    return float4(in.color.rgb * light, 1.0);
}

vertex VertexOut wire_vertex(device const DrawData* data [[buffer(1)]], uint vid [[vertex_id]]) {
    float3 p = float3(data->vertices[vid].position) * 1.08;
    VertexOut out;
    out.position = data->mvp * float4(p, 1.0);
    out.world = p;
    out.color = data->color;
    return out;
}

fragment float4 wire_fragment(VertexOut in [[stage_in]]) {
    return float4(in.color.rgb * 1.4, 1.0);
}
