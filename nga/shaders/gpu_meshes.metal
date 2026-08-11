kernel void encode(device const EncodeData* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= data->count) {
        return;
    }
    device const Object* obj = data->objects + i;
    float t = data->time * obj->orbit + obj->phase;
    float3 center = obj->cell + float3(cos(t), sin(t * 1.3), sin(t)) * 0.12;
    if (length(center) > data->cull_radius) {
        return;
    }

    device const Mesh* mesh = data->meshes + obj->mesh;
    uint slot = atomic_fetch_add_explicit(&data->range->length, 1, memory_order_relaxed);

    float c = cos(t * 1.7);
    float s = sin(t * 1.7);
    float k = obj->scale;
    float4x4 model = float4x4(
        float4(c * k, 0.0, -s * k, 0.0),
        float4(0.0, k, 0.0, 0.0),
        float4(s * k, 0.0, c * k, 0.0),
        float4(center, 1.0));

    device DrawData* draw = data->draws + slot;
    draw->vertices = data->vertices;
    draw->mvp = data->viewproj * model;
    draw->model = model;
    draw->color = obj->color;

    bool wire = obj->style != 0;
    uint start = wire ? mesh->line_start : mesh->tri_start;
    uint count = wire ? mesh->line_count : mesh->tri_count;

    render_command cmd(data->commands.cb, slot);
    cmd.set_render_pipeline_state(data->pipelines[obj->style].p);
    cmd.set_vertex_buffer(draw, 0);
    cmd.draw_indexed_primitives(
        wire ? primitive_type::line : primitive_type::triangle,
        count,
        data->indices + start,
        1,
        mesh->base_vertex,
        0);
}

struct VertexOut {
    float4 position [[position]];
    float3 world;
    float4 color;
};

vertex VertexOut solid_vertex(device const DrawData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    float3 p = data->vertices[vid].position;
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

vertex VertexOut wire_vertex(device const DrawData* data [[buffer(0)]], uint vid [[vertex_id]]) {
    float3 p = data->vertices[vid].position * 1.08;
    VertexOut out;
    out.position = data->mvp * float4(p, 1.0);
    out.world = p;
    out.color = data->color;
    return out;
}

fragment float4 wire_fragment(VertexOut in [[stage_in]]) {
    return float4(in.color.rgb * 1.4, 0.85);
}
