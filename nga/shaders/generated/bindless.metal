
#include <metal_stdlib>
using namespace metal;

[[maybe_unused]] constexpr sampler nga_nearest_clamp(filter::nearest, address::clamp_to_edge);
[[maybe_unused]] constexpr sampler nga_linear_clamp(filter::linear, mip_filter::linear, address::clamp_to_edge);
[[maybe_unused]] constexpr sampler nga_linear_repeat(filter::linear, mip_filter::linear, address::repeat);
[[maybe_unused]] constexpr sampler nga_aniso_repeat(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(16));

struct nga_texture2d { texture2d<float> t; };
struct nga_texture2d_array { texture2d_array<float> t; };
struct nga_texturecube { texturecube<float> t; };
struct nga_sampler { sampler s; };

static_assert(sizeof(nga_texture2d) == 8, "nga_texture2d is not one resource id");
static_assert(sizeof(nga_texture2d_array) == 8, "nga_texture2d_array is not one resource id");
static_assert(sizeof(nga_texturecube) == 8, "nga_texturecube is not one resource id");
static_assert(sizeof(nga_sampler) == 8, "nga_sampler is not one resource id");

struct Vertex {
    packed_float2 position;
    packed_float2 uv;
};

struct Material {
    uint texture_base;
    packed_float3 tint;
};

struct Data {
    device const Vertex* vertices;
    device const Material* materials;
    float4x4 mvp;
};

static_assert(sizeof(Vertex) == 16, "Vertex has the wrong size");
static_assert(alignof(Vertex) == 4, "Vertex has the wrong alignment");
static_assert(__builtin_offsetof(Vertex, position) == 0, "Vertex::position is at the wrong offset");
static_assert(__builtin_offsetof(Vertex, uv) == 8, "Vertex::uv is at the wrong offset");

static_assert(sizeof(Material) == 16, "Material has the wrong size");
static_assert(alignof(Material) == 4, "Material has the wrong alignment");
static_assert(__builtin_offsetof(Material, texture_base) == 0, "Material::texture_base is at the wrong offset");
static_assert(__builtin_offsetof(Material, tint) == 4, "Material::tint is at the wrong offset");

static_assert(sizeof(Data) == 80, "Data has the wrong size");
static_assert(alignof(Data) == 16, "Data has the wrong alignment");
static_assert(__builtin_offsetof(Data, vertices) == 0, "Data::vertices is at the wrong offset");
static_assert(__builtin_offsetof(Data, materials) == 8, "Data::materials is at the wrong offset");
static_assert(__builtin_offsetof(Data, mvp) == 16, "Data::mvp is at the wrong offset");

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

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              device const Data* data [[buffer(0)]],
                              device const nga_texture2d* textures [[buffer(1)]]) {
    Material m = data->materials[in.material];
    uint layer = min(uint(in.uv.x * 3.0), 2u);
    float3 c = textures[m.texture_base + layer].t.sample(nga_linear_clamp, in.uv).rgb;
    return float4(c * m.tint, 1.0);
}
