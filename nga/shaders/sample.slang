// The full M3 chain: a texture and a sampler, both reached through handles the host wrote
// into the resource heap, sampled into a storage texture.
//
// Note what this shader does *not* do: arithmetic on a handle. `texture_base + layer` works
// on Vulkan, where a handle is an index, and is meaningless on Metal, where it is an opaque
// MTLResourceID. An array of handles is the portable spelling.
struct Data {
    nga_texture2d layers[3];
    nga_sampler samp;
    nga_texture2d_rw target;
    uint2 size;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void sample_main(uint3 tid : SV_DispatchThreadID) {
    Data* data = (Data*)push.root;
    if (tid.x >= data->size.x || tid.y >= data->size.y) {
        return;
    }
    float2 uv = (float2(tid.xy) + 0.5) / float2(data->size);
    uint layer = min(uint(uv.x * 3.0), 2u);
    float4 c = nga_sample_level(data->layers[layer], data->samp, uv, 0.0);
    nga_store(data->target, tid.xy, c);
}
