// M3's exit criterion: a bindless storage texture, written through a handle that the host
// wrote into the resource heap. Nothing here knows which backend it is running on.
struct Data {
    nga_texture2d_rw target;
    uint2 size;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void gradient_main(uint3 tid : SV_DispatchThreadID) {
    Data* data = (Data*)push.root;
    if (tid.x >= data->size.x || tid.y >= data->size.y) {
        return;
    }
    float2 uv = float2(tid.xy) / float2(data->size);
    nga_store(data->target, tid.xy, float4(uv.x, uv.y, 0.0, 1.0));
}
