// Exercises the forked half of the prelude: a struct holding bindless handles has to land on
// the same offsets whether the handles are descriptor indices or MTLResourceIDs.
struct Material {
    nga_texture2d albedo;
    nga_sampler samp;
    nga_texture2d_rw target;
    float2 uv;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void shade_main(uint3 tid : SV_DispatchThreadID) {
    Material* m = (Material*)push.root;
    nga_store(m->target, tid.xy, nga_sample(m->albedo, m->samp, m->uv));
}
