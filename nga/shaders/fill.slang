struct Data {
    uint* output;
    uint count;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void fill_main(uint3 tid : SV_DispatchThreadID) {
    Data* data = (Data*)push.root;
    if (tid.x < data.count) {
        data.output[tid.x] = tid.x * 3 + 1;
    }
}
