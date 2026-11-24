// Achievable wmma_f32_16x16x16_f16 rate on gfx1151: register-only, 8 independent accumulators per wave (like the FA O tiles).
#include <hip/hip_runtime.h>
#include <cstdio>
typedef __attribute__((ext_vector_type(16))) _Float16 halfx16;
typedef __attribute__((ext_vector_type(8))) float floatx8;
template <int NACC>
__global__ __launch_bounds__(256) void peak(const _Float16 * in, float * out, int iters) {
    halfx16 a, b; for (int e = 0; e < 16; ++e) { a[e] = in[threadIdx.x*16 + e]; b[e] = in[1024 + threadIdx.x*16 + e]; }
    floatx8 c[NACC]; for (int i = 0; i < NACC; ++i) for (int e = 0; e < 8; ++e) c[i][e] = 0.0f;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i) c[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[i]);
        a[0] += (_Float16) 1e-7f;    // keep the compiler from folding the loop
    }
    float s = 0; for (int i = 0; i < NACC; ++i) for (int e = 0; e < 8; ++e) s += c[i][e];
    out[blockIdx.x*256 + threadIdx.x] = s;
}
template <int NACC> void run(int blocks, int iters) {
    _Float16 * in; float * out; hipMalloc(&in, 4096); hipMalloc(&out, blocks*256*4); hipMemset(in, 0, 4096);
    peak<NACC><<<blocks, 256>>>(in, out, 10); hipDeviceSynchronize();
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a); peak<NACC><<<blocks, 256>>>(in, out, iters); hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    const double flops = 2.0*16*16*16*NACC*(double) iters*8*blocks;   // 8 waves per block
    printf("NACC %d, %d blocks x 8 waves, %d iters: %.2f ms, %.1f TFLOPS\n", NACC, blocks, iters, ms, flops/(ms*1e-3)/1e12);
    hipFree(in); hipFree(out);
}
int main() {
    run<8>(40*4, 20000); run<8>(40*8, 20000); run<4>(40*8, 20000); run<16>(40*4, 10000); run<8>(40*2, 40000); run<8>(40*1, 40000);
    return 0;
}
