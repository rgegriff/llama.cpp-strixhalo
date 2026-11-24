// Achievable wmma_i32_16x16x16_iu8 rate on gfx1151 (register-only, independent accumulators per wave), next to the
// f16 rate, for the compute ceiling of the MMQ expert matmuls.
#include <hip/hip_runtime.h>
#include <cstdio>
typedef __attribute__((ext_vector_type(4))) int intx4;
typedef __attribute__((ext_vector_type(8))) int intx8;
typedef __attribute__((ext_vector_type(16))) _Float16 halfx16;
typedef __attribute__((ext_vector_type(8))) float floatx8;

template <int NACC>
__global__ __launch_bounds__(256) void peak_iu8(const int * in, int * out, int iters) {
    intx4 a, b; for (int e = 0; e < 4; ++e) { a[e] = in[threadIdx.x*4 + e]; b[e] = in[1024 + threadIdx.x*4 + e]; }
    intx8 c[NACC]; for (int i = 0; i < NACC; ++i) for (int e = 0; e < 8; ++e) c[i][e] = 0;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i) c[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c[i], false);
        a[0] += 1;
    }
    int s = 0; for (int i = 0; i < NACC; ++i) for (int e = 0; e < 8; ++e) s += c[i][e];
    out[blockIdx.x*256 + threadIdx.x] = s;
}
template <int NACC>
__global__ __launch_bounds__(256) void peak_f16(const _Float16 * in, float * out, int iters) {
    halfx16 a, b; for (int e = 0; e < 16; ++e) { a[e] = in[threadIdx.x*16 + e]; b[e] = in[1024 + threadIdx.x*16 + e]; }
    floatx8 c[NACC]; for (int i = 0; i < NACC; ++i) for (int e = 0; e < 8; ++e) c[i][e] = 0.0f;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i) c[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[i]);
        a[0] += (_Float16) 1e-7f;
    }
    float s = 0; for (int i = 0; i < NACC; ++i) for (int e = 0; e < 8; ++e) s += c[i][e];
    out[blockIdx.x*256 + threadIdx.x] = s;
}
// dp4a (v_dot4_i32_i8) rate for comparison: 4 independent chains per lane
__global__ __launch_bounds__(256) void peak_dp4a(const int * in, int * out, int iters) {
    int a = in[threadIdx.x], b = in[256 + threadIdx.x];
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int it = 0; it < iters; ++it) {
        c0 = __builtin_amdgcn_sudot4(true, a, true, b, c0, false);
        c1 = __builtin_amdgcn_sudot4(true, a ^ 1, true, b, c1, false);
        c2 = __builtin_amdgcn_sudot4(true, a ^ 2, true, b, c2, false);
        c3 = __builtin_amdgcn_sudot4(true, a ^ 3, true, b, c3, false);
        a += 1;
    }
    out[blockIdx.x*256 + threadIdx.x] = c0 + c1 + c2 + c3;
}

template <int NACC> void run_iu8(int blocks, int iters) {
    int * in; int * out; hipMalloc(&in, 8192); hipMalloc(&out, blocks*256*4); hipMemset(in, 1, 8192);
    peak_iu8<NACC><<<blocks, 256>>>(in, out, 10); hipDeviceSynchronize();
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a); peak_iu8<NACC><<<blocks, 256>>>(in, out, iters); hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    const double ops = 2.0*16*16*16*NACC*(double) iters*8*blocks;
    printf("iu8  NACC %2d, %4d blocks x 8 waves, %d iters: %7.2f ms, %6.1f TOPS\n", NACC, blocks, iters, ms, ops/(ms*1e-3)/1e12);
    hipFree(in); hipFree(out);
}
template <int NACC> void run_f16(int blocks, int iters) {
    _Float16 * in; float * out; hipMalloc(&in, 8192); hipMalloc(&out, blocks*256*4); hipMemset(in, 0, 8192);
    peak_f16<NACC><<<blocks, 256>>>(in, out, 10); hipDeviceSynchronize();
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a); peak_f16<NACC><<<blocks, 256>>>(in, out, iters); hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    const double ops = 2.0*16*16*16*NACC*(double) iters*8*blocks;
    printf("f16  NACC %2d, %4d blocks x 8 waves, %d iters: %7.2f ms, %6.1f TFLOPS\n", NACC, blocks, iters, ms, ops/(ms*1e-3)/1e12);
    hipFree(in); hipFree(out);
}
void run_dp4a(int blocks, int iters) {
    int * in; int * out; hipMalloc(&in, 8192); hipMalloc(&out, blocks*256*4); hipMemset(in, 1, 8192);
    peak_dp4a<<<blocks, 256>>>(in, out, 10); hipDeviceSynchronize();
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a); peak_dp4a<<<blocks, 256>>>(in, out, iters); hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    const double ops = 2.0*4*4*(double) iters*256*blocks;   // 4 dot4 per iter per lane, 8 ops each
    printf("dp4a %4d blocks x 256 lanes, %d iters: %7.2f ms, %6.1f TOPS\n", blocks, iters, ms, ops/(ms*1e-3)/1e12);
    hipFree(in); hipFree(out);
}
int main() {
    run_iu8<8>(40*4, 20000); run_iu8<8>(40*8, 20000); run_iu8<4>(40*8, 20000); run_iu8<16>(40*4, 10000); run_iu8<8>(40*1, 40000);
    run_f16<8>(40*4, 20000); run_f16<8>(40*8, 20000);
    run_dp4a(40*8, 200000); run_dp4a(40*4, 200000);
    return 0;
}
