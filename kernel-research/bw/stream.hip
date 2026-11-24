// Read-bandwidth ceiling on this GPU: stream N bytes with 4/8/16-byte loads, many blocks, sum into one word.
// Sizes match a full-attn decode visit's K+V at 131k: f16 537 MB, q8_0 285 MB, q4_0 151 MB; also 32 MB (MALL-sized).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

template <typename T>
__global__ void rd(const T * p, size_t n, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const T v = p[i];
        const unsigned * w = (const unsigned *) &v;
        if constexpr (sizeof(T) < 4) { acc ^= (unsigned) *(const unsigned short *) &v; } else { for (int k = 0; k < (int) (sizeof(T) / 4); ++k) acc ^= w[k]; }
    }
    if (acc == 0x12345678u) atomicAdd(out, acc);  // keep the loads alive
}

template <typename T>
float bench(const char * buf, size_t bytes, int grid, int block, unsigned * out) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    const size_t n = bytes / sizeof(T);
    rd<T><<<grid, block>>>((const T *) buf, n, out);
    hipDeviceSynchronize();
    const int reps = 20;
    hipEventRecord(a);
    for (int r = 0; r < reps; ++r) rd<T><<<grid, block>>>((const T *) buf, n, out);
    hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    return bytes / (ms / reps * 1e-3f) / 1e9f;
}

int main() {
    const size_t maxb = 600ull << 20;
    char * buf; unsigned * out;
    hipMalloc(&buf, maxb); hipMalloc(&out, 4);
    hipMemset(buf, 1, maxb);
    int nsm; hipDeviceGetAttribute(&nsm, hipDeviceAttributeMultiprocessorCount, 0);
    printf("CUs %d\n", nsm);
    for (int off : {0, 2}) {
        for (size_t mb : {32, 285}) {
            const size_t bytes = (mb << 20) - 64;
            const int grid = nsm * 64;
            printf("offset %d %4zu MB  2B %6.1f  4B %6.1f  8B %6.1f  16B %6.1f GB/s\n", off, mb,
                   bench<unsigned short>(buf + off, bytes, grid, 256, out), bench<unsigned>(buf + off, bytes, grid, 256, out),
                   bench<uint2>(buf + off, bytes, grid, 256, out), bench<uint4>(buf + off, bytes, grid, 256, out));
        }
    }
    return 0;
}
