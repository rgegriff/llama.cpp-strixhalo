// Determine the gfx11 wave32 WMMA f32_16x16x16_f16 fragment layout empirically.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
typedef __attribute__((ext_vector_type(16))) _Float16 halfx16;
typedef __attribute__((ext_vector_type(8))) float floatx8;
// a[l][e], b[l][e] given per lane; output c[l][e] per lane
__global__ void probe(const _Float16 * a, const _Float16 * b, float * c) {
    const int l = threadIdx.x;
    halfx16 af, bf; floatx8 cf;
    for (int e = 0; e < 16; ++e) { af[e] = a[l*16 + e]; bf[e] = b[l*16 + e]; }
    for (int e = 0; e < 8; ++e) cf[e] = 0.0f;
    cf = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, cf);
    for (int e = 0; e < 8; ++e) c[l*8 + e] = cf[e];
}
int main() {
    _Float16 ha[32*16], hb[32*16]; float hc[32*8];
    // Test 1: A lane l holds row l%16 of A where A[i][k] = (i==k) (identity), B lane l holds col... B[l][k] = l%16*16 + k (distinct)
    // then D[i][j] = sum_k A_i[k] B_j[k]; with A_i = e_i: D[i][j] = B_j[i] = j*16+i -> tells (i,j) of each output slot.
    for (int l = 0; l < 32; ++l) for (int k = 0; k < 16; ++k) { ha[l*16+k] = ((l%16) == k) ? 1.0f : 0.0f; hb[l*16+k] = (float)((l%16)*16 + k); }
    _Float16 *da, *db; float *dc;
    hipMalloc(&da, sizeof(ha)); hipMalloc(&db, sizeof(hb)); hipMalloc(&dc, sizeof(hc));
    hipMemcpy(da, ha, sizeof(ha), hipMemcpyHostToDevice); hipMemcpy(db, hb, sizeof(hb), hipMemcpyHostToDevice);
    probe<<<1,32>>>(da, db, dc); hipMemcpy(hc, dc, sizeof(hc), hipMemcpyDeviceToHost);
    printf("Test1 (A=identity rows per lane, B_j[k]=16j+k): lane e -> value = 16*j + i\n");
    for (int l = 0; l < 32; l += 1) { printf("lane %2d:", l); for (int e = 0; e < 8; ++e) { int v = (int)hc[l*8+e]; printf(" (i=%2d,j=%2d)", v%16, v/16); } printf("\n"); }
    // Test 2: do upper lanes matter? garbage in lanes 16..31 of A and B
    for (int l = 16; l < 32; ++l) for (int k = 0; k < 16; ++k) { ha[l*16+k] = 7.0f; hb[l*16+k] = -3.0f; }
    hipMemcpy(da, ha, sizeof(ha), hipMemcpyHostToDevice); hipMemcpy(db, hb, sizeof(hb), hipMemcpyHostToDevice);
    float hc2[32*8]; probe<<<1,32>>>(da, db, dc); hipMemcpy(hc2, dc, sizeof(hc2), hipMemcpyDeviceToHost);
    int diff = 0; for (int i = 0; i < 256; ++i) diff += hc2[i] != hc[i];
    printf("Test2 upper-lane garbage: %d of 256 outputs differ\n", diff);
    // Test 3: garbage only in lower lanes 0..15, good data in upper
    for (int l = 0; l < 32; ++l) for (int k = 0; k < 16; ++k) { bool good = l >= 16; ha[l*16+k] = good ? (((l%16) == k) ? 1.0f : 0.0f) : 7.0f; hb[l*16+k] = good ? (float)((l%16)*16 + k) : -3.0f; }
    hipMemcpy(da, ha, sizeof(ha), hipMemcpyHostToDevice); hipMemcpy(db, hb, sizeof(hb), hipMemcpyHostToDevice);
    probe<<<1,32>>>(da, db, dc); hipMemcpy(hc2, dc, sizeof(hc2), hipMemcpyDeviceToHost);
    diff = 0; for (int i = 0; i < 256; ++i) diff += hc2[i] != hc[i];
    printf("Test3 lower-lane garbage: %d of 256 outputs differ\n", diff);
    return 0;
}
