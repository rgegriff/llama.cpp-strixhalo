// Probe the gfx11 wave32 WMMA i32_16x16x16_iu8 fragment layout (expect: same as f16: lane l holds D[2e + l/16][l%16]).
#include <hip/hip_runtime.h>
#include <cstdio>
typedef __attribute__((ext_vector_type(4))) int intx4;
typedef __attribute__((ext_vector_type(8))) int intx8;
__global__ void probe(const signed char * a, const signed char * b, int * c, int garbage_upper) {
    const int l = threadIdx.x;
    intx4 af, bf; intx8 cf;
    for (int e = 0; e < 4; ++e) {
        unsigned ua = 0, ub = 0;
        for (int j = 0; j < 4; ++j) {
            int va = a[l*16 + 4*e + j], vb = b[l*16 + 4*e + j];
            if (garbage_upper && l >= 16) { va = 77; vb = -5; }
            ua |= ((unsigned) va & 0xff) << (8*j); ub |= ((unsigned) vb & 0xff) << (8*j);
        }
        af[e] = (int) ua; bf[e] = (int) ub;
    }
    for (int e = 0; e < 8; ++e) cf[e] = 0;
    cf = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af, true, bf, cf, false);
    for (int e = 0; e < 8; ++e) c[l*8 + e] = cf[e];
}
int main() {
    signed char ha[32*16], hb[32*16]; int hc[32*8], hc2[32*8];
    // A lane l: row l%16 = e_{l%16} (identity); B lane l: B[j][k] = 8*j + k - 60 (signed, distinct) -> D[i][j] = B[j][i]
    for (int l = 0; l < 32; ++l) for (int k = 0; k < 16; ++k) { ha[l*16+k] = ((l%16) == k) ? 1 : 0; hb[l*16+k] = (signed char) (8*(l%16) + k - 60); }
    signed char *da, *db; int *dc;
    hipMalloc(&da, sizeof(ha)); hipMalloc(&db, sizeof(hb)); hipMalloc(&dc, sizeof(hc));
    hipMemcpy(da, ha, sizeof(ha), hipMemcpyHostToDevice); hipMemcpy(db, hb, sizeof(hb), hipMemcpyHostToDevice);
    probe<<<1,32>>>(da, db, dc, 0); hipMemcpy(hc, dc, sizeof(hc), hipMemcpyDeviceToHost);
    int bad = 0;
    for (int l = 0; l < 32; ++l) for (int e = 0; e < 8; ++e) { const int i = 2*e + l/16, j = l % 16; if (hc[l*8+e] != 8*j + i - 60) ++bad; }
    printf("iu8 layout D[2e + l/16][l%%16]: %d of 256 mismatches\n", bad);
    probe<<<1,32>>>(da, db, dc, 1); hipMemcpy(hc2, dc, sizeof(hc2), hipMemcpyDeviceToHost);
    int diff = 0; for (int i = 0; i < 256; ++i) diff += hc2[i] != hc[i];
    printf("upper-lane garbage in A and B: %d of 256 outputs differ\n", diff);
    return 0;
}
