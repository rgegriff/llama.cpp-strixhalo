// Which 16-lane half supplies each A row / B column of gfx11 wave32 WMMA f32_16x16x16_f16? A lower lanes hold e_i (identity),
// upper lanes hold 2*e_i; B lower lanes hold B[j][k] = 16j + k + 1, upper lanes hold 3x that. D[i][j] = fA(i)*fB(j)*(16j + i + 1).
#include <hip/hip_runtime.h>
#include <cstdio>
typedef __attribute__((ext_vector_type(16))) _Float16 halfx16;
typedef __attribute__((ext_vector_type(8))) float floatx8;
__global__ void probe(float * c) {
    const int l = threadIdx.x, r = l % 16, hi = l / 16;
    halfx16 a, b; floatx8 acc;
    for (int k = 0; k < 16; ++k) { a[k] = (_Float16) ((r == k ? 1.0f : 0.0f)*(hi ? 2.0f : 1.0f)); b[k] = (_Float16) ((16*r + k + 1)*(hi ? 3.0f : 1.0f)); }
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    for (int e = 0; e < 8; ++e) c[l*8 + e] = acc[e];
}
int main() {
    float hc[256]; float * dc; (void) hipMalloc(&dc, sizeof(hc));
    probe<<<1, 32>>>(dc); (void) hipMemcpy(hc, dc, sizeof(hc), hipMemcpyDeviceToHost);
    int fa[16] = {0}, fb[16] = {0}, bad = 0;
    for (int l = 0; l < 32; ++l) for (int e = 0; e < 8; ++e) {
        const int i = 2*e + l/16, j = l % 16; const float base = 16*j + i + 1, v = hc[l*8 + e], f = v/base;
        // f in {1, 2, 3, 6}: fA = 1 or 2, fB = 1 or 3
        int fA = ((int) (f + 0.5f) % 3 == 0) ? (int) (f + 0.5f)/3 : (int) (f + 0.5f), fB = (int) (f + 0.5f)/fA;
        if (fa[i] && fa[i] != fA) ++bad; fa[i] = fA; if (fb[j] && fb[j] != fB) ++bad; fb[j] = fB;
    }
    printf("A row i supplied by: "); for (int i = 0; i < 16; ++i) printf("%s ", fa[i] == 1 ? "lo" : fa[i] == 2 ? "HI" : "??"); printf("\n");
    printf("B col j supplied by: "); for (int j = 0; j < 16; ++j) printf("%s ", fb[j] == 1 ? "lo" : fb[j] == 3 ? "HI" : "??"); printf("\n");
    printf("inconsistencies: %d\n", bad);
    printf("factor matrix f[i][j] (1: A lo,B lo; 2: A HI,B lo; 3: A lo,B HI; 6: A HI,B HI):\n");
    for (int i = 0; i < 16; ++i) { printf("row %2d:", i); for (int j = 0; j < 16; ++j) { const int l = j + 16*(i % 2), e = i/2; printf(" %d", (int) (hc[l*8 + e]/(16*j + i + 1) + 0.5f)); } printf("\n"); }
    return 0;
}
