// Which operand lanes/elements does gfx11 wmma_f32_16x16x16_f16 actually read? Corrupt subsets and see which outputs change.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
typedef __attribute__((ext_vector_type(16))) _Float16 halfx16;
typedef __attribute__((ext_vector_type(8))) float floatx8;
__global__ void probe(const _Float16 * a, const _Float16 * b, float * c) {
    const int l = threadIdx.x; halfx16 af, bf; floatx8 cf;
    for (int e = 0; e < 16; ++e) { af[e] = a[l*16 + e]; bf[e] = b[l*16 + e]; }
    for (int e = 0; e < 8; ++e) cf[e] = 0.0f;
    cf = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, cf);
    for (int e = 0; e < 8; ++e) c[l*8 + e] = cf[e];
}
static _Float16 *da, *db; static float *dc;
static void run(const _Float16 * ha, const _Float16 * hb, float * hc) {
    hipMemcpy(da, ha, 512*2, hipMemcpyHostToDevice); hipMemcpy(db, hb, 512*2, hipMemcpyHostToDevice);
    probe<<<1,32>>>(da, db, dc); hipMemcpy(hc, dc, 256*4, hipMemcpyDeviceToHost);
}
int main() {
    hipMalloc(&da, 1024); hipMalloc(&db, 1024); hipMalloc(&dc, 1024);
    _Float16 ha[512], hb[512], ta[512], tb[512]; float ref[256], out[256];
    for (int l = 0; l < 32; ++l) for (int k = 0; k < 16; ++k) { ha[l*16+k] = (_Float16)(0.25f*(((l%16)*3 + k) % 7) - 2.0f); hb[l*16+k] = (_Float16)(0.5f*(((l%16)*5 + 3*k) % 11) - 2.0f); }
    run(ha, hb, ref);
    auto test = [&](const char * name, bool corruptA, int lane_lo, int lane_hi, int e_lo, int e_hi) {
        memcpy(ta, ha, sizeof ha); memcpy(tb, hb, sizeof hb);
        _Float16 * t = corruptA ? ta : tb;
        for (int l = lane_lo; l < lane_hi; ++l) for (int e = e_lo; e < e_hi; ++e) t[l*16 + e] = (_Float16) 99.0f;
        run(ta, tb, out);
        int diff = 0, rows = 0, cols = 0; bool rowset[16] = {}, colset[16] = {};
        for (int l = 0; l < 32; ++l) for (int e = 0; e < 8; ++e) if (out[l*8+e] != ref[l*8+e]) { ++diff; rowset[2*e + l/16] = true; colset[l%16] = true; }
        for (int i = 0; i < 16; ++i) { rows += rowset[i]; cols += colset[i]; }
        printf("%-44s: %3d outputs differ; rows touched %2d, cols touched %2d\n", name, diff, rows, cols);
    };
    test("A lanes 16-31 all elems garbage",   true,  16, 32, 0, 16);
    test("A lanes 0-15 all elems garbage",    true,   0, 16, 0, 16);
    test("A lanes 16-31 elems 0-7 garbage",   true,  16, 32, 0, 8);
    test("A lanes 16-31 elems 8-15 garbage",  true,  16, 32, 8, 16);
    test("A lanes 0-15 elems 0-7 garbage",    true,   0, 16, 0, 8);
    test("A lanes 0-15 elems 8-15 garbage",   true,   0, 16, 8, 16);
    test("B lanes 16-31 all elems garbage",   false, 16, 32, 0, 16);
    test("B lanes 16-31 elems 0-7 garbage",   false, 16, 32, 0, 8);
    test("B lanes 16-31 elems 8-15 garbage",  false, 16, 32, 8, 16);
    test("B lanes 0-15 elems 0-7 garbage",    false,  0, 16, 0, 8);
    test("B lanes 0-15 elems 8-15 garbage",   false,  0, 16, 8, 16);
    return 0;
}
