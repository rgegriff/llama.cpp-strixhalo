// Standalone prototype: Inkling 2 Mini SMALL-BATCH decode attention (n_q = 1..16 queries, e.g. the
// verify step of n-gram speculative decoding) over a q8_0 or q4_0 KV cache, gfx1151. Reads K/V once per chunk for all
// queries and all 3 Q heads of a KV head.
//
// Semantics (ggml FLASH_ATTN_EXT with Inkling's indexed bias):
//   Q     f32  element (d, qi, hq) at Q[qi*qs_q + hq*qs_h + d]      (ggml ne = [128, n_q, 24]; in llama.cpp qs_h = 128, qs_q = 24*128)
//   K/V   q8_0 or q4_0, position k / KV head h row at base + k*nb_pos + h*nb_head (llama.cpp cache: nb_head = row bytes,
//         nb_pos = 8*row bytes). Q head hq uses KV head hq/3.
//   mask  f16  mask[qi*mask_stride + k]; the queries are the last n_q positions and causal among themselves (built by the host).
//   bias  idx i32 [n_q][idx_stride >= n_kv], table f32 [n_q][24][n_bins]; logit bias per (qi, hq, k):
//         bin = idx[qi][k]; (uint32) bin < n_bins ? half(float(mask) + table[qi][hq][bin]) : mask   (half rounding as llama.cpp)
//   sinks f32 [24] or null: denominator += exp(sink - max)
//   out   f32 out[(qi*24 + hq)*128 + d]
//
// Design: one block per chunk of positions (and per group of HPB KV heads). n_q <= 2: one wave per KV head, the fa_decode2
// int8-dot design with 3*n_q columns. n_q >= 4: WPH = ceil(3*n_q/16) waves per KV head, each owning 16 (query, head) columns
// (col = g*n_q + qi); per 32 keys each wave loads the 32 K rows (lane = key) and the V rows (lane = 8 dims, 2 keys per load),
// S^T = K.Q^T with int8 WMMA (q8_0 block scales applied to the int32 tiles), online softmax in the WMMA tile layout, P and V
// to f16 and O^T += V^T.P^T with f16 WMMA. The waves of one KV head read the same K/V bytes (L2 hits). Split-K + combine.
//
//   ./fa_smallbatch <q8_0|q4_0> <n_kv> <n_q> [chunk...]
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Build-time switches. Defaults are the measured-best configuration; the FA_ABL_* switches are timing-only ablations
// (results are wrong with them). See NOTES.md.
#ifndef FA_BRANCH
#define FA_BRANCH 1        // real branch between the two key tiles: splits scheduling regions, no spills (212 VGPRs vs 256 + spills)
#endif
#ifndef FA_FENCE
#define FA_FENCE 0
#endif
#ifndef FA_V0_EARLY
#define FA_V0_EARLY 0      // 1: load tile 0 (V, mask, indices) before the QK phase
#endif
#ifndef FA_V1_EARLY
#define FA_V1_EARLY 0      // 1: load tile 1 before the QK phase as well (one load phase per iteration; 256 VGPRs)
#endif
#ifndef FA_KPREFETCH
#define FA_KPREFETCH 0     // 1: K rows one iteration ahead (spills badly)
#endif
#ifndef FA_SYNC
#define FA_SYNC 0          // 1: block barrier per iteration, 3: before every load phase (no gain)
#endif
#ifndef FA_PREFETCH_TOP
#define FA_PREFETCH_TOP 0  // staged kernel: issue the next stage's loads at the top of the iteration (1) or after QK (0)
#endif
#ifndef FA_ABL_NOKV
#define FA_ABL_NOKV 0
#endif
#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int D = 128, HKV = 8, G = 3, HQ = HKV*G, QK = 32, NB = D/QK, WAVE = 32;

struct q8_0_t { static constexpr int BLK = 34; static constexpr bool Q4 = false; };  // half d, int8 qs[32]
struct q4_0_t { static constexpr int BLK = 18; static constexpr bool Q4 = true;  };  // half d, uint8 qs[16]: lo = i, hi = i+16, value-8
template <typename T> constexpr int ROW_BYTES = NB*T::BLK;

typedef __attribute__((ext_vector_type(16))) _Float16 halfx16;
typedef __attribute__((ext_vector_type(8)))  float    floatx8;
typedef __attribute__((ext_vector_type(4)))  int      intx4;
typedef __attribute__((ext_vector_type(8)))  int      intx8;

struct Params {
    const float * Q; int64_t qs_q, qs_h; float scale;
    const char * K; const char * V; int64_t nb_pos, nb_head;
    const __half * mask; int64_t mask_stride;
    const int32_t * idx; int64_t idx_stride;
    const float * table; int64_t table_stride_q; int n_bins;
    int n_kv, chunk;
    float * part_acc; float2 * part_ml;   // [chunk][col][D], [chunk][col]
};

__device__ __forceinline__ uint32_t word_at(const uint32_t * w, int byte) {   // 4 bytes at a 2-byte aligned offset
    return (byte % 4 == 0) ? w[byte/4] : __builtin_amdgcn_alignbyte(w[byte/4 + 1], w[byte/4], byte % 4);
}
__device__ __forceinline__ float half_at(const uint32_t * w, int byte) {
    return __half2float(__ushort_as_half((unsigned short) (w[byte/4] >> (8*(byte % 4)))));
}
__device__ __forceinline__ float half_bits(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short) bits)); }
// value of lane^16: v_permlanex16 is a VALU op (ds_bpermute would be an LDS round trip)
__device__ __forceinline__ uint32_t xchg16(uint32_t v) { return __builtin_amdgcn_permlanex16(0u, v, 0x76543210u, 0xfedcba98u, true, true); }
__device__ __forceinline__ float    xchg16(float v)    { return __uint_as_float(xchg16(__float_as_uint(v))); }
__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    return (uint32_t) __half_as_ushort(__float2half(a)) | ((uint32_t) __half_as_ushort(__float2half(b)) << 16);
}
// mask + indexed bias, rounded to half as fattn_indexed_bias::add does
__device__ __forceinline__ float bias_val(uint32_t mask_bits, int bin, const float * tbl, int n_bins) {
    const float mk = half_bits(mask_bits);
    return (uint32_t) bin < (uint32_t) n_bins ? __half2float(__float2half(mk + tbl[bin])) : mk;
}

// ---------------------------------------------------------------------------------------------------------------------
// n_q <= 2: one wave per KV head, int8 dot products (fa_decode2 with NC = 3*NQ columns)
template <typename T, int NQ, int HPB>
__global__ __launch_bounds__(HPB*WAVE) void fa_small_valu(const Params p) {
    constexpr int ROW = ROW_BYTES<T>, NC = G*NQ, NCOL = HQ*NQ;
    const int lane = threadIdx.x % WAVE, hl = threadIdx.x / WAVE, h = blockIdx.y*HPB + hl, c = blockIdx.x, k0 = c*p.chunk;

    __shared__ int   q_i8[HPB][NC][D/4];
    __shared__ float q_d [HPB][NC][NB];
    __shared__ int   q_s [HPB][NC][NB];
    __shared__ float p_s [HPB][NC][WAVE];

    for (int cc = 0; cc < NC; ++cc) {
        const int g = cc/NQ, qi = cc % NQ;
        const float4 x = ((const float4 *) (p.Q + qi*p.qs_q + (h*G + g)*p.qs_h))[lane];
        const float x0 = x.x*p.scale, x1 = x.y*p.scale, x2 = x.z*p.scale, x3 = x.w*p.scale;
        float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
        for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor(amax, o, WAVE));
        const float d = amax/127.0f, id = d > 0.0f ? 1.0f/d : 0.0f;
        const int b0 = (int) lrintf(x0*id), b1 = (int) lrintf(x1*id), b2 = (int) lrintf(x2*id), b3 = (int) lrintf(x3*id);
        q_i8[hl][cc][lane] = (b0 & 0xff) | ((b1 & 0xff) << 8) | ((b2 & 0xff) << 16) | ((b3 & 0xff) << 24);
        int s = b0 + b1 + b2 + b3;
        for (int o = 1; o < 8; o <<= 1) s += __shfl_xor(s, o, WAVE);
        if (lane % 8 == 0) { q_d[hl][cc][lane/8] = d; q_s[hl][cc][lane/8] = s; }
    }
    __syncthreads();

    float m[NC], l[NC], acc[NC][4];
    #pragma unroll
    for (int cc = 0; cc < NC; ++cc) { m[cc] = -1e30f; l[cc] = 0.0f; acc[cc][0] = acc[cc][1] = acc[cc][2] = acc[cc][3] = 0.0f; }

    const int kend = min(k0 + p.chunk, p.n_kv);
    for (int t0 = k0; t0 < kend; t0 += WAVE) {
        const int k = t0 + lane;
        float s[NC];
        #pragma unroll
        for (int cc = 0; cc < NC; ++cc) s[cc] = -INFINITY;
        // NC <= 3: the compiler hoists the wave's Q (32*NC dwords) into registers, as fa_decode2 does (196 VGPRs); for more
        // columns that would spill, so launder the LDS index to keep the reads inside the loop
        int qz = 0;
        if constexpr (NC > 3) { asm volatile("" : "+v"(qz)); }
        if (k < kend) {
            const char * row = p.K + (int64_t) k*p.nb_pos + (int64_t) h*p.nb_head;
            uint32_t w[ROW/4 + 1];
            #pragma unroll
            for (int i = 0; i < ROW/8; ++i) { const uint2 v = ((const uint2 *) row)[i]; w[2*i] = v.x; w[2*i + 1] = v.y; }
            w[ROW/4] = 0;
            float sum[NC];
            #pragma unroll
            for (int cc = 0; cc < NC; ++cc) sum[cc] = 0.0f;
            #pragma unroll
            for (int b = 0; b < NB; ++b) {
                const int off = b*T::BLK;
                const float dk = half_at(w, off);
                int isum[NC];
                #pragma unroll
                for (int cc = 0; cc < NC; ++cc) isum[cc] = 0;
                int4 qv[NC][2];
                #pragma unroll
                for (int cc = 0; cc < NC; ++cc) { qv[cc][0] = *(const int4 *) &q_i8[hl][cc][qz + b*8]; qv[cc][1] = *(const int4 *) &q_i8[hl][cc][qz + b*8 + 4]; }
                if constexpr (!T::Q4) {
                    #pragma unroll
                    for (int j = 0; j < QK/4; ++j) {
                        const uint32_t kq = word_at(w, off + 2 + 4*j);
                        #pragma unroll
                        for (int cc = 0; cc < NC; ++cc) isum[cc] = __builtin_amdgcn_sudot4(true, (int) kq, true, qv[cc][j/4][j % 4], isum[cc], false);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < QK/8; ++j) {
                        const uint32_t kq = word_at(w, off + 2 + 4*j);
                        const int lo = (int) (kq & 0x0F0F0F0Fu), hi = (int) ((kq >> 4) & 0x0F0F0F0Fu);
                        #pragma unroll
                        for (int cc = 0; cc < NC; ++cc) {
                            isum[cc] = __builtin_amdgcn_sudot4(false, lo, true, qv[cc][0][j], isum[cc], false);
                            isum[cc] = __builtin_amdgcn_sudot4(false, hi, true, qv[cc][1][j], isum[cc], false);
                        }
                    }
                    #pragma unroll
                    for (int cc = 0; cc < NC; ++cc) isum[cc] -= 8*q_s[hl][cc][b];
                }
                #pragma unroll
                for (int cc = 0; cc < NC; ++cc) sum[cc] += dk*q_d[hl][cc][b]*(float) isum[cc];
            }
            #pragma unroll
            for (int qi = 0; qi < NQ; ++qi) {
                const uint32_t mb = __half_as_ushort(p.mask[qi*p.mask_stride + k]);
                const int bi = p.idx[qi*p.idx_stride + k];
                #pragma unroll
                for (int g = 0; g < G; ++g) s[g*NQ + qi] = sum[g*NQ + qi] + bias_val(mb, bi, p.table + qi*p.table_stride_q + (h*G + g)*p.n_bins, p.n_bins);
            }
        }
        #pragma unroll
        for (int cc = 0; cc < NC; ++cc) {
            float mt = s[cc];
            for (int o = 1; o < WAVE; o <<= 1) mt = fmaxf(mt, __shfl_xor(mt, o, WAVE));
            const float mn = fmaxf(m[cc], mt), sc = __expf(m[cc] - mn);
            m[cc] = mn;
            const float pr = __expf(s[cc] - mn);
            l[cc] = l[cc]*sc + pr;
            acc[cc][0] *= sc; acc[cc][1] *= sc; acc[cc][2] *= sc; acc[cc][3] *= sc;
            p_s[hl][cc][lane] = pr;
        }
        __builtin_amdgcn_wave_barrier();
        const int vb = lane/8, vi = 4*(lane % 8);
        const int vo = T::Q4 ? vb*T::BLK + 2 + (vi % 16) : vb*T::BLK + 2 + vi;
        const int nt = min(WAVE, kend - t0);
        for (int i = 0; i < nt; ++i) {
            const char * row = p.V + (int64_t) (t0 + i)*p.nb_pos + (int64_t) h*p.nb_head;
            const float dv = __half2float(*(const __half *) (row + vb*T::BLK));
            uint32_t q = *(const uint32_t *) (row + vo);
            float v0, v1, v2, v3;
            if constexpr (!T::Q4) {
                v0 = dv*(float) (int8_t) q; v1 = dv*(float) (int8_t) (q >> 8); v2 = dv*(float) (int8_t) (q >> 16); v3 = dv*(float) (int8_t) (q >> 24);
            } else {
                q = (vi < 16 ? q : q >> 4) & 0x0F0F0F0Fu;
                v0 = dv*(float) ((int) (q & 0xff) - 8); v1 = dv*(float) ((int) ((q >> 8) & 0xff) - 8);
                v2 = dv*(float) ((int) ((q >> 16) & 0xff) - 8); v3 = dv*(float) ((int) (q >> 24) - 8);
            }
            #pragma unroll
            for (int cc = 0; cc < NC; ++cc) {
                const float pr = p_s[hl][cc][i];
                acc[cc][0] += pr*v0; acc[cc][1] += pr*v1; acc[cc][2] += pr*v2; acc[cc][3] += pr*v3;
            }
        }
        __builtin_amdgcn_wave_barrier();
    }
    #pragma unroll
    for (int cc = 0; cc < NC; ++cc) {
        float ls = l[cc];
        for (int o = 1; o < WAVE; o <<= 1) ls += __shfl_xor(ls, o, WAVE);
        const int col = (h*G + cc/NQ)*NQ + cc % NQ;    // col = hq*NQ + qi
        ((float4 *) (p.part_acc + ((int64_t) c*NCOL + col)*D))[lane] = make_float4(acc[cc][0], acc[cc][1], acc[cc][2], acc[cc][3]);
        if (lane == 0) p.part_ml[(int64_t) c*NCOL + col] = make_float2(m[cc], ls);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// n_q >= 2: WPH waves per KV head, 16 columns each, f16 WMMA for both products. Fragment layout (probed: wmma_probe.hip,
// wmma_iu8_probe.hip, wmma_half_probe.hip): A/B lane l holds row l%16; D lane l holds D[2e + l/16][l%16], e = 0..7; the EVEN
// output rows are computed from the lower 16 lanes' A and B rows, the ODD rows from the upper 16 lanes'. So with B identical
// in both halves, A = the 32 K rows as loaded (lane = key) yields S for keys {2e} (lower lanes) and {17 + 2e} (upper), and
// A = the halves exchanged yields {16 + 2e} and {1 + 2e}: two 16-key tiles without any select. A tile's key for output lane
// (hh, e) is t0 + 16*(hh ^ tile) + 2e + hh; V, mask and indices are loaded per lane for exactly those keys.
typedef __attribute__((ext_vector_type(2))) _Float16 half2_t;
typedef const __attribute__((address_space(1))) char * gptr;
typedef __attribute__((ext_vector_type(2))) uint32_t u32x2;
typedef __attribute__((ext_vector_type(4))) uint32_t u32x4;
typedef __attribute__((ext_vector_type(4))) int32_t  i32x4;
template <typename U> __device__ __forceinline__ U gload(gptr base, uint32_t off) { return *(const __attribute__((address_space(1))) U *) (base + off); }
__device__ __forceinline__ uint32_t h2_bits(half2_t v) { return __builtin_bit_cast(uint32_t, v); }
__device__ __forceinline__ half2_t  bits_h2(uint32_t v) { return __builtin_bit_cast(half2_t, v); }
// two bytes (b_lo, b_hi) of a dword -> half2 (1024 + b_lo, 1024 + b_hi) via v_perm with the magic 0x6400 high bytes
constexpr uint32_t MAGIC_H2 = 0x64006400u, SEL_LO01 = 0x07010500u, SEL_HI23 = 0x07030502u;

template <typename T, int NQ, int HPB>
__global__ __launch_bounds__(HPB*((G*NQ + 15)/16)*WAVE) void fa_small_wmma(const Params p) {
    constexpr int ROW = ROW_BYTES<T>, NC = G*NQ, WPH = (NC + 15)/16, WAVES = HPB*WPH, NCOL = HQ*NQ;
    static_assert(WAVES*16*D*2 <= 65536, "Q f16 tiles exceed the LDS");
    static_assert(ROW % 16 == 8, "K row load pattern");
    const int lane = threadIdx.x % WAVE, wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE), hl = wave / WPH, w = wave % WPH;
    const int h = blockIdx.y*HPB + hl, c = blockIdx.x, k0 = c*p.chunk;
    const int nb_pos = (int) p.nb_pos;    // row pitch in bytes (fits 32 bits; lane offsets below stay 32-bit)
    const int l16 = lane & 15, hh = lane >> 4;
#if FA_ABL_HEADSHIFT
    const int hkv = (h + w) % HKV;        // ablation (wrong results): every wave of a head reads a different head's K/V
#else
    const int hkv = h;
#endif
#if FA_ABL_DUPCACHED
    const int tmul = w == 0 ? 1 : 0;      // ablation (wrong results): waves w > 0 always read positions 0..31 (cache hits)
#else
    const int tmul = 1;
#endif

    __shared__ uint32_t q_f16[WAVES][16][D/2];   // per wave: its 16 columns' scaled Q in f16; 16 B quads XOR-swizzled by (col & 7)

    // ---- setup: Q columns (col = g*NQ + qi) of this wave to f16, scaled, like llama.cpp's f16 kernels
    for (int cc = 0; cc < 16; ++cc) {
        const int col = 16*w + cc;
        float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (col < NC) {
            const int g = col/NQ, qi = col % NQ;
            x = ((const float4 *) (p.Q + qi*p.qs_q + (h*G + g)*p.qs_h))[lane];
        }
        const half2_t lo = {(_Float16) (x.x*p.scale), (_Float16) (x.y*p.scale)}, hi = {(_Float16) (x.z*p.scale), (_Float16) (x.w*p.scale)};
        uint32_t * dst = &q_f16[wave][cc][4*((lane/2) ^ (cc & 7)) + 2*(lane % 2)];
        dst[0] = h2_bits(lo); dst[1] = h2_bits(hi);
    }
    __syncthreads();

    const int col = 16*w + l16, colc = min(col, NC - 1), qi = colc % NQ, g = colc/NQ, hq = h*G + g;
    const float * tbl = p.table + qi*p.table_stride_q + hq*p.n_bins;

    // V operand: lane l16 owns dims 8*l16..8*l16+7 (block vb = l16/4), 2 keys per load (lanes 0-15: even key, 16-31: odd)
    const int vb = l16/4, vo = T::Q4 ? vb*T::BLK + 2 + 8*((l16 % 4) % 2) : vb*T::BLK + 2 + 8*(l16 % 4);
    const bool vhi = T::Q4 && (l16 % 4) >= 2;
    const uint32_t koff = lane*nb_pos;
    const uint32_t moff = (uint32_t) (qi*p.mask_stride)*2u, ioff = (uint32_t) (qi*p.idx_stride)*4u;    // byte offsets
    // readfirstlane: the result is known-uniform (SGPR pair; inline-asm outputs would count as divergent), opaque to LSR/LICM;
    // the explicit global address space keeps the loads global_load (saddr + 32-bit lane offset), not flat
    auto uniform_ptr = [](const char * ptr) {
        const uint64_t v = (uint64_t) ptr;
        const uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t) v), hi = __builtin_amdgcn_readfirstlane((uint32_t) (v >> 32));
        return (gptr) (((uint64_t) hi << 32) | lo);
    };
    // v_perm selectors: interleave (x = this half's keys, y = the other half's) into key order; low/high half-words
    const uint32_t sel_lo = hh ? 0x05040100u : 0x01000504u, sel_hi = hh ? 0x07060302u : 0x03020706u;
    auto interleave = [&](const uint32_t * x, const uint32_t * y, halfx16 & out) {
        union { uint32_t u[8]; halfx16 h; } r;
        #pragma unroll
        for (int i = 0; i < 4; ++i) { r.u[2*i] = __builtin_amdgcn_perm(x[i], y[i], sel_lo); r.u[2*i + 1] = __builtin_amdgcn_perm(x[i], y[i], sel_hi); }
        out = r.h;
    };
    const half2_t koffs = bits_h2(T::Q4 ? 0xE408E408u : 0xE480E480u);     // -(1024 + 8) / -(1024 + 128)

    floatx8 O[8];                              // O[t][e] = O[col][d = 8*(2e + hh) + t]
    #pragma unroll
    for (int t = 0; t < 8; ++t) O[t] = (floatx8) {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -1e30f, l = 0.0f;                // l: this lane's 8-of-16 keys; summed with lane^16 at the end

    const int kend = min(k0 + p.chunk, p.n_kv);

    // V rows of one key tile: lane holds keys 2jj + hh (8 keys), 8 bytes of dims + the block scale each; and the lane's
    // query's mask halves / bias indices of those keys
    struct Tile { u32x2 vraw[8]; uint32_t vsc[8]; uint32_t mb[8]; int bin[8]; };
    // tile 0/1 of the 32 keys at t0: this lane's keys are t0 + kofs(tile) + 2jj, kofs = 16*(hh ^ tile) + hh
    auto launder = [](uint32_t v) { asm volatile("" : "+v"(v)); return v; };
    auto load_tile = [&](int t0, int tile, Tile & T_) {
        const int kofs = 16*(hh ^ tile) + hh;
        const uint32_t voff = launder(kofs*nb_pos + vo), soff = launder(kofs*nb_pos + vb*T::BLK);
        const uint32_t mo = launder(moff + 32*(hh ^ tile)), io = launder(ioff + 64*(hh ^ tile));
        const char * vbase = p.V + (int64_t) (t0*tmul)*p.nb_pos + (int64_t) hkv*p.nb_head;
#if FA_ABL_NODUP
        if (w == 0 && !FA_ABL_NOKV) {
#endif
        #pragma unroll
        for (int jj = 0; jj < 8; ++jj) {       // per-key uniform base (SALU) + one lane offset: no hoisted per-key VGPR offsets
            const gptr bj = uniform_ptr(vbase + 2*jj*nb_pos);
            T_.vraw[jj] = gload<u32x2>(bj, voff);
            T_.vsc[jj]  = gload<unsigned short>(bj, soff);
        }
#if FA_ABL_NODUP
        } else {
            #pragma unroll
            for (int jj = 0; jj < 8; ++jj) { T_.vraw[jj] = (u32x2) {(uint32_t) (t0*jj + lane), (uint32_t) (t0 ^ jj)}; T_.vsc[jj] = 0x2c00 + (t0 & 7); }
        }
#endif
#if FA_ABL_NOMASK
        #pragma unroll
        for (int e = 0; e < 8; ++e) { T_.mb[e] = (t0 & 1) ? 0x8001u : 0u; T_.bin[e] = 1024; }
#else
        const gptr mp = uniform_ptr((const char *) (p.mask + t0));
        const u32x4 m0 = gload<u32x4>(mp, mo), m1 = gload<u32x4>(mp, mo + 16);
        const uint32_t md[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
        #pragma unroll
        for (int e = 0; e < 8; ++e) T_.mb[e] = hh ? (md[e] >> 16) : (md[e] & 0xffffu);
        const gptr ip = uniform_ptr((const char *) (p.idx + t0));
        #pragma unroll
        for (int i = 0; i < 4; ++i) { const i32x4 v = gload<i32x4>(ip, io + 16*i); T_.bin[2*i] = hh ? v.y : v.x; T_.bin[2*i + 1] = hh ? v.w : v.z; }
#endif
    };
    // bias + online softmax + PV of one key tile; S holds this lane's 8 logits (keys t0 + kofs + 2e); chunk and n_kv are
    // multiples of 32, so every tile is full
    auto process_tile = [&](const Tile & T_, float * S) {
        bool uni = true;
        #pragma unroll
        for (int e = 1; e < 8; ++e) uni = uni && T_.bin[e] == T_.bin[0];
        if (uni) {
            const bool valid = (uint32_t) T_.bin[0] < (uint32_t) p.n_bins;
            const float tb = valid ? tbl[T_.bin[0]] : 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) { const float mkf = half_bits(T_.mb[e]); S[e] += valid ? __half2float(__float2half(mkf + tb)) : mkf; }
        } else {
            #pragma unroll
            for (int e = 0; e < 8; ++e) S[e] += bias_val(T_.mb[e], T_.bin[e], tbl, p.n_bins);
        }
        float mt = S[0];
        #pragma unroll
        for (int e = 1; e < 8; ++e) mt = fmaxf(mt, S[e]);
        mt = fmaxf(mt, xchg16(mt));
        const float mn = fmaxf(m, mt), alpha = __expf(m - mn);
        m = mn;
        float pr[8], ps = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) { pr[e] = __expf(S[e] - mn); ps += pr[e]; }
        l = l*alpha + ps;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) O[t] *= alpha;
        }
        // P as the B operand: lane holds P[col][16 keys in order]
        halfx16 Pb;
        {
            uint32_t x[4], y[4];
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) x[e2] = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(pr[2*e2], pr[2*e2 + 1]));    // keys 4e2 + hh, 4e2 + 2 + hh
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) y[e2] = xchg16(x[e2]);
            interleave(x, y, Pb);
        }
        // V as the A operand, one dim slot t at a time: lane holds V[16 keys][d = 8*l16 + t]
        float dv[8], ndv[8]; uint32_t vx[8][2];
        #pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            dv[jj]  = half_bits(T_.vsc[jj]);
            ndv[jj] = T::Q4 ? -8.0f*dv[jj] : -128.0f*dv[jj];
            const u32x2 v = T_.vraw[jj];
            if constexpr (!T::Q4) { vx[jj][0] = v.x ^ 0x80808080u; vx[jj][1] = v.y ^ 0x80808080u; }
            else { vx[jj][0] = (vhi ? v.x >> 4 : v.x) & 0x0F0F0F0Fu; vx[jj][1] = (vhi ? v.y >> 4 : v.y) & 0x0F0F0F0Fu; }
        }
#if FA_ABL_NOPV
        #pragma unroll
        for (int t = 0; t < 8; ++t) O[t][0] += __uint_as_float(vx[t][0] ^ vx[t][1]) + Pb[t];   // keep the loads and P alive
#else
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            uint32_t x[4], y[4];
            #pragma unroll
            for (int j2 = 0; j2 < 4; ++j2) {        // keys 4j2 + hh, 4j2 + 2 + hh: v_cvt_f32_ubyte + v_fma_mixlo/hi
                half2_t r;
                r.x = (_Float16) fmaf((float) ((vx[2*j2][t/4] >> (8*(t % 4))) & 0xffu), dv[2*j2], ndv[2*j2]);
                r.y = (_Float16) fmaf((float) ((vx[2*j2 + 1][t/4] >> (8*(t % 4))) & 0xffu), dv[2*j2 + 1], ndv[2*j2 + 1]);
                x[j2] = h2_bits(r);
            }
            #pragma unroll
            for (int j2 = 0; j2 < 4; ++j2) y[j2] = xchg16(x[j2]);
            halfx16 Va;
            interleave(x, y, Va);
            O[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Va, Pb, O[t]);
            __builtin_amdgcn_sched_barrier(0);
        }
#endif
    };

    // K rows, lane = key (n_kv % 32 == 0: no clamping); FA_KPREFETCH: loaded one iteration ahead, after the QK phase
    uint32_t w_[ROW/4 + 1];
    auto load_K = [&](int t0) {
#if FA_ABL_NODUP
        if (w == 0 && !FA_ABL_NOKV)
#endif
        {
            const gptr row = uniform_ptr(p.K + (int64_t) (t0*tmul)*p.nb_pos + (int64_t) hkv*p.nb_head);
            const uint32_t ko = launder(koff);
            #pragma unroll
            for (int i = 0; i < ROW/16; ++i) { const u32x4 v = gload<u32x4>(row, ko + 16*i); w_[4*i] = v.x; w_[4*i + 1] = v.y; w_[4*i + 2] = v.z; w_[4*i + 3] = v.w; }
            { const u32x2 v = gload<u32x2>(row, ko + ROW - 8); w_[ROW/4 - 2] = v.x; w_[ROW/4 - 1] = v.y; }   // ROW % 16 == 8
            w_[ROW/4] = 0;
        }
#if FA_ABL_NODUP
        else {
            #pragma unroll
            for (int i = 0; i < ROW/4 + 1; ++i) w_[i] = (t0*i + lane) ^ 0x2c002c00u;
        }
#endif
    };
#if FA_KPREFETCH
    load_K(k0);
#endif
    #pragma unroll 1
    for (int t0 = k0; t0 < kend; t0 += WAVE) {
#if FA_SYNC
        __syncthreads();      // keep the WPH waves of a head in lockstep so their duplicate K/V reads hit the L0 cache
#endif
#if !FA_KPREFETCH
        load_K(t0);
#endif
        Tile T0;
#if FA_V0_EARLY
        load_tile(t0, 0, T0);
#endif
#if FA_V1_EARLY
        Tile T1;
        load_tile(t0, 1, T1);
#endif
        // launder the Q swizzle so the B-operand LDS reads stay in the loop (hoisted they would cost 64 VGPRs)
        int sw = l16 & 7;
        asm volatile("" : "+v"(sw));

        // ---- S^T = K.Q^T for both key tiles: lane holds S[kt][key = 16*kt + 2e + hh][col]
        floatx8 S0 = (floatx8) {0, 0, 0, 0, 0, 0, 0, 0}, S1 = S0;
#if FA_ABL_NOQK
        #pragma unroll
        for (int i = 0; i < 8; ++i) { S0[i] = __uint_as_float(w_[4*i] ^ w_[4*i + 1])*1e-30f; S1[i] = __uint_as_float(w_[4*i + 2] ^ w_[4*i + 3])*1e-30f; }
        if (false)
#endif
        #pragma unroll
        for (int dt = 0; dt < D/16; ++dt) {
            const int b = dt/2, half = dt % 2;
            const half2_t dk2 = {(_Float16) half_at(w_, b*T::BLK), (_Float16) half_at(w_, b*T::BLK)};
            // the lane's 16 dims of this d-tile as f16 (own); the halves exchanged (oth) give the second key tile (see above)
            uint32_t own[8], oth[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t kq;
                if constexpr (!T::Q4) kq = word_at(w_, b*T::BLK + 2 + 16*half + 4*j) ^ 0x80808080u;
                else { const uint32_t raw = word_at(w_, b*T::BLK + 2 + 4*j); kq = (half ? raw >> 4 : raw) & 0x0F0F0F0Fu; }
                const half2_t lo = (bits_h2(__builtin_amdgcn_perm(MAGIC_H2, kq, SEL_LO01)) + koffs)*dk2;
                const half2_t hi = (bits_h2(__builtin_amdgcn_perm(MAGIC_H2, kq, SEL_HI23)) + koffs)*dk2;
                own[2*j] = h2_bits(lo); own[2*j + 1] = h2_bits(hi);
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) oth[j] = xchg16(own[j]);
            union { uint32_t u[8]; halfx16 h; } bq;
            *(uint4 *) &bq.u[0] = *(const uint4 *) &q_f16[wave][l16][4*((2*dt) ^ sw)];
            *(uint4 *) &bq.u[4] = *(const uint4 *) &q_f16[wave][l16][4*((2*dt + 1) ^ sw)];
            S0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const halfx16 *) own, bq.h, S0);
            S1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const halfx16 *) oth, bq.h, S1);
            __builtin_amdgcn_sched_barrier(0);
        }
        float S[2][8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) { S[0][e] = S0[e]; S[1][e] = S1[e]; }
#if FA_KPREFETCH
        if (t0 + WAVE < kend) load_K(t0 + WAVE);      // next iteration's K rows, in flight during the two PV phases
#endif

#if FA_FENCE
        asm volatile("" ::: "memory");
#endif
#if !FA_V0_EARLY
#if FA_SYNC >= 3
        __syncthreads();
#endif
        load_tile(t0, 0, T0);
#endif
        process_tile(T0, S[0]);
#if FA_FENCE
        asm volatile("" ::: "memory");
#endif
#if FA_BRANCH
        if (t0 + WAVE <= kend)      // always true (chunk % 32 == 0): a real branch splits the scheduling regions, which keeps the register allocator sane
#endif
        {
#if FA_SYNC >= 3
            __syncthreads();
#endif
#if !FA_V1_EARLY
            Tile T1;
            load_tile(t0, 1, T1);
#endif
            process_tile(T1, S[1]);
        }
    }

    const float ltot = l + xchg16(l);
    if (col < NC) {
        const int gcol = hq*NQ + qi;         // global column = hq*NQ + qi (same as the VALU path and the combine)
        float * dst = p.part_acc + ((int64_t) c*NCOL + gcol)*D + 8*hh;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            ((float4 *) (dst + 16*e))[0] = make_float4(O[0][e], O[1][e], O[2][e], O[3][e]);
            ((float4 *) (dst + 16*e))[1] = make_float4(O[4][e], O[5][e], O[6][e], O[7][e]);
        }
        if (hh == 0) p.part_ml[(int64_t) c*NCOL + gcol] = make_float2(m, ltot);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// n_q >= 8, staged: the block stages the raw K/V rows of its HPB heads for 32 positions (plus the mask/index rows of the
// queries) in LDS with coalesced 16-byte loads shared by all its waves and prefetched one iteration ahead; each wave then
// reads its operands from LDS. QK uses the int8 WMMA on the raw K bytes and an int8 Q (per-32 scales as the vec kernel),
// PV the f16 WMMA as above. Same key tiling as fa_small_wmma (two 16-key tiles from the 32 rows via the half-lane property).
template <typename T, int NQ, int HPB>
__global__ __launch_bounds__(HPB*((G*NQ + 15)/16)*WAVE) void fa_small_stage(const Params p) {
    constexpr int ROW = ROW_BYTES<T>, NC = G*NQ, WPH = (NC + 15)/16, WAVES = HPB*WPH, THREADS = WAVES*WAVE, NCOL = HQ*NQ;
    constexpr int PROW = HPB*ROW;                       // bytes staged per position per tensor (the HPB heads' rows, contiguous)
    constexpr int STAGE = WAVE*PROW;                    // per tensor
    constexpr int CHUNKS = 2*STAGE/16, CPL = (CHUNKS + THREADS - 1)/THREADS;   // 16-byte chunks per iteration, per lane
    constexpr int MCH = NQ*4, ICH = NQ*8;               // mask / index chunks per iteration (64 B and 128 B per query row)
    static_assert(PROW % 16 == 0 && ROW % 8 == 0, "staging alignment");
    static_assert(MCH + ICH <= THREADS, "mask/index staging in one round");
    const int lane = threadIdx.x % WAVE, wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE), hl = wave / WPH, w = wave % WPH;
    const int h0 = blockIdx.y*HPB, h = h0 + hl, c = blockIdx.x, k0 = c*p.chunk;
    const int l16 = lane & 15, hh = lane >> 4;

    __shared__ int      q_i8[WAVES][16][D/4];           // per wave: its 16 columns' int8 Q, 16 B quads XOR-swizzled by (col & 7)
    __shared__ __attribute__((aligned(16))) char kv_s[2][STAGE];   // [K|V][pos][hl*ROW + byte]
    __shared__ __attribute__((aligned(16))) char mi_s[MCH*16 + ICH*16];  // mask rows [qi][32 halves], index rows [qi][32 ints]
    float * scratch = (float *) &kv_s[0][0];            // setup only: dq/qs per column (aliases the K/V stage, written later)
    static_assert(WAVES*16*8*4 <= STAGE, "dq/qs scratch fits in the stage");

    // ---- setup: this wave's 16 columns to int8 (as the vec kernel does)
    for (int cc = 0; cc < 16; ++cc) {
        const int col = 16*w + cc;
        float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (col < NC) { const int g = col/NQ, qi = col % NQ; x = ((const float4 *) (p.Q + qi*p.qs_q + (h*G + g)*p.qs_h))[lane]; }
        const float x0 = x.x*p.scale, x1 = x.y*p.scale, x2 = x.z*p.scale, x3 = x.w*p.scale;
        float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
        for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor(amax, o, WAVE));
        const float d = amax/127.0f, id = d > 0.0f ? 1.0f/d : 0.0f;
        const int b0 = (int) lrintf(x0*id), b1 = (int) lrintf(x1*id), b2 = (int) lrintf(x2*id), b3 = (int) lrintf(x3*id);
        q_i8[wave][cc][4*((lane/4) ^ (cc & 7)) + lane % 4] = (b0 & 0xff) | ((b1 & 0xff) << 8) | ((b2 & 0xff) << 16) | ((b3 & 0xff) << 24);
        int s = b0 + b1 + b2 + b3;
        for (int o = 1; o < 8; o <<= 1) s += __shfl_xor(s, o, WAVE);
        if (lane % 8 == 0) { scratch[(wave*16 + cc)*8 + lane/8] = d; scratch[(wave*16 + cc)*8 + 4 + lane/8] = __int_as_float(s); }
    }
    __syncthreads();
    float dq[NB]; int qs[NB];
    #pragma unroll
    for (int b = 0; b < NB; ++b) { dq[b] = scratch[(wave*16 + l16)*8 + b]; qs[b] = __float_as_int(scratch[(wave*16 + l16)*8 + 4 + b]); }
    __syncthreads();

    const int col = 16*w + l16, colc = min(col, NC - 1), qi = colc % NQ, g = colc/NQ, hq = h*G + g;
    const float * tbl = p.table + qi*p.table_stride_q + hq*p.n_bins;
    const int vb = l16/4, vo = T::Q4 ? vb*T::BLK + 2 + 8*((l16 % 4) % 2) : vb*T::BLK + 2 + 8*(l16 % 4);
    const bool vhi = T::Q4 && (l16 % 4) >= 2;
    const uint32_t sel_lo = hh ? 0x05040100u : 0x01000504u, sel_hi = hh ? 0x07060302u : 0x03020706u;
    auto interleave = [&](const uint32_t * x, const uint32_t * y, halfx16 & out) {
        union { uint32_t u[8]; halfx16 h; } r;
        #pragma unroll
        for (int i = 0; i < 4; ++i) { r.u[2*i] = __builtin_amdgcn_perm(x[i], y[i], sel_lo); r.u[2*i + 1] = __builtin_amdgcn_perm(x[i], y[i], sel_hi); }
        out = r.h;
    };
    auto launder = [](uint32_t v) { asm volatile("" : "+v"(v)); return v; };
    auto uniform_ptr = [](const char * ptr) {
        const uint64_t v = (uint64_t) ptr;
        const uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t) v), hi = __builtin_amdgcn_readfirstlane((uint32_t) (v >> 32));
        return (gptr) (((uint64_t) hi << 32) | lo);
    };

    // ---- cooperative staging: lane t loads chunks t, t + THREADS, ... of [K rows | V rows] of 32 positions, then mask/index
    const int tid = threadIdx.x;
    u32x4 R[CPL]; u32x4 RM = (u32x4) {0, 0, 0, 0};
    auto stage_load = [&](int t0) {
        const gptr Kb = uniform_ptr(p.K + (int64_t) t0*p.nb_pos + (int64_t) h0*p.nb_head);
        const gptr Vb = uniform_ptr(p.V + (int64_t) t0*p.nb_pos + (int64_t) h0*p.nb_head);
        #pragma unroll
        for (int i = 0; i < CPL; ++i) {
            const int ch = tid + i*THREADS;
            if (CHUNKS % THREADS == 0 || ch < CHUNKS) {
                const int tensor = ch/(CHUNKS/2), r = ch % (CHUNKS/2), pos = r/(PROW/16), off = (r % (PROW/16))*16;
                const uint32_t goff = launder((uint32_t) (pos*(int) p.nb_pos + off));
                R[i] = gload<u32x4>(tensor ? Vb : Kb, goff);
            }
        }
        if (tid < MCH) {            // mask: query qi = tid/4, 16 B chunk tid % 4 of its 32 keys
            const int q = tid/4, cc = tid % 4;
            RM = gload<u32x4>(uniform_ptr((const char *) (p.mask + t0)), launder((uint32_t) (q*p.mask_stride*2 + cc*16)));
        } else if (tid < MCH + ICH) {   // index: query (tid - MCH)/8, chunk (tid - MCH) % 8
            const int q = (tid - MCH)/8, cc = (tid - MCH) % 8;
            RM = gload<u32x4>(uniform_ptr((const char *) (p.idx + t0)), launder((uint32_t) (q*p.idx_stride*4 + cc*16)));
        }
    };
    auto stage_store = [&]() {
        #pragma unroll
        for (int i = 0; i < CPL; ++i) {
            const int ch = tid + i*THREADS;
            if (CHUNKS % THREADS == 0 || ch < CHUNKS) *(u32x4 *) (&kv_s[0][0] + ch*16) = R[i];
        }
        if (tid < MCH + ICH) *(u32x4 *) (mi_s + tid*16) = RM;
    };

    floatx8 O[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) O[t] = (floatx8) {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -1e30f, l = 0.0f;
    const int kend = min(k0 + p.chunk, p.n_kv);

    // one key tile (16 keys) from the staged data: keys t0 + 16*(hh ^ tile) + 2e + hh (see fa_small_wmma)
    auto process_tile = [&](int tile, float * S) {
        const int kofs = 16*(hh ^ tile) + hh;
        // bias: mask halves and bins of this lane's 8 keys from the staged rows (keys kofs + 2e)
        uint32_t mb[8]; int bin[8];
        {
            const uint32_t mo = launder(qi*64 + 32*(hh ^ tile)), io = launder(MCH*16 + qi*128 + 64*(hh ^ tile));
            const u32x4 * mrow = (const u32x4 *) (mi_s + mo);
            const u32x4 m0 = mrow[0], m1 = mrow[1];
            const uint32_t md[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
            #pragma unroll
            for (int e = 0; e < 8; ++e) mb[e] = hh ? (md[e] >> 16) : (md[e] & 0xffffu);
            const i32x4 * irow = (const i32x4 *) (mi_s + io);
            #pragma unroll
            for (int i = 0; i < 4; ++i) { const i32x4 v = irow[i]; bin[2*i] = hh ? v.y : v.x; bin[2*i + 1] = hh ? v.w : v.z; }
        }
        bool uni = true;
        #pragma unroll
        for (int e = 1; e < 8; ++e) uni = uni && bin[e] == bin[0];
        if (uni) {
            const bool valid = (uint32_t) bin[0] < (uint32_t) p.n_bins;
            const float tb = valid ? tbl[bin[0]] : 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) { const float mkf = half_bits(mb[e]); S[e] += valid ? __half2float(__float2half(mkf + tb)) : mkf; }
        } else {
            #pragma unroll
            for (int e = 0; e < 8; ++e) S[e] += bias_val(mb[e], bin[e], tbl, p.n_bins);
        }
        float mt = S[0];
        #pragma unroll
        for (int e = 1; e < 8; ++e) mt = fmaxf(mt, S[e]);
        mt = fmaxf(mt, xchg16(mt));
        const float mn = fmaxf(m, mt), alpha = __expf(m - mn);
        m = mn;
        float pr[8], ps = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) { pr[e] = __expf(S[e] - mn); ps += pr[e]; }
        l = l*alpha + ps;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) O[t] *= alpha;
        }
        halfx16 Pb;
        {
            uint32_t x[4], y[4];
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) x[e2] = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(pr[2*e2], pr[2*e2 + 1]));
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) y[e2] = xchg16(x[e2]);
            interleave(x, y, Pb);
        }
        // V of this lane's 8 keys from the staged rows: 8 bytes at a 2-byte aligned offset -> two aligned 8-byte reads
        float dv[8], ndv[8]; uint32_t vx[8][2];
        const uint32_t vrow0 = launder(kofs*PROW + hl*ROW);
        #pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            const char * row = &kv_s[1][0] + vrow0 + 2*jj*PROW;
            dv[jj]  = __half2float(*(const __half *) (row + vb*T::BLK));
            ndv[jj] = T::Q4 ? -8.0f*dv[jj] : -128.0f*dv[jj];
            const uint32_t * d = (const uint32_t *) (row + (vo & ~3));          // 4-byte aligned; vo & 3 == 2 or 0
            const uint32_t d0 = d[0], d1 = d[1], d2 = d[2];
            const uint32_t q0 = (vo & 3) ? __builtin_amdgcn_alignbyte(d1, d0, 2) : d0;
            const uint32_t q1 = (vo & 3) ? __builtin_amdgcn_alignbyte(d2, d1, 2) : d1;
            if constexpr (!T::Q4) { vx[jj][0] = q0 ^ 0x80808080u; vx[jj][1] = q1 ^ 0x80808080u; }
            else { vx[jj][0] = (vhi ? q0 >> 4 : q0) & 0x0F0F0F0Fu; vx[jj][1] = (vhi ? q1 >> 4 : q1) & 0x0F0F0F0Fu; }
        }
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            uint32_t x[4], y[4];
            #pragma unroll
            for (int j2 = 0; j2 < 4; ++j2) {
                half2_t r;
                r.x = (_Float16) fmaf((float) ((vx[2*j2][t/4] >> (8*(t % 4))) & 0xffu), dv[2*j2], ndv[2*j2]);
                r.y = (_Float16) fmaf((float) ((vx[2*j2 + 1][t/4] >> (8*(t % 4))) & 0xffu), dv[2*j2 + 1], ndv[2*j2 + 1]);
                x[j2] = h2_bits(r);
            }
            #pragma unroll
            for (int j2 = 0; j2 < 4; ++j2) y[j2] = xchg16(x[j2]);
            halfx16 Va;
            interleave(x, y, Va);
            O[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Va, Pb, O[t]);
            __builtin_amdgcn_sched_barrier(0);
        }
    };

    stage_load(k0);
    stage_store();
    __syncthreads();
    #pragma unroll 1
    for (int t0 = k0; t0 < kend; t0 += WAVE) {
        const bool more = t0 + WAVE < kend;
#if FA_PREFETCH_TOP
        if (more) stage_load(t0 + WAVE);            // next iteration's rows, in flight during this iteration's compute
#endif

        // ---- S^T = K.Q^T (int8 WMMA on the raw rows; lane = key); scaled per 32-block by dk (from the staged row) and dq
        uint32_t w_[ROW/4 + 1];
        {
            const u32x2 * krow = (const u32x2 *) (&kv_s[0][0] + launder(lane*PROW + hl*ROW));
            #pragma unroll
            for (int i = 0; i < ROW/8; ++i) { const u32x2 v = krow[i]; w_[2*i] = v.x; w_[2*i + 1] = v.y; }
            w_[ROW/4] = 0;
        }
        int sw = l16 & 7;
        asm volatile("" : "+v"(sw));
        float S[2][8];
        #pragma unroll
        for (int kt = 0; kt < 2; ++kt)
            #pragma unroll
            for (int e = 0; e < 8; ++e) S[kt][e] = 0.0f;
        #pragma unroll
        for (int b = 0; b < NB; ++b) {
            intx8 ia0 = (intx8) {0, 0, 0, 0, 0, 0, 0, 0}, ia1 = ia0;
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                intx4 own, oth;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if constexpr (!T::Q4) own[j] = (int) word_at(w_, b*T::BLK + 2 + 16*half + 4*j);
                    else { const uint32_t kq = word_at(w_, b*T::BLK + 2 + 4*j); own[j] = (int) ((half ? kq >> 4 : kq) & 0x0F0F0F0Fu); }
                    oth[j] = (int) xchg16((uint32_t) own[j]);
                }
                const intx4 bq = *(const intx4 *) &q_i8[wave][l16][4*((2*b + half) ^ sw)];
                ia0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(!T::Q4, own, true, bq, ia0, false);
                ia1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(!T::Q4, oth, true, bq, ia1, false);
                __builtin_amdgcn_sched_barrier(0);
            }
            // dk of the 16 output keys of this block: read after this block's WMMAs (the fake dependency on ia0 keeps the
            // scheduler from issuing all 64 reads at the top of the iteration, which costs 64 VGPRs and spills)
            // per-lane bases laundered per block (LICM would otherwise hoist 16 per-lane address VGPRs out of the loop and spill them)
            uint32_t kb0 = (hh*16 + hh)*PROW + hl*ROW + b*T::BLK, kb1 = ((hh ^ 1)*16 + hh)*PROW + hl*ROW + b*T::BLK;
            asm volatile("" : "+v"(kb0), "+v"(kb1) : "v"(ia0[0]));
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                int i0 = ia0[e], i1 = ia1[e];
                if constexpr (T::Q4) { i0 -= 8*qs[b]; i1 -= 8*qs[b]; }
#if FA_ABL_NODK
                const float dk0 = dq[b], dk1 = dq[(b + 1) % NB];
#elif FA_ABL_NODK2
                const float dk0 = __half2float(*(const __half *) (&kv_s[0][0] + kb0 + 2*e*0)), dk1 = __half2float(*(const __half *) (&kv_s[0][0] + kb1));
#else
                const float dk0 = __half2float(*(const __half *) (&kv_s[0][0] + kb0 + 2*e*PROW));    // tile 0 key 16hh + 2e + hh
                const float dk1 = __half2float(*(const __half *) (&kv_s[0][0] + kb1 + 2*e*PROW));    // tile 1 key 16(hh^1) + 2e + hh
#endif
                S[0][e] += (float) i0*dq[b]*dk0;
                S[1][e] += (float) i1*dq[b]*dk1;
            }
        }
#if !FA_PREFETCH_TOP && !FA_ABL_NOPREFETCH
        if (more) stage_load(t0 + WAVE);            // next iteration's rows, in flight during the two PV phases
#endif
#if !FA_ABL_STNOPV
        process_tile(0, S[0]);
        if (t0 + WAVE <= kend) {       // always true: splits the scheduling regions (see fa_small_wmma)
            process_tile(1, S[1]);
        }
#else
        #pragma unroll
        for (int e = 0; e < 8; ++e) O[e][0] += S[0][e] + S[1][e];
#endif
        __syncthreads();               // everyone is done with this stage
#if FA_ABL_NOPREFETCH
        if (more) stage_load(t0 + WAVE);
#endif
        if (more) stage_store();
        __syncthreads();
    }

    const float ltot = l + xchg16(l);
    if (col < NC) {
        const int gcol = hq*NQ + qi;
        float * dst = p.part_acc + ((int64_t) c*NCOL + gcol)*D + 8*hh;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            ((float4 *) (dst + 16*e))[0] = make_float4(O[0][e], O[1][e], O[2][e], O[3][e]);
            ((float4 *) (dst + 16*e))[1] = make_float4(O[4][e], O[5][e], O[6][e], O[7][e]);
        }
        if (hh == 0) p.part_ml[(int64_t) c*NCOL + gcol] = make_float2(m, ltot);
    }
}

// grid.x = NQ*HQ columns (col = hq*NQ + qi), block = D: combine the chunks and the sink logit of the head
template <int NQ>
__global__ void fa_combine(const float * __restrict__ part_acc, const float2 * __restrict__ part_ml, int chunks,
        const float * __restrict__ sinks, float * __restrict__ out) {
    constexpr int NCOL = HQ*NQ;
    const int col = blockIdx.x, d = threadIdx.x, hq = col/NQ, qi = col % NQ;
    const float sink = sinks ? sinks[hq] : -INFINITY;
    float M = sinks ? sink : -1e30f;
    for (int c = 0; c < chunks; ++c) M = fmaxf(M, part_ml[(int64_t) c*NCOL + col].x);
    float L = sinks ? __expf(sink - M) : 0.0f, acc = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float2 ml = part_ml[(int64_t) c*NCOL + col];
        const float w = __expf(ml.x - M);
        L += ml.y*w; acc += part_acc[((int64_t) c*NCOL + col)*D + d]*w;
    }
    out[((int64_t) qi*HQ + hq)*D + d] = acc/L;
}

// ---- host: quantizers and dequantizers matching ggml's reference ----------------------------------------------------
template <typename T> static void quantize_row(const float * x, char * y) {
    for (int b = 0; b < NB; ++b) {
        char * blk = y + b*T::BLK;
        if constexpr (!T::Q4) {
            float amax = 0; for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(x[b*QK + i]));
            const float d = amax/127.0f, id = d ? 1.0f/d : 0.0f;
            *(__half *) blk = __float2half(d);
            for (int i = 0; i < QK; ++i) ((int8_t *) (blk + 2))[i] = (int8_t) std::lrint(x[b*QK + i]*id);
        } else {   // ggml quantize_row_q4_0_ref: d = max/-8 by signed max
            float amax = 0, mx = 0;
            for (int i = 0; i < QK; ++i) { const float v = x[b*QK + i]; if (std::fabs(v) > amax) { amax = std::fabs(v); mx = v; } }
            const float d = mx/-8.0f, id = d ? 1.0f/d : 0.0f;
            *(__half *) blk = __float2half(d);
            for (int i = 0; i < QK/2; ++i) {
                const uint8_t a = (uint8_t) std::min(15.0f, x[b*QK + i]*id + 8.5f), c = (uint8_t) std::min(15.0f, x[b*QK + 16 + i]*id + 8.5f);
                ((uint8_t *) (blk + 2))[i] = a | (c << 4);
            }
        }
    }
}
template <typename T> static float deq(const char * y, int i) {
    const char * blk = y + (i/QK)*T::BLK; const float d = __half2float(*(const __half *) blk); const int j = i % QK;
    if constexpr (!T::Q4) return d*((const int8_t *) (blk + 2))[j];
    else { const uint8_t q = ((const uint8_t *) (blk + 2))[j % 16]; return d*((int) (j < 16 ? q & 15 : q >> 4) - 8); }
}

// cheap deterministic pseudo-random: uniform in [-1, 1) and an approximately normal variate (sum of 4 uniforms)
static inline uint64_t mix64(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }
static inline float unif(uint64_t s) { return (float) ((mix64(s) >> 11)*(1.0/9007199254740992.0))*2.0f - 1.0f; }
static inline float nrm(uint64_t s) { return unif(s*4 + 0) + unif(s*4 + 1) + unif(s*4 + 2) + unif(s*4 + 3); }

template <typename F> static void par_for(int n, F f) {
    const int nt = std::min(n, std::max(1, (int) std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) th.emplace_back([=]() { for (int i = t; i < n; i += nt) f(i); });
    for (auto & x : th) x.join();
}

template <typename T, int NQ> static void run_case(int n_kv, const std::vector<int> & chunks_to_try, bool use_sinks, int reps) {
    constexpr int ROW = ROW_BYTES<T>, NCOL = HQ*NQ;
    const int n_bins = 1025; const float scale = 1.0f/sqrtf((float) D);
    const int64_t nb_head = ROW, nb_pos = (int64_t) HKV*ROW;
    const double kv_bytes = 2.0*n_kv*nb_pos;
    printf("== %s n_kv %d n_q %d: K+V %.1f MB, sinks %s\n", T::Q4 ? "q4_0" : "q8_0", n_kv, NQ, kv_bytes/1e6, use_sinks ? "nonzero" : "none");

    std::vector<char> hK((size_t) n_kv*nb_pos), hV((size_t) n_kv*nb_pos);
    par_for(n_kv, [&](int k) {
        float tmp[D];
        for (int h = 0; h < HKV; ++h) {
            for (int i = 0; i < D; ++i) tmp[i] = nrm(((uint64_t) k*HKV + h)*D + i)*(i % 17 == 0 ? 3.0f : 1.0f);
            quantize_row<T>(tmp, hK.data() + k*nb_pos + h*nb_head);
            for (int i = 0; i < D; ++i) tmp[i] = nrm((((uint64_t) k*HKV + h)*D + i) ^ 0x5555555555ULL);
            quantize_row<T>(tmp, hV.data() + k*nb_pos + h*nb_head);
        }
    });
    // Q in the llama.cpp view: element (d, qi, hq) at qi*qs_q + hq*qs_h + d, qs_h = D, qs_q = HQ*D
    const int64_t qs_h = D, qs_q = (int64_t) HQ*D;
    std::vector<float> hQ((size_t) NQ*HQ*D), hT((size_t) NQ*HQ*n_bins), hS(HQ);
    for (size_t i = 0; i < hQ.size(); ++i) hQ[i] = nrm(0x1000000 + i)*1.6f;      // sum of 4 uniforms ~ N(0, 1.15): logits of a few units
    for (size_t i = 0; i < hT.size(); ++i) hT[i] = nrm(0x2000000 + i)*0.5f;
    for (int hq = 0; hq < HQ; ++hq) hS[hq] = use_sinks ? nrm(0x3000000 + hq)*1.5f : 0.0f;
    // mask: the NQ queries are the last NQ positions, causal among themselves; indices as in the model: bin = min(distance, 1024)
    // (so the far field is one uniform bin), plus -- to exercise every path -- invalid bins (negative / >= n_bins) and random
    // near bins sprinkled over a 2048-key window in the middle (IDX_SPRINKLE=all: over the whole range, slow gather path everywhere)
    const int64_t mask_stride = n_kv, idx_stride = n_kv;
    const bool sprinkle_all = getenv("IDX_SPRINKLE") && std::string(getenv("IDX_SPRINKLE")) == "all";
    const int sp0 = std::max(0, n_kv/2 - 1024), sp1 = std::min(n_kv, sp0 + 2048);
    std::vector<__half> hM((size_t) NQ*mask_stride); std::vector<int32_t> hI((size_t) NQ*idx_stride);
    for (int qi = 0; qi < NQ; ++qi) {
        const int pos = n_kv - NQ + qi;
        for (int k = 0; k < n_kv; ++k) {
            const int dist = pos - k;
            hM[qi*mask_stride + k] = __float2half(k <= pos ? 0.0f : -INFINITY);
            int bin = dist < 0 ? -1 : std::min(dist, 1024);
            const uint64_t r = mix64(((uint64_t) qi << 40) ^ k ^ 0x77);
            if (sprinkle_all || (k >= sp0 && k < sp1)) {
                if (r % 97 == 0) bin = (r >> 8) % 2 ? -(int) ((r >> 9) % 1000) - 1 : n_bins + (int) ((r >> 9) % 1000);
                else if (r % 7 == 0) bin = (int) ((r >> 9) % 1024);
            }
            hI[qi*idx_stride + k] = bin;
        }
    }

    // double-precision reference; REF_QUANT=1 feeds it the Q the kernel actually uses (q8_0 blocks for n_q = 1 and the staged
    // path, f16 otherwise)
    const bool staged = NQ >= 8 && getenv("STAGE") && atoi(getenv("STAGE")) != 0;   // experimental, slower (see NOTES.md)
    std::vector<float> hQk(hQ);
    if (getenv("REF_QUANT")) {
        for (int qi = 0; qi < NQ; ++qi) for (int hq = 0; hq < HQ; ++hq) {
            float * q = hQk.data() + qi*qs_q + hq*qs_h;
            if (NQ <= 1 || staged) {
                for (int b = 0; b < NB; ++b) {
                    float amax = 0; for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(q[b*QK + i]*scale));
                    const float d = amax/127.0f, id = d > 0 ? 1.0f/d : 0.0f;
                    for (int i = 0; i < QK; ++i) q[b*QK + i] = d*(float) std::lrint(q[b*QK + i]*scale*id)/scale;
                }
            } else {
                for (int i = 0; i < D; ++i) q[i] = __half2float(__float2half(q[i]*scale))/scale;
            }
        }
    }
    std::vector<double> ref((size_t) NCOL*D);
    par_for(NCOL, [&](int col) {
        const int hq = col/NQ, qi = col % NQ, h = hq/G;
        std::vector<double> s(n_kv), o(D, 0.0);
        double M = use_sinks ? hS[hq] : -1e300;
        const float * q = hQk.data() + qi*qs_q + hq*qs_h;
        const float * tbl = hT.data() + ((size_t) qi*HQ + hq)*n_bins;
        for (int k = 0; k < n_kv; ++k) {
            const char * kr = hK.data() + (size_t) k*nb_pos + h*nb_head;
            double dot = 0; for (int i = 0; i < D; ++i) dot += (double) q[i]*scale*deq<T>(kr, i);
            const float mk = __half2float(hM[qi*mask_stride + k]); const int bin = hI[qi*idx_stride + k];
            const float bias = (uint32_t) bin < (uint32_t) n_bins ? __half2float(__float2half(mk + tbl[bin])) : mk;
            s[k] = dot + bias; M = std::max(M, s[k]);
        }
        double L = use_sinks ? std::exp(hS[hq] - M) : 0.0;
        for (int k = 0; k < n_kv; ++k) {
            if (s[k] == -INFINITY) continue;
            const double pr = std::exp(s[k] - M); L += pr; const char * vr = hV.data() + (size_t) k*nb_pos + h*nb_head;
            for (int i = 0; i < D; ++i) o[i] += pr*deq<T>(vr, i);
        }
        for (int i = 0; i < D; ++i) ref[(size_t) col*D + i] = o[i]/L;
    });

    int min_chunk = 1 << 30; for (int ch : chunks_to_try) min_chunk = std::min(min_chunk, ch);
    const int max_chunks = (n_kv + min_chunk - 1)/min_chunk;
    float *dQ, *dT, *dS, *dPA, *dO; char *dK, *dV; __half *dM; int32_t *dI; float2 *dML;
    CHECK(hipMalloc(&dQ, hQ.size()*4)); CHECK(hipMalloc(&dT, hT.size()*4)); CHECK(hipMalloc(&dS, HQ*4));
    CHECK(hipMalloc(&dK, hK.size())); CHECK(hipMalloc(&dV, hV.size()));
    CHECK(hipMalloc(&dM, hM.size()*2)); CHECK(hipMalloc(&dI, hI.size()*4));
    CHECK(hipMalloc(&dPA, (size_t) max_chunks*NCOL*D*4)); CHECK(hipMalloc(&dML, (size_t) max_chunks*NCOL*8)); CHECK(hipMalloc(&dO, (size_t) NCOL*D*4));
    CHECK(hipMemcpy(dQ, hQ.data(), hQ.size()*4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dT, hT.data(), hT.size()*4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dS, hS.data(), HQ*4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dK, hK.data(), hK.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dV, hV.data(), hV.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dM, hM.data(), hM.size()*2, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dI, hI.data(), hI.size()*4, hipMemcpyHostToDevice));

    Params p;
    p.Q = dQ; p.qs_q = qs_q; p.qs_h = qs_h; p.scale = scale;
    p.K = dK; p.V = dV; p.nb_pos = nb_pos; p.nb_head = nb_head;
    p.mask = dM; p.mask_stride = mask_stride; p.idx = dI; p.idx_stride = idx_stride;
    p.table = dT; p.table_stride_q = (int64_t) HQ*n_bins; p.n_bins = n_bins;
    p.n_kv = n_kv; p.part_acc = dPA; p.part_ml = dML;

    auto bench = [&](int chunk, int hpb) {
        if (chunk % 32 != 0) { printf("  chunk %d not a multiple of 32\n", chunk); return; }
        if (staged && hpb != 4) return;
        if (staged && (nb_head != ROW || mask_stride % 8 != 0 || idx_stride % 4 != 0)) { printf("  staged path needs contiguous heads and 16 B-aligned mask/index rows\n"); return; }
        if (NQ <= 1 && hpb != 8 && hpb != 4) return;
        if (NQ > 1 && n_kv % 32 != 0) { printf("  the WMMA path needs n_kv %% 32 == 0\n"); return; }
        p.chunk = chunk;
        const int chunks = (n_kv + chunk - 1)/chunk;
        const dim3 grid(chunks, HKV/hpb);
        auto run = [&]() {
            if constexpr (NQ <= 1) {
                if (hpb == 8) fa_small_valu<T, NQ, 8><<<grid, 8*WAVE>>>(p); else fa_small_valu<T, NQ, 4><<<grid, 4*WAVE>>>(p);
            } else if (staged) {
                constexpr int WPH = (G*NQ + 15)/16;
                if constexpr (NQ >= 8) { if (hpb == 4) fa_small_stage<T, NQ, 4><<<grid, 4*WPH*WAVE>>>(p); }
            } else {
                constexpr int WPH = (G*NQ + 15)/16;
                if constexpr (8*WPH <= 16) { if (hpb == 8) fa_small_wmma<T, NQ, 8><<<grid, 8*WPH*WAVE>>>(p); }
                if constexpr (4*WPH <= 16) { if (hpb == 4) fa_small_wmma<T, NQ, 4><<<grid, 4*WPH*WAVE>>>(p); }
                if (hpb == 2) fa_small_wmma<T, NQ, 2><<<grid, 2*WPH*WAVE>>>(p);
            }
            fa_combine<NQ><<<NCOL, D>>>(dPA, dML, chunks, use_sinks ? dS : nullptr, dO);
        };
        run(); CHECK(hipDeviceSynchronize()); CHECK(hipGetLastError());
        std::vector<float> out((size_t) NCOL*D); CHECK(hipMemcpy(out.data(), dO, out.size()*4, hipMemcpyDeviceToHost));
        double max_err = 0, max_ref = 0; int nan = 0; int worst_col = -1;
        for (int col = 0; col < NCOL; ++col) for (int i = 0; i < D; ++i) {       // out is [qi][hq][d], ref/col = hq*NQ + qi
            const double r = ref[(size_t) col*D + i], v = out[((size_t) (col % NQ)*HQ + col/NQ)*D + i];
            if (!std::isfinite(v)) ++nan;
            const double e = std::fabs(v - r);
            if (e > max_err) { max_err = e; worst_col = col; }
            max_ref = std::max(max_ref, std::fabs(r));
        }
        hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
        for (int w = 0; w < 3; ++w) run();
        CHECK(hipEventRecord(a)); for (int r = 0; r < reps; ++r) run(); CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b));
        float ms; CHECK(hipEventElapsedTime(&ms, a, b)); ms /= reps;
        CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(b));
        printf("  %s chunk %4d hpb %d (%5d blocks x %3d waves): %7.1f us, %6.1f GB/s K+V, max err %.3g (rel %.2g, col %d = hq %d qi %d)%s\n",
               NQ <= 1 ? "valu " : staged ? "stage" : "wmma ", chunk, hpb, chunks*(HKV/hpb), (NQ <= 1 ? 1 : (G*NQ + 15)/16)*hpb, ms*1e3, kv_bytes/(ms*1e-3)/1e9,
               max_err, max_err/max_ref, worst_col, worst_col/NQ, worst_col % NQ, nan ? "  NAN!" : "");
    };
    constexpr int WPH = NQ <= 1 ? 1 : (G*NQ + 15)/16;
    for (int ch : chunks_to_try) for (int hpb : {8, 4, 2}) if (hpb*WPH <= 16 && (hpb == 8 || NQ > 1)) bench(ch, hpb);
    hipFree(dQ); hipFree(dT); hipFree(dS); hipFree(dK); hipFree(dV); hipFree(dM); hipFree(dI); hipFree(dPA); hipFree(dML); hipFree(dO);
}

template <typename T> static void run_type(int n_kv, int n_q, const std::vector<int> & chunks, bool sinks, int reps) {
    switch (n_q) {
        case 1:  run_case<T, 1 >(n_kv, chunks, sinks, reps); break;
        case 2:  run_case<T, 2 >(n_kv, chunks, sinks, reps); break;
        case 4:  run_case<T, 4 >(n_kv, chunks, sinks, reps); break;
        case 8:  run_case<T, 8 >(n_kv, chunks, sinks, reps); break;
        case 16: run_case<T, 16>(n_kv, chunks, sinks, reps); break;
        default: printf("n_q must be one of 1 2 4 8 16\n");
    }
}

int main(int argc, char ** argv) {
    const std::string type = argc > 1 ? argv[1] : "q8_0";
    const int n_kv = argc > 2 ? atoi(argv[2]) : 131072;
    const int n_q  = argc > 3 ? atoi(argv[3]) : 4;
    std::vector<int> chunks;
    for (int i = 4; i < argc; ++i) chunks.push_back(atoi(argv[i]));
    if (chunks.empty()) chunks = {512, 1024};
    const bool sinks = getenv("NO_SINKS") == nullptr;
    const int reps = getenv("REPS") ? atoi(getenv("REPS")) : 20;
    if (type == "q4_0") run_type<q4_0_t>(n_kv, n_q, chunks, sinks, reps); else run_type<q8_0_t>(n_kv, n_q, chunks, sinks, reps);
    return 0;
}
