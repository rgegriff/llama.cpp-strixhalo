// Standalone prototype: Inkling 2 Mini PREFILL attention with WMMA on gfx1151 (RDNA3.5, wave32).
//
// Semantics follow ggml's FLASH_ATTN_EXT with the Inkling indexed bias (see ggml-cuda/fattn-common.cuh):
//   logit(q,h,k) = scale*(Q.K) + float(bias_h(q,h,k)),  bias_h = bin valid ? half(float(mask[q][k]) + table[q][h][bin]) : mask[q][k]
//   softmax over k with an extra sink term exp(sink[h] - max) in the denominator (no value), O = P.V.
// Q: f32 [n_q][24][128] (ggml memory order), K/V: f16 in the real cache layout (per position the 8 KV heads' rows back to back),
// mask f16 [n_q][n_kv], indices i32 [n_q][n_kv], table f32 [n_q][24][n_bins], out f32 [n_q][24][128]. GQA: Q head hq -> KV head hq/3.
//
// Design: flash-attention-2. Workgroup = (QT queries, one KV head, one KV split); wave = 16 queries x 1 Q head, so the
// 3 Q heads of a KV head share the K/V tiles staged in LDS (K as [key][d], V transposed to [d][key] with an 8x8 register
// transpose on the way in). S^T = K.Q^T and O^T = V^T.P^T with the raw gfx11 WMMA builtin (16x16x16 f16 -> f32), chosen so that
// every lane holds exactly one query row: the online softmax needs one cross-lane op per 64 keys.
// The bias is handled per (16-query, 64-key) tile with a precomputed flag: 0 = gather per element, 1 = per-row constant
// (indices and mask constant along the tile's rows, which is the case for all keys farther than the relative-position extent),
// 2 = all 16 rows masked out (skipped: the causal triangle). The flags pass is exact for arbitrary indices/mask.
//
//   ./fa_prefill <n_kv> <n_q> [check|bench|both] [qt=64] [splits=1] [maskmode=causal|random]
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int D = 128, HKV = 8, G = 3, HQ = HKV*G, WAVE = 32;
constexpr int KT = 64;                    // keys per LDS tile (one softmax step)
constexpr int NSUB = KT/16;               // 16-key WMMA sub-tiles per tile
// LDS tile layout: FA_PAD=1 pads each row by 16 B against bank conflicts (35 KB per tile set, single-buffered);
// FA_PAD=0 keeps rows unpadded and XOR-swizzles their 16 B chunks instead (32 KB), which allows FA_DBUF=1: two buffers in
// exactly 64 KB and one barrier per tile. Measured (kv 65536, qt 128, 2 splits): PAD=1 24-25.6 ms, PAD=0 single 30-32 ms,
// PAD=0 double-buffered 30-36 ms, so the padding wins and the double buffer is kept only as a documented experiment.
#ifndef FA_PAD
#define FA_PAD 1
#endif
#ifndef FA_DBUF
#define FA_DBUF 0
#endif
constexpr int K_LD16 = D*2/16 + FA_PAD;   // K row in LDS: 16 (+1) x 16 B
constexpr int V_LD16 = KT*2/16 + FA_PAD;  // V^T row in LDS: 8 (+1) x 16 B
constexpr int LDS_K = KT*K_LD16, LDS_V = D*V_LD16;   // in uint4
constexpr int NBUF = (FA_DBUF && !FA_PAD) ? 2 : 1;
static_assert(NBUF*(LDS_K + LDS_V)*16 <= 65536, "LDS");
#ifndef FA_LOOKAHEAD
#define FA_LOOKAHEAD 2
#endif
#ifndef FA_STEPS
#define FA_STEPS 2
#endif
#ifndef FA_SCHED
#define FA_SCHED 1
#endif
// ablations for timing experiments only (results are wrong): drop LDS fragment reads / the WMMA op / the K,V tile traffic
#ifndef FA_ABL_NOLDS
#define FA_ABL_NOLDS 0
#endif
#ifndef FA_ABL_NOWMMA
#define FA_ABL_NOWMMA 0
#endif
#ifndef FA_ABL_NOGLOBAL
#define FA_ABL_NOGLOBAL 0
#endif
#ifndef FA_ABL_NOBARRIER
#define FA_ABL_NOBARRIER 0
#endif

typedef __attribute__((ext_vector_type(16))) _Float16 halfx16;
typedef __attribute__((ext_vector_type(8)))  float    floatx8;

__device__ __forceinline__ halfx16 ld_frag(const uint4 * p) {        // 32 contiguous bytes = 16 halves (one WMMA A/B operand)
    union { uint4 u[2]; halfx16 h; } r; r.u[0] = p[0]; r.u[1] = p[1]; return r.h;
}
// Bank-conflict avoidance without padding: a row's 16 B chunks are XOR-swizzled by an even amount, so a fragment's two
// chunks stay adjacent. K rows (256 B): readers take 16 consecutive rows -> spread by row % 8. V^T rows (128 B): readers take
// 16 consecutive rows, the transposing writers 16 rows 8 apart -> spread by both row % 4 and (row/8) % 4.
// With padded rows the readers are conflict-free by the padding; only the V^T writers (rows 8 apart = 9 x 128 B) need a swizzle.
__device__ __forceinline__ int kswz(int row) { return FA_PAD ? 0 : 2*(row % 8); }
__device__ __forceinline__ int vswz(int row) { return FA_PAD ? (row/8) % 8 : 2*((row % 4) ^ ((row/8) % 4)); }
__device__ __forceinline__ halfx16 ld_frag_s(const uint4 * row, int chunk, int swz) {   // chunks chunk, chunk+1 of a swizzled row
    union { uint4 u[2]; halfx16 h; } r; r.u[0] = row[chunk ^ swz]; r.u[1] = row[(chunk + 1) ^ swz]; return r.h;
}
__device__ __forceinline__ floatx8 wmma(const halfx16 & a, const halfx16 & b, const floatx8 & c) {
    // D[i][j] = sum_k a_lane(i)[k] * b_lane(j)[k]; lane l holds D[2e + l/16][l%16] (probed: wmma_probe.hip)
#if FA_ABL_NOWMMA      // ablation (timing only): keep the dependency chain, drop the matrix op
    floatx8 r = c; r[0] += (float) a[0]*(float) b[0]; return r;
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}
// ask the scheduler for: 2*LA ds_reads up front, then (1 WMMA, 2 ds_reads) x (8-LA), then LA WMMAs (one 16-key sub-tile)
__device__ __forceinline__ void sched_pipeline() {
#if FA_SCHED
    constexpr int LA = FA_LOOKAHEAD;
    __builtin_amdgcn_sched_group_barrier(0x100, 2*LA, 0);
    #pragma unroll
    for (int i = 0; i < D/16 - LA; ++i) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
    }
    __builtin_amdgcn_sched_group_barrier(0x008, LA, 0);
#endif
}
__device__ __forceinline__ float bias_val(__half mk, int bin, const float * tbl, int n_bins) {    // fattn_indexed_bias::add
    return (uint32_t) bin < (uint32_t) n_bins ? __half2float(__float2half(__half2float(mk) + tbl[bin])) : __half2float(mk);
}
// value of lane^16: v_permlanex16 is a VALU op; __shfl_xor compiles to ds_bpermute, an LDS round trip on the critical path
__device__ __forceinline__ uint32_t xchg16(uint32_t v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false); }
__device__ __forceinline__ float xchg16(float v) { return __uint_as_float(xchg16(__float_as_uint(v))); }
__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    return (uint32_t) __half_as_ushort(__float2half(a)) | ((uint32_t) __half_as_ushort(__float2half(b)) << 16);
}

// flags[kt][qt]: 0 gather, 1 per-row constant (indices and mask constant along each of the 16 rows), 2 constant and all rows -inf
__global__ void fa_flags(const __half * __restrict__ mask, const int32_t * __restrict__ idx, int n_kv, uint8_t * __restrict__ flags) {
    const int kt = blockIdx.x, qt = blockIdx.y, n_kt = gridDim.x, t = threadIdx.x;     // 256 threads: 16 rows x 16 groups of 4 keys
    const int r = t/16, q = qt*16 + r, k = kt*KT + (t % 16)*4;
    const int4 bi = *(const int4 *) (idx + (int64_t) q*n_kv + k);
    const uint2 mu = *(const uint2 *) (mask + (int64_t) q*n_kv + k);
    const int ref_i = idx[(int64_t) q*n_kv + kt*KT];
    const float ref_m = __half2float(mask[(int64_t) q*n_kv + kt*KT]);
    bool uni = bi.x == ref_i && bi.y == ref_i && bi.z == ref_i && bi.w == ref_i;
    uni = uni && __half2float(__ushort_as_half(mu.x & 0xffff)) == ref_m && __half2float(__ushort_as_half(mu.x >> 16)) == ref_m
              && __half2float(__ushort_as_half(mu.y & 0xffff)) == ref_m && __half2float(__ushort_as_half(mu.y >> 16)) == ref_m;
    const bool all_uni = __syncthreads_and(uni);
    const bool all_inf = __syncthreads_and(ref_m == -INFINITY);
    if (t == 0) flags[(int64_t) kt*gridDim.y + qt] = !all_uni ? 0 : (all_inf ? 2 : 1);
}

template <int QT>
__global__ __launch_bounds__(QT/16*G*WAVE) void fa_prefill(
        const float * __restrict__ Q, const __half * __restrict__ K, const __half * __restrict__ V, int64_t kv_pos_stride,
        const __half * __restrict__ mask, const int32_t * __restrict__ idx, const float * __restrict__ table, int n_bins,
        const uint8_t * __restrict__ flags, int n_q, int n_kv, int kt_per_split, float scale,
        float * __restrict__ part_O, float2 * __restrict__ part_ml) {
    constexpr int NW = QT/16*G, NT = NW*WAVE;
    // tile prefetch roles (wave-uniform): with >= 768 threads the first 256 load the V tasks and the rest the K chunks,
    // so no thread holds more than 16 prefetch VGPRs; smaller workgroups share both
    constexpr int NVT = KT/4*16;                        // 4-key x 8-d V tasks per tile (256)
    constexpr bool ROLES = NT >= 768;
    constexpr int KT0 = ROLES ? NVT : 0, NKT = NT - KT0;            // first thread and thread count for K chunks
    constexpr int NKLD = (KT*16 + NKT - 1)/NKT;                      // 16 B K chunks per K thread per tile
    constexpr int NVLD = (NVT + NT - 1)/NT;                          // V tasks per thread per tile
    constexpr int STEPS = FA_STEPS, SUBS = NSUB/STEPS;  // softmax steps per tile, 16-key sub-tiles per step
    constexpr int LA = QT >= 128 ? FA_LOOKAHEAD : 1;    // LDS fragments in flight ahead of the WMMA that consumes them
    __shared__ uint4 lds[NBUF*(LDS_K + LDS_V)];

    const int tid = threadIdx.x, lane = tid % WAVE, w = tid / WAVE, l16 = lane % 16, hh = lane / 16;
    const int hk = blockIdx.z, split = blockIdx.y, qbase = blockIdx.x*QT;
    const int g = w % G, hq = hk*G + g, q0 = qbase + (w/G)*16, q = q0 + l16;
    const int n_kt = n_kv/KT;
    const int kt_begin = split*kt_per_split, kt_end = min(kt_begin + kt_per_split, n_kt);
    const float * tbl = table + (int64_t) q*HQ*n_bins + hq*n_bins;
    const __half   * mrow = mask + (int64_t) q*n_kv;
    const int32_t  * irow = idx  + (int64_t) q*n_kv;
    const __half * Kh = K + hk*D, * Vh = V + hk*D;

    // Q as WMMA B operands (lane holds query row q, 16 d's per fragment), scaled and rounded to f16 like llama.cpp
    halfx16 Qb[D/16];
    {
        const float4 * qp = (const float4 *) (Q + ((int64_t) q*HQ + hq)*D);
        #pragma unroll
        for (int dt = 0; dt < D/16; ++dt) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float4 x = qp[dt*4 + c];
                Qb[dt][4*c + 0] = (_Float16) (x.x*scale); Qb[dt][4*c + 1] = (_Float16) (x.y*scale);
                Qb[dt][4*c + 2] = (_Float16) (x.z*scale); Qb[dt][4*c + 3] = (_Float16) (x.w*scale);
            }
        }
    }
    floatx8 O[D/16];
    #pragma unroll
    for (int dt = 0; dt < D/16; ++dt) O[dt] = (floatx8) {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -1e30f, l = 0.0f;

    // the next tile that is not skipped by the whole workgroup (all QT/16 flags == 2); kt_end if none
    const int n_qt = n_q/16;
    typedef typename std::conditional<QT == 32, uint16_t, typename std::conditional<QT == 64, uint32_t, uint64_t>::type>::type flagword_t;
    constexpr flagword_t ALL2 = (flagword_t) 0x0202020202020202ull;
    auto flagword = [&](int kt) { return *(const flagword_t *) (flags + (int64_t) kt*n_qt + qbase/16); };
    flagword_t fw = 0;                     // the workgroup's QT/16 flags of the tile next_tile() returned
    auto next_tile = [&](int kt, flagword_t w) {      // w = flags of kt, already loaded
        for (; kt < kt_end; ++kt) {
            if (w != ALL2) break;
            if (kt + 1 < kt_end) w = flagword(kt + 1);
        }
        fw = w;
        return kt;
    };
    // c < limit is the same for all lanes of a wave (NT, limit are multiples of 32): say so, or the compiler masks the
    // load and merges its result immediately, which defeats the prefetch
    auto uni = [](int c, int limit) { return __builtin_amdgcn_readfirstlane(c) < limit; };
    uint4 pk[NKLD]; uint32_t pv[NVLD][4][4];
    __half mk_nx; int bin_nx;               // the row's mask/bin at the first key of the prefetched tile (uniform tiles)
    flagword_t fw_nx = ALL2;                // flags of the tile after the prefetched one
    auto prefetch = [&](int kt) {        // global loads of tile kt into registers
        if (kt + 1 < kt_end) fw_nx = flagword(kt + 1);
        if (FA_ABL_NOGLOBAL) return;
        const int kbase = kt*KT;
        mk_nx = mrow[kbase]; bin_nx = irow[kbase];
        #pragma unroll
        for (int i = 0; i < NKLD; ++i) {
            const int c = tid - KT0 + i*NKT, key = c/16, dc = c % 16;
            if (uni(c, KT*16) && (!ROLES || __builtin_amdgcn_readfirstlane(tid) >= KT0))
                pk[i] = *(const uint4 *) (Kh + (int64_t) (kbase + key)*kv_pos_stride + dc*8);
        }
        #pragma unroll
        for (int t = 0; t < NVLD; ++t) {
            const int c = tid + t*NT, dc = c % 16, kg = c/16;
            if (uni(c, KT/4*16)) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 x = *(const uint4 *) (Vh + (int64_t) (kbase + kg*4 + i)*kv_pos_stride + dc*8);
                    pv[t][i][0] = x.x; pv[t][i][1] = x.y; pv[t][i][2] = x.z; pv[t][i][3] = x.w;
                }
            }
        }
    };
    auto store_lds = [&](int buf) {      // registers -> LDS buffer: K as [key][d], V transposed to [d][key]
        if (FA_ABL_NOGLOBAL) return;
        uint4 * Ks = lds + buf*(LDS_K + LDS_V);
        #pragma unroll
        for (int i = 0; i < NKLD; ++i) {
            const int c = tid - KT0 + i*NKT, key = c/16, dc = c % 16;
            if (uni(c, KT*16) && (!ROLES || __builtin_amdgcn_readfirstlane(tid) >= KT0)) Ks[key*K_LD16 + (dc ^ kswz(key))] = pk[i];
        }
        uint2 * dst = (uint2 *) (Ks + LDS_K);
        #pragma unroll
        for (int t = 0; t < NVLD; ++t) {
            const int c = tid + t*NT, dc = c % 16, kg = c/16;
            if (uni(c, KT/4*16)) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t sel = (j & 1) ? 0x07060302u : 0x05040100u;
                    const uint32_t a = __builtin_amdgcn_perm(pv[t][1][j/2], pv[t][0][j/2], sel);
                    const uint32_t b = __builtin_amdgcn_perm(pv[t][3][j/2], pv[t][2][j/2], sel);
                    const int row = dc*8 + j, chunk = (kg/2) ^ vswz(row);
                    dst[row*(V_LD16*2) + chunk*2 + (kg & 1)] = make_uint2(a, b);
                }
            }
        }
    };

    // Pipeline per tile: [barrier] issue the next tile's global loads; compute this tile from LDS buffer `buf`; store the
    // next tile into the other buffer (its loads had the whole compute to land). With one buffer the store needs a second
    // barrier (everyone done reading) and happens at the top instead.
    int kt = next_tile(kt_begin, kt_begin < kt_end ? flagword(kt_begin) : ALL2);
    if (kt < kt_end) { prefetch(kt); if (NBUF == 2) store_lds(0); }
    int buf = 0;
    __half mk_cur = __float2half(0.0f); int bin_cur = -1; float brow = bias_val(mk_cur, bin_cur, tbl, n_bins);
    while (kt < kt_end) {
        const int flag = (int) ((fw >> (8*(w/G))) & 0xff);
        const int kbase = kt*KT;
        if (NBUF == 1) store_lds(0);
        const __half mk = mk_nx; const int bin = bin_nx;
        if (!FA_ABL_NOBARRIER) __syncthreads();
        const uint4 * Ks = lds + buf*(LDS_K + LDS_V), * Vt = Ks + LDS_K;
        const int kt_nx = next_tile(kt + 1, fw_nx);
        if (kt_nx < kt_end) prefetch(kt_nx);          // in flight while this tile is computed
        // row-constant bias of a uniform tile: only re-gather the table when the bin or the mask changes (rare)
        if (flag == 1 && (bin != bin_cur || __half_as_ushort(mk) != __half_as_ushort(mk_cur))) {
            bin_cur = bin; mk_cur = mk; brow = bias_val(mk, bin, tbl, n_bins);
        }

        if (flag != 2) {
            #pragma unroll 1
            for (int step = 0; step < STEPS; ++step) {
                // S^T = K.Q^T: lane holds S[key = kbase + sub*16 + 2e + hh][q]; K fragments pipelined LA ahead
                floatx8 s[SUBS];
                #pragma unroll
                for (int ss = 0; ss < SUBS; ++ss) {
                    const int sub = step*SUBS + ss;
                    const uint4 * kp = Ks + (sub*16 + l16)*K_LD16;
                    const int ksw = kswz(l16);                       // row sub*16 + l16
                    halfx16 kf[LA];
                    #pragma unroll
                    for (int i = 0; i < LA; ++i) kf[i] = FA_ABL_NOLDS ? Qb[i] : ld_frag_s(kp, 2*i, ksw);
                    floatx8 c = (floatx8) {0, 0, 0, 0, 0, 0, 0, 0};
                    #pragma unroll
                    for (int dt = 0; dt < D/16; ++dt) {
                        const halfx16 a = kf[dt % LA];
                        if (dt + LA < D/16) kf[dt % LA] = FA_ABL_NOLDS ? Qb[dt + LA] : ld_frag_s(kp, 2*(dt + LA), ksw);
                        c = wmma(a, Qb[dt], c);
                    }
                    sched_pipeline();
                    s[ss] = c;
                }
                if (flag == 1) {
                    #pragma unroll
                    for (int ss = 0; ss < SUBS; ++ss)
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) s[ss][e] += brow;
                } else {
                    #pragma unroll
                    for (int ss = 0; ss < SUBS; ++ss)
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            const int k = kbase + (step*SUBS + ss)*16 + 2*e + hh;
                            s[ss][e] += bias_val(mrow[k], irow[k], tbl, n_bins);
                        }
                }
                // online softmax over this step's keys (the row's other keys live in lane^16)
                float mt = s[0][0];
                #pragma unroll
                for (int ss = 0; ss < SUBS; ++ss)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) mt = fmaxf(mt, s[ss][e]);
                mt = fmaxf(mt, xchg16(mt));
                const float m_old = m, mn = fmaxf(m, mt), alpha = __expf(m - mn);
                m = mn;
                float ps = 0.0f;
                #pragma unroll
                for (int ss = 0; ss < SUBS; ++ss)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) { const float p = __expf(s[ss][e] - mn); s[ss][e] = p; ps += p; }
                l = l*alpha + ps;
                if (__builtin_amdgcn_ballot_w32(mn != m_old)) {           // wave-uniform: usually false far from the diagonal
                    #pragma unroll
                    for (int dt = 0; dt < D/16; ++dt) O[dt] *= alpha;
                }

                // P as WMMA B operands, then O^T += V^T.P^T (lane holds O[d = dt*16 + 2e + hh][q]); V fragments pipelined
                #pragma unroll
                for (int ss = 0; ss < SUBS; ++ss) {
                    const int sub = step*SUBS + ss;
                    uint32_t x[4], y[4];
                    #pragma unroll
                    for (int e2 = 0; e2 < 4; ++e2) x[e2] = pack_half2(s[ss][2*e2], s[ss][2*e2 + 1]);
                    #pragma unroll
                    for (int e2 = 0; e2 < 4; ++e2) y[e2] = xchg16(x[e2]);
                    union { uint32_t u[8]; halfx16 h; } r;
                    #pragma unroll
                    for (int e2 = 0; e2 < 4; ++e2) {
                        const uint32_t a = hh ? y[e2] : x[e2], b = hh ? x[e2] : y[e2];   // a = even keys, b = odd keys
                        r.u[2*e2]     = (a & 0xffffu) | (b << 16);
                        r.u[2*e2 + 1] = (a >> 16) | (b & 0xffff0000u);
                    }
                    const halfx16 Pb = r.h;
                    const uint4 * vp = Vt + l16*V_LD16;            // rows dt*16 + l16
                    halfx16 vf[LA];
                    #pragma unroll
                    for (int i = 0; i < LA; ++i) vf[i] = FA_ABL_NOLDS ? Qb[i] : ld_frag_s(vp + i*16*V_LD16, 2*sub, vswz(i*16 + l16));
                    #pragma unroll
                    for (int dt = 0; dt < D/16; ++dt) {
                        const halfx16 a = vf[dt % LA];
                        if (dt + LA < D/16) vf[dt % LA] = FA_ABL_NOLDS ? Qb[dt + LA] : ld_frag_s(vp + (dt + LA)*16*V_LD16, 2*sub, vswz((dt + LA)*16 + l16));
                        O[dt] = wmma(a, Pb, O[dt]);
                    }
                    sched_pipeline();
                }
            }
        }
        if (NBUF == 2) { if (kt_nx < kt_end) store_lds(buf ^ 1); buf ^= 1; }
        else if (!FA_ABL_NOBARRIER) __syncthreads();
        kt = kt_nx;
    }

    l += xchg16(l);
    float * op = part_O + (((int64_t) split*HQ + hq)*n_q + q)*D;
    #pragma unroll
    for (int dt = 0; dt < D/16; ++dt)
        #pragma unroll
        for (int e = 0; e < 8; ++e) op[dt*16 + 2*e + hh] = O[dt][e];
    if (hh == 0) part_ml[((int64_t) split*HQ + hq)*n_q + q] = make_float2(m, l);
}

// combine the KV splits, add the sink, normalize; grid (n_q, HQ), block D
__global__ void fa_combine(const float * __restrict__ part_O, const float2 * __restrict__ part_ml, const float * __restrict__ sinks,
        int splits, int n_q, float * __restrict__ out) {
    const int q = blockIdx.x, hq = blockIdx.y, d = threadIdx.x;
    const float sink = sinks[hq];
    float M = sink;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, part_ml[((int64_t) s*HQ + hq)*n_q + q].x);
    float L = __expf(sink - M), acc = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float2 ml = part_ml[((int64_t) s*HQ + hq)*n_q + q];
        const float wgt = __expf(ml.x - M);
        L += ml.y*wgt; acc += part_O[(((int64_t) s*HQ + hq)*n_q + q)*D + d]*wgt;
    }
    out[((int64_t) q*HQ + hq)*D + d] = acc/L;
}

// ---- host -----------------------------------------------------------------------------------------------------------
struct Problem {
    int n_kv, n_q, n_bins, extent;
    float scale;
    std::vector<float> Q, table, sinks;
    std::vector<__half> K, V, mask;
    std::vector<int32_t> idx;
    int64_t kv_pos_stride;
};

static Problem make_problem(int n_kv, int n_q, const std::string & maskmode, bool invalid_bins, bool nonzero_sinks) {
    Problem P; P.n_kv = n_kv; P.n_q = n_q; P.extent = 1024; P.n_bins = P.extent + 1; P.scale = 1.0f/sqrtf((float) D);
    P.kv_pos_stride = (int64_t) HKV*D;
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.0f, 1.0f); std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    P.K.resize((size_t) n_kv*P.kv_pos_stride); P.V.resize(P.K.size());
    for (size_t i = 0; i < P.K.size(); ++i) { P.K[i] = __float2half(nd(rng)); P.V[i] = __float2half(nd(rng)); }
    P.Q.resize((size_t) n_q*HQ*D); for (auto & x : P.Q) x = nd(rng);
    P.table.resize((size_t) n_q*HQ*P.n_bins); for (auto & x : P.table) x = 0.5f*nd(rng);
    P.sinks.assign(HQ, 0.0f); if (nonzero_sinks) for (auto & x : P.sinks) x = nd(rng);
    P.mask.resize((size_t) n_q*n_kv); P.idx.resize((size_t) n_q*n_kv);
    for (int i = 0; i < n_q; ++i) {
        const int64_t qpos = n_kv - n_q + i;
        for (int k = 0; k < n_kv; ++k) {
            const int64_t dist = qpos - k;
            bool masked = maskmode == "causal" ? k > qpos : (ud(rng) < 0.3f);
            P.mask[(size_t) i*n_kv + k] = __float2half(masked ? -INFINITY : 0.0f);
            // llama_kv_cache::set_input_relative_pos: distance in [0, extent) -> distance, else extent (a valid bin)
            int bin = dist >= 0 && dist < P.extent ? (int) dist : P.extent;
            if (invalid_bins && dist >= 0 && dist < P.extent && (i*7 + k) % 23 == 0) bin = ((i + k) & 1) ? -1 : P.n_bins + 5;
            P.idx[(size_t) i*n_kv + k] = bin;
        }
    }
    return P;
}

// double-precision reference of the op as defined above
static void reference(const Problem & P, std::vector<double> & ref) {
    ref.assign((size_t) P.n_q*HQ*D, 0.0);
    auto work = [&](int hq) {
        const int hk = hq/G;
        std::vector<double> s(P.n_kv), kf(P.n_kv*D);
        for (int k = 0; k < P.n_kv; ++k) for (int d = 0; d < D; ++d) kf[(size_t) k*D + d] = __half2float(P.K[(size_t) k*P.kv_pos_stride + hk*D + d]);
        for (int q = 0; q < P.n_q; ++q) {
            const float * qv = P.Q.data() + ((size_t) q*HQ + hq)*D;
            double M = P.sinks[hq];
            for (int k = 0; k < P.n_kv; ++k) {
                double dot = 0; for (int d = 0; d < D; ++d) dot += (double) qv[d]*kf[(size_t) k*D + d];
                const __half mk = P.mask[(size_t) q*P.n_kv + k]; const int bin = P.idx[(size_t) q*P.n_kv + k];
                const float b = (uint32_t) bin < (uint32_t) P.n_bins
                    ? __half2float(__float2half(__half2float(mk) + P.table[((size_t) q*HQ + hq)*P.n_bins + bin])) : __half2float(mk);
                s[k] = P.scale*dot + (double) b; M = std::max(M, s[k]);
            }
            double L = std::exp(P.sinks[hq] - M); std::vector<double> o(D, 0.0);
            for (int k = 0; k < P.n_kv; ++k) {
                if (s[k] == -INFINITY) continue;
                const double p = std::exp(s[k] - M); L += p;
                const __half * vr = P.V.data() + (size_t) k*P.kv_pos_stride + hk*D;
                for (int d = 0; d < D; ++d) o[d] += p*__half2float(vr[d]);
            }
            for (int d = 0; d < D; ++d) ref[((size_t) q*HQ + hq)*D + d] = o[d]/L;
        }
    };
    std::vector<std::thread> th; for (int hq = 0; hq < HQ; ++hq) th.emplace_back(work, hq); for (auto & t : th) t.join();
}

struct Dev {
    float *Q, *table, *sinks, *part_O, *out; __half *K, *V, *mask; int32_t *idx; uint8_t *flags; float2 *part_ml;
    int max_splits;
};

static Dev upload(const Problem & P, int max_splits) {
    Dev d; d.max_splits = max_splits;
    CHECK(hipMalloc(&d.Q, P.Q.size()*4)); CHECK(hipMalloc(&d.table, P.table.size()*4)); CHECK(hipMalloc(&d.sinks, HQ*4));
    CHECK(hipMalloc(&d.K, P.K.size()*2)); CHECK(hipMalloc(&d.V, P.V.size()*2)); CHECK(hipMalloc(&d.mask, P.mask.size()*2));
    CHECK(hipMalloc(&d.idx, P.idx.size()*4)); CHECK(hipMalloc(&d.flags, (size_t) (P.n_q/16)*(P.n_kv/KT)));
    CHECK(hipMalloc(&d.part_O, (size_t) max_splits*HQ*P.n_q*D*4)); CHECK(hipMalloc(&d.part_ml, (size_t) max_splits*HQ*P.n_q*8));
    CHECK(hipMalloc(&d.out, (size_t) P.n_q*HQ*D*4));
    CHECK(hipMemcpy(d.Q, P.Q.data(), P.Q.size()*4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(d.table, P.table.data(), P.table.size()*4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.sinks, P.sinks.data(), HQ*4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.K, P.K.data(), P.K.size()*2, hipMemcpyHostToDevice)); CHECK(hipMemcpy(d.V, P.V.data(), P.V.size()*2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.mask, P.mask.data(), P.mask.size()*2, hipMemcpyHostToDevice)); CHECK(hipMemcpy(d.idx, P.idx.data(), P.idx.size()*4, hipMemcpyHostToDevice));
    return d;
}
static void release(Dev & d) {
    hipFree(d.Q); hipFree(d.table); hipFree(d.sinks); hipFree(d.K); hipFree(d.V); hipFree(d.mask); hipFree(d.idx); hipFree(d.flags);
    hipFree(d.part_O); hipFree(d.part_ml); hipFree(d.out);
}

template <int QT>
static void launch(const Problem & P, const Dev & d, int splits, bool with_flags) {
    const int n_kt = P.n_kv/KT, kps = (n_kt + splits - 1)/splits;
    if (with_flags) fa_flags<<<dim3(n_kt, P.n_q/16), 256>>>(d.mask, d.idx, P.n_kv, d.flags);
    fa_prefill<QT><<<dim3(P.n_q/QT, splits, HKV), QT/16*G*WAVE>>>(d.Q, d.K, d.V, P.kv_pos_stride, d.mask, d.idx, d.table, P.n_bins,
            d.flags, P.n_q, P.n_kv, kps, P.scale, d.part_O, d.part_ml);
    fa_combine<<<dim3(P.n_q, HQ), D>>>(d.part_O, d.part_ml, d.sinks, splits, P.n_q, d.out);
}
static void launch_qt(int qt, const Problem & P, const Dev & d, int splits, bool with_flags) {
    switch (qt) {
        case 32:  launch<32>(P, d, splits, with_flags); break;
        case 64:  launch<64>(P, d, splits, with_flags); break;
        case 128: launch<128>(P, d, splits, with_flags); break;
        default: printf("unsupported qt %d\n", qt); exit(1);
    }
}

static void check(const Problem & P, const Dev & d, int qt, int splits, const std::vector<double> & ref, const char * label) {
    launch_qt(qt, P, d, splits, true); CHECK(hipDeviceSynchronize());
    std::vector<float> out((size_t) P.n_q*HQ*D); CHECK(hipMemcpy(out.data(), d.out, out.size()*4, hipMemcpyDeviceToHost));
    double max_abs = 0, max_ref = 0, max_rel = 0, sum_sq = 0, sum_ref_sq = 0; size_t worst = 0; int nan = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        if (!std::isfinite(out[i])) { ++nan; continue; }
        const double e = std::fabs(out[i] - ref[i]); max_ref = std::max(max_ref, std::fabs(ref[i]));
        if (e > max_abs) { max_abs = e; worst = i; }
        sum_sq += e*e; sum_ref_sq += ref[i]*ref[i];
        if (std::fabs(ref[i]) > 1e-2) max_rel = std::max(max_rel, e/std::fabs(ref[i]));
    }
    // error over the bias-heavy rows: the last 16 queries (near-diagonal tiles with the gather path)
    double near_abs = 0;
    for (size_t i = (size_t) (P.n_q - 16)*HQ*D; i < out.size(); ++i) if (std::isfinite(out[i])) near_abs = std::max(near_abs, std::fabs(out[i] - ref[i]));
    printf("  [%s] qt %d splits %d: max|err| %.3g (max|ref| %.3g, rel-to-max %.2e), rms rel %.2e, max rel (|ref|>1e-2) %.2e, "
           "last-16-queries max|err| %.3g, non-finite %d, worst at q %zu h %zu d %zu\n",
           label, qt, splits, max_abs, max_ref, max_abs/max_ref, std::sqrt(sum_sq/sum_ref_sq), max_rel, near_abs, nan,
           worst/(HQ*D), (worst/D) % HQ, worst % D);
}

static void bench(const Problem & P, const Dev & d, int qt, int splits) {
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    auto time_it = [&](auto fn, int reps) { for (int w = 0; w < 2; ++w) fn(); CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(a)); for (int r = 0; r < reps; ++r) fn(); CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b));
        float ms; CHECK(hipEventElapsedTime(&ms, a, b)); return ms/reps; };
    const int reps = P.n_kv >= 65536 ? 10 : 20;
    const float ms_all   = time_it([&] { launch_qt(qt, P, d, splits, true); }, reps);
    const float ms_main  = time_it([&] { launch_qt(qt, P, d, splits, false); }, reps);
    const int n_kt = P.n_kv/KT;
    const float ms_flags = time_it([&] { fa_flags<<<dim3(n_kt, P.n_q/16), 256>>>(d.mask, d.idx, P.n_kv, d.flags); }, reps);
    const double flops = 4.0*HQ*P.n_q*(double) P.n_kv*D;
    const double kv_bytes = 2.0*P.n_kv*P.kv_pos_stride*2;
    printf("  qt %3d splits %d: total %8.3f ms (flags %.3f, attn+combine %.3f) | attn+combine %.2f TFLOPS, unique K/V %.1f GB/s; "
           "with flags %.2f TFLOPS\n", qt, splits, ms_all, ms_flags, ms_main, flops/(ms_main*1e-3)/1e12, kv_bytes/(ms_main*1e-3)/1e9,
           flops/(ms_all*1e-3)/1e12);
}

int main(int argc, char ** argv) {
    const int n_kv = argc > 1 ? atoi(argv[1]) : 4096;
    const int n_q  = argc > 2 ? atoi(argv[2]) : 512;
    const std::string mode = argc > 3 ? argv[3] : "both";
    const int qt = argc > 4 ? atoi(argv[4]) : 64;
    const int splits = argc > 5 ? atoi(argv[5]) : 1;
    const std::string maskmode = argc > 6 ? argv[6] : "causal";
    if (n_kv % KT != 0 || n_q % qt != 0 || n_q % 16 != 0) { printf("need n_kv %% %d == 0 and n_q %% qt == 0\n", KT); return 1; }
    const bool do_check = mode == "check" || mode == "both", do_bench = mode == "bench" || mode == "both";
    if (do_check && n_kv > 8192) { printf("check only up to n_kv 8192 (CPU reference)\n"); return 1; }
    printf("fa_prefill: n_kv %d, n_q %d, mask %s, K+V %.1f MB, mask %.1f MB, indices %.1f MB\n", n_kv, n_q, maskmode.c_str(),
           2.0*n_kv*HKV*D*2/1e6, 2.0*n_q*n_kv/1e6, 4.0*n_q*n_kv/1e6);
    const int max_splits = 8;
    if (do_check) {
        for (int variant = 0; variant < 2; ++variant) {
            Problem P = make_problem(n_kv, n_q, maskmode, variant == 1, variant == 1);
            std::vector<double> ref; reference(P, ref);
            Dev d = upload(P, max_splits);
            const char * label = variant == 0 ? "clean bins, zero sinks" : "invalid bins + nonzero sinks";
            for (int sp : {1, 2, 3}) check(P, d, qt, sp, ref, label);
            if (n_q % 128 == 0) check(P, d, 128, 1, ref, label);
            if (n_q % 32 == 0) check(P, d, 32, 2, ref, label);
            release(d);
        }
    }
    if (do_bench) {
        Problem P = make_problem(n_kv, n_q, maskmode, false, false);
        Dev d = upload(P, max_splits);
        if (argc > 4) bench(P, d, qt, splits);
        else for (int q : {32, 64, 128}) for (int sp : {1, 2, 4}) if (n_q % q == 0) bench(P, d, q, sp);
        release(d);
    }
    return 0;
}
