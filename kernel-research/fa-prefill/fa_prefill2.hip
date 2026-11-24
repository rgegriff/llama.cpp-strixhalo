// Prototype v2: Inkling 2 Mini PREFILL attention with WMMA on gfx1151 -- V consumed pre-transposed.
// Derived from fa_prefill.hip @ 71c77209 (kept intact as the reference); differences:
//   * V is taken as f16 [kv_head][d][key] (written once per layer visit by the q8_0 -> f16 conversion that precedes prefill
//     attention; conv_vt below is that conversion, standalone and timed). The V tile is then a plain coalesced copy into LDS
//     like K: no 8x8 register transpose, no conflicted 8 B writes, and the prefetch footprint drops (12 VGPRs at qt 128).
//   * KT (keys per LDS tile) is a template parameter: 64 as before, or 32 so that a qt 64 workgroup needs 18.5 KB of LDS
//     and ~12 prefetch VGPRs, which lets two independent 12-wave workgroups share a WGP -- the "desynchronized halves".
//   * LDS DMA (global_load_lds) was the plan for a VGPR-free prefetch; LLVM has no vmem-to-lds-load-insts feature for gfx11
//     ("needs target feature" error), so the prefetch stays in registers.
// Everything else (semantics, flags pass, online softmax, P exchange, split combine, CPU reference) is as in fa_prefill.hip.
//
//   ./fa_prefill2 <n_kv> <n_q> [check|bench|both|conv] [qt=128] [splits=2] [kt=64] [maskmode=causal|random]
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
#include <functional>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int D = 128, HKV = 8, G = 3, HQ = HKV*G, WAVE = 32;
constexpr int K_LD16 = D*2/16 + 1;        // K row in LDS: 17 x 16 B (padded against bank conflicts)
template <int KT> constexpr int V_LD16 = KT*2/16 + 1;     // V^T row in LDS: KT/8 + 1 x 16 B
template <int KT> constexpr int LDS_K = KT*K_LD16;
template <int KT> constexpr int LDS_V = D*V_LD16<KT>;
#ifndef FA_LOOKAHEAD
#define FA_LOOKAHEAD 2
#endif
#ifndef FA_SCHED
#define FA_SCHED 1
#endif

typedef __attribute__((ext_vector_type(16))) _Float16 halfx16;
typedef __attribute__((ext_vector_type(8)))  float    floatx8;

__device__ __forceinline__ halfx16 ld_frag(const uint4 * p) {        // 32 contiguous bytes = 16 halves (one WMMA A/B operand)
    union { uint4 u[2]; halfx16 h; } r; r.u[0] = p[0]; r.u[1] = p[1]; return r.h;
}
__device__ __forceinline__ floatx8 wmma(const halfx16 & a, const halfx16 & b, const floatx8 & c) {
    // D[i][j] = sum_k a_lane(i)[k] * b_lane(j)[k]; lane l holds D[2e + l/16][l%16] (probed: wmma_probe.hip)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ void sched_pipeline() {      // 2*LA ds_reads, then (1 WMMA, 2 ds_reads) x (8-LA), then LA WMMAs
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
__device__ __forceinline__ uint32_t xchg16(uint32_t v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false); }
__device__ __forceinline__ float xchg16(float v) { return __uint_as_float(xchg16(__float_as_uint(v))); }
__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    return (uint32_t) __half_as_ushort(__float2half(a)) | ((uint32_t) __half_as_ushort(__float2half(b)) << 16);
}

// flags[kt][qt]: 0 gather, 1 per-row constant (indices and mask constant along each of the 16 rows), 2 constant and all rows -inf
template <int KT>
__global__ void fa_flags(const __half * __restrict__ mask, const int32_t * __restrict__ idx, int n_kv, uint8_t * __restrict__ flags) {
    constexpr int KPT = KT/16;                       // keys per thread: 256 threads = 16 rows x 16 groups
    const int kt = blockIdx.x, qt = blockIdx.y, t = threadIdx.x;
    const int r = t/16, q = qt*16 + r, k = kt*KT + (t % 16)*KPT;
    const int ref_i = idx[(int64_t) q*n_kv + kt*KT];
    const float ref_m = __half2float(mask[(int64_t) q*n_kv + kt*KT]);
    bool uni = true;
    #pragma unroll
    for (int i = 0; i < KPT; ++i) uni = uni && idx[(int64_t) q*n_kv + k + i] == ref_i && __half2float(mask[(int64_t) q*n_kv + k + i]) == ref_m;
    const bool all_uni = __syncthreads_and(uni);
    const bool all_inf = __syncthreads_and(ref_m == -INFINITY);
    if (t == 0) flags[(int64_t) kt*gridDim.y + qt] = !all_uni ? 0 : (all_inf ? 2 : 1);
}

// Q f32 [n_q][24][128]; K f16 real cache layout (kv_pos_stride halves per position, head hk at +hk*128);
// Vt f16 [8][128][vt_stride] (transposed per KV head; vt_stride = n_kv + 64 so the 128 d-rows of a tile do not alias one L2 set)
template <int QT, int KT>
__global__ __launch_bounds__(QT/16*G*WAVE) void fa_prefill2(
        const float * __restrict__ Q, const __half * __restrict__ K, const __half * __restrict__ Vt_g, int64_t kv_pos_stride, int64_t vt_stride,
        const __half * __restrict__ mask, const int32_t * __restrict__ idx, const float * __restrict__ table, int n_bins,
        const uint8_t * __restrict__ flags, int n_q, int n_kv, int kt_per_split, float scale,
        float * __restrict__ part_O, float2 * __restrict__ part_ml) {
    constexpr int NW = QT/16*G, NT = NW*WAVE;
    constexpr int NSUB = KT/16, STEPS = KT == 64 ? 2 : 1, SUBS = NSUB/STEPS;
    constexpr int LA = FA_LOOKAHEAD;
    constexpr int NKC = KT*16, NVC = D*(KT/8);       // 16 B chunks of the K tile and of the V^T tile
    constexpr int NLD = (NKC + NVC + NT - 1)/NT;     // chunks per thread per tile (prefetch registers = 4*NLD)
    constexpr int VLD = V_LD16<KT>;
    __shared__ uint4 lds[LDS_K<KT> + LDS_V<KT>];
    const uint4 * Ks = lds; const uint4 * Vt = lds + LDS_K<KT>;

    const int tid = threadIdx.x, lane = tid % WAVE, w = tid / WAVE, l16 = lane % 16, hh = lane / 16;
    const int hk = blockIdx.z, split = blockIdx.y, qbase = blockIdx.x*QT;
    const int g = w % G, hq = hk*G + g, q0 = qbase + (w/G)*16, q = q0 + l16;
    const int n_kt = n_kv/KT;
    const int kt_begin = split*kt_per_split, kt_end = min(kt_begin + kt_per_split, n_kt);
    const float * tbl = table + (int64_t) q*HQ*n_bins + hq*n_bins;
    const __half   * mrow = mask + (int64_t) q*n_kv;
    const int32_t  * irow = idx  + (int64_t) q*n_kv;
    const __half * Kh = K + hk*D, * Vh = Vt_g + (int64_t) hk*D*vt_stride;

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

    const int n_qt = n_q/16;
    typedef typename std::conditional<QT == 32, uint16_t, typename std::conditional<QT == 64, uint32_t, uint64_t>::type>::type flagword_t;
    constexpr flagword_t ALL2 = (flagword_t) 0x0202020202020202ull;
    auto flagword = [&](int kt) { return *(const flagword_t *) (flags + (int64_t) kt*n_qt + qbase/16); };
    flagword_t fw = 0;
    auto next_tile = [&](int kt, flagword_t wd) {      // first tile >= kt not skipped by the whole workgroup; wd = flags of kt
        for (; kt < kt_end; ++kt) {
            if (wd != ALL2) break;
            if (kt + 1 < kt_end) wd = flagword(kt + 1);
        }
        fw = wd;
        return kt;
    };
    auto uni = [](int c, int limit) { return __builtin_amdgcn_readfirstlane(c) < limit; };   // wave-uniform guard

    // tile prefetch: chunk c < NKC is K chunk (key = c/16, dc = c%16), else V^T chunk (row d = (c-NKC)/(KT/8), kc = rest)
    uint4 pf[NLD];
    __half mk_nx; int bin_nx; flagword_t fw_nx = ALL2;
    auto prefetch = [&](int kt) {
        if (kt + 1 < kt_end) fw_nx = flagword(kt + 1);
        const int kbase = kt*KT;
        mk_nx = mrow[kbase]; bin_nx = irow[kbase];
        // branch-free: a K-or-V branch per chunk made pf[] live across control flow and the compiler spilled every chunk to
        // scratch with an immediate vmcnt(0) (measured: slower than no prefetch); out-of-range chunks reload a valid address
        #pragma unroll
        for (int i = 0; i < NLD; ++i) {
            const int c = min(tid + i*NT, NKC + NVC - 1);
            const int key = c/16, dc = c % 16;
            const int cv = max(c - NKC, 0), d = cv/(KT/8), kc = cv % (KT/8);
            const __half * pk = Kh + (int64_t) (kbase + key)*kv_pos_stride + dc*8;
            const __half * pv = Vh + (int64_t) d*vt_stride + kbase + kc*8;
            pf[i] = *(const uint4 *) (c < NKC ? pk : pv);
        }
    };
    auto store_lds = [&]() {
        uint4 * Kw = lds; uint4 * Vw = lds + LDS_K<KT>;
        #pragma unroll
        for (int i = 0; i < NLD; ++i) {
            const int c = tid + i*NT;
            if (uni(c, NKC)) Kw[(c/16)*K_LD16 + c % 16] = pf[i];
            else if (uni(c, NKC + NVC)) { const int cv = c - NKC; Vw[(cv/(KT/8))*VLD + cv % (KT/8)] = pf[i]; }
        }
    };

    int kt = next_tile(kt_begin, kt_begin < kt_end ? flagword(kt_begin) : ALL2);
    if (kt < kt_end) prefetch(kt);
    __half mk_cur = __float2half(0.0f); int bin_cur = -1; float brow = bias_val(mk_cur, bin_cur, tbl, n_bins);
    while (kt < kt_end) {
        const int flag = (int) ((fw >> (8*(w/G))) & 0xff);
        const int kbase = kt*KT;
        store_lds();
        const __half mk = mk_nx; const int bin = bin_nx;
        __syncthreads();
        const int kt_nx = next_tile(kt + 1, fw_nx);
        if (kt_nx < kt_end) prefetch(kt_nx);          // in flight while this tile is computed
        if (flag == 1 && (bin != bin_cur || __half_as_ushort(mk) != __half_as_ushort(mk_cur))) {
            bin_cur = bin; mk_cur = mk; brow = bias_val(mk, bin, tbl, n_bins);
        }

        if (flag != 2) {
            #pragma unroll 1
            for (int step = 0; step < STEPS; ++step) {
                floatx8 s[SUBS];
                #pragma unroll
                for (int ss = 0; ss < SUBS; ++ss) {
                    const int sub = step*SUBS + ss;
                    const uint4 * kp = Ks + (sub*16 + l16)*K_LD16;
                    halfx16 kf[LA];
                    #pragma unroll
                    for (int i = 0; i < LA; ++i) kf[i] = ld_frag(kp + 2*i);
                    floatx8 c = (floatx8) {0, 0, 0, 0, 0, 0, 0, 0};
                    #pragma unroll
                    for (int dt = 0; dt < D/16; ++dt) {
                        const halfx16 a = kf[dt % LA];
                        if (dt + LA < D/16) kf[dt % LA] = ld_frag(kp + 2*(dt + LA));
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
                if (__builtin_amdgcn_ballot_w32(mn != m_old)) {
                    #pragma unroll
                    for (int dt = 0; dt < D/16; ++dt) O[dt] *= alpha;
                }
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
                        const uint32_t a = hh ? y[e2] : x[e2], b = hh ? x[e2] : y[e2];
                        r.u[2*e2]     = (a & 0xffffu) | (b << 16);
                        r.u[2*e2 + 1] = (a >> 16) | (b & 0xffff0000u);
                    }
                    const halfx16 Pb = r.h;
                    const uint4 * vp = Vt + l16*VLD + 2*sub;        // rows dt*16 + l16, keys sub*16..+15
                    halfx16 vf[LA];
                    #pragma unroll
                    for (int i = 0; i < LA; ++i) vf[i] = ld_frag(vp + i*16*VLD);
                    #pragma unroll
                    for (int dt = 0; dt < D/16; ++dt) {
                        const halfx16 a = vf[dt % LA];
                        if (dt + LA < D/16) vf[dt % LA] = ld_frag(vp + (dt + LA)*16*VLD);
                        O[dt] = wmma(a, Pb, O[dt]);
                    }
                    sched_pipeline();
                }
            }
        }
        __syncthreads();
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

// ---- q8_0 V (real per-position layout: 8 heads x 4 blocks x 34 B = 1088 B per position) -> f16 ----------------------------
constexpr int Q8_BLK = 34, Q8_ROW = 4*Q8_BLK, Q8_POS = HKV*Q8_ROW;
__device__ __forceinline__ uint32_t word_at(const uint32_t * w, int byte) {   // 4 bytes at a 2-byte aligned offset
    return (byte % 4 == 0) ? w[byte/4] : __builtin_amdgcn_alignbyte(w[byte/4 + 1], w[byte/4], byte % 4);
}
__device__ __forceinline__ float half_at(const uint32_t * w, int byte) {
    return __half2float(__ushort_as_half((unsigned short) (w[byte/4] >> (8*(byte % 4)))));
}
// dequantize one block (32 values of one key) into 16 packed half2
__device__ __forceinline__ void deq_block(const char * row, int b, uint32_t out[16]) {
    const uint32_t * w = (const uint32_t *) row;
    const float d = half_at(w, b*Q8_BLK);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t qw = word_at(w, b*Q8_BLK + 2 + 4*j);
        out[2*j]     = pack_half2(d*(float) (int8_t) qw,         d*(float) (int8_t) (qw >> 8));
        out[2*j + 1] = pack_half2(d*(float) (int8_t) (qw >> 16), d*(float) (int8_t) (qw >> 24));
    }
}
// transposed: out[(hk*128 + d)*vt_stride + key]; grid (n_kv/64, 8), block 256
__global__ void conv_vt(const char * __restrict__ Vq, int n_kv, int64_t vt_stride, __half * __restrict__ out) {
    __shared__ uint4 tile[64*K_LD16];                 // [key][d] f16, padded rows
    const int kt = blockIdx.x, hk = blockIdx.y, t = threadIdx.x;
    {   // phase 1: thread = (key, block)
        const int key = t/4, b = t % 4;
        uint32_t v[16]; deq_block(Vq + (int64_t) (kt*64 + key)*Q8_POS + hk*Q8_ROW, b, v);
        uint4 * dst = tile + key*K_LD16 + b*4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = make_uint4(v[4*i], v[4*i + 1], v[4*i + 2], v[4*i + 3]);
    }
    __syncthreads();
    {   // phase 2: thread = (8 d, 4 keys) -> 8 x 8 B transposed writes; 16 lanes with consecutive key groups -> 128 B per row
        const int kg = t % 16, dg = t/16;
        uint32_t v[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) { const uint4 x = tile[(kg*4 + i)*K_LD16 + dg]; v[i][0] = x.x; v[i][1] = x.y; v[i][2] = x.z; v[i][3] = x.w; }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t sel = (j & 1) ? 0x07060302u : 0x05040100u;
            const uint32_t a = __builtin_amdgcn_perm(v[1][j/2], v[0][j/2], sel), b = __builtin_amdgcn_perm(v[3][j/2], v[2][j/2], sel);
            *(uint2 *) (out + ((int64_t) hk*D + dg*8 + j)*vt_stride + kt*64 + kg*4) = make_uint2(a, b);
        }
    }
}
// plain (ggml's current conversion target): out[(key*8 + hk)*128 + d]; grid (n_kv/64, 8), block 256
__global__ void conv_plain(const char * __restrict__ Vq, int n_kv, __half * __restrict__ out) {
    const int kt = blockIdx.x, hk = blockIdx.y, t = threadIdx.x, key = kt*64 + t/4, b = t % 4;
    uint32_t v[16]; deq_block(Vq + (int64_t) key*Q8_POS + hk*Q8_ROW, b, v);
    uint4 * dst = (uint4 *) (out + ((int64_t) key*HKV + hk)*D + b*32);
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = make_uint4(v[4*i], v[4*i + 1], v[4*i + 2], v[4*i + 3]);
}

// ---- host -----------------------------------------------------------------------------------------------------------
struct Problem {
    int n_kv, n_q, n_bins, extent;
    float scale;
    std::vector<float> Q, table, sinks;
    std::vector<__half> K, V, Vt, mask;
    std::vector<int32_t> idx;
    int64_t kv_pos_stride, vt_stride;
};

static Problem make_problem(int n_kv, int n_q, const std::string & maskmode, bool invalid_bins, bool nonzero_sinks) {
    Problem P; P.n_kv = n_kv; P.n_q = n_q; P.extent = 1024; P.n_bins = P.extent + 1; P.scale = 1.0f/sqrtf((float) D);
    P.kv_pos_stride = (int64_t) HKV*D; P.vt_stride = n_kv + 64;
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.0f, 1.0f); std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    P.K.resize((size_t) n_kv*P.kv_pos_stride); P.V.resize(P.K.size()); P.Vt.assign((size_t) HKV*D*P.vt_stride, __float2half(0.0f));
    for (size_t i = 0; i < P.K.size(); ++i) { P.K[i] = __float2half(nd(rng)); P.V[i] = __float2half(nd(rng)); }
    for (int k = 0; k < n_kv; ++k) for (int h = 0; h < HKV; ++h) for (int d = 0; d < D; ++d)
        P.Vt[((size_t) h*D + d)*P.vt_stride + k] = P.V[(size_t) k*P.kv_pos_stride + h*D + d];
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
            int bin = dist >= 0 && dist < P.extent ? (int) dist : P.extent;     // llama_kv_cache::set_input_relative_pos
            if (invalid_bins && dist >= 0 && dist < P.extent && (i*7 + k) % 23 == 0) bin = ((i + k) & 1) ? -1 : P.n_bins + 5;
            P.idx[(size_t) i*n_kv + k] = bin;
        }
    }
    return P;
}

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
    float *Q, *table, *sinks, *part_O, *out; __half *K, *Vt, *mask; int32_t *idx; uint8_t *flags; float2 *part_ml;
};
static Dev upload(const Problem & P, int max_splits) {
    Dev d;
    CHECK(hipMalloc(&d.Q, P.Q.size()*4)); CHECK(hipMalloc(&d.table, P.table.size()*4)); CHECK(hipMalloc(&d.sinks, HQ*4));
    CHECK(hipMalloc(&d.K, P.K.size()*2)); CHECK(hipMalloc(&d.Vt, P.Vt.size()*2)); CHECK(hipMalloc(&d.mask, P.mask.size()*2));
    CHECK(hipMalloc(&d.idx, P.idx.size()*4)); CHECK(hipMalloc(&d.flags, (size_t) (P.n_q/16)*(P.n_kv/32)));
    CHECK(hipMalloc(&d.part_O, (size_t) max_splits*HQ*P.n_q*D*4)); CHECK(hipMalloc(&d.part_ml, (size_t) max_splits*HQ*P.n_q*8));
    CHECK(hipMalloc(&d.out, (size_t) P.n_q*HQ*D*4));
    CHECK(hipMemcpy(d.Q, P.Q.data(), P.Q.size()*4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(d.table, P.table.data(), P.table.size()*4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.sinks, P.sinks.data(), HQ*4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.K, P.K.data(), P.K.size()*2, hipMemcpyHostToDevice)); CHECK(hipMemcpy(d.Vt, P.Vt.data(), P.Vt.size()*2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.mask, P.mask.data(), P.mask.size()*2, hipMemcpyHostToDevice)); CHECK(hipMemcpy(d.idx, P.idx.data(), P.idx.size()*4, hipMemcpyHostToDevice));
    return d;
}
static void release(Dev & d) {
    hipFree(d.Q); hipFree(d.table); hipFree(d.sinks); hipFree(d.K); hipFree(d.Vt); hipFree(d.mask); hipFree(d.idx); hipFree(d.flags);
    hipFree(d.part_O); hipFree(d.part_ml); hipFree(d.out);
}

template <int QT, int KT>
static void launch(const Problem & P, const Dev & d, int splits, bool with_flags) {
    const int n_kt = P.n_kv/KT, kps = (n_kt + splits - 1)/splits;
    if (with_flags) fa_flags<KT><<<dim3(n_kt, P.n_q/16), 256>>>(d.mask, d.idx, P.n_kv, d.flags);
    fa_prefill2<QT, KT><<<dim3(P.n_q/QT, splits, HKV), QT/16*G*WAVE>>>(d.Q, d.K, d.Vt, P.kv_pos_stride, P.vt_stride, d.mask, d.idx, d.table, P.n_bins,
            d.flags, P.n_q, P.n_kv, kps, P.scale, d.part_O, d.part_ml);
    fa_combine<<<dim3(P.n_q, HQ), D>>>(d.part_O, d.part_ml, d.sinks, splits, P.n_q, d.out);
}
static void launch_cfg(int qt, int kt, const Problem & P, const Dev & d, int splits, bool with_flags) {
    if (kt == 64) switch (qt) { case 64: launch<64, 64>(P, d, splits, with_flags); return; case 128: launch<128, 64>(P, d, splits, with_flags); return; }
    if (kt == 32) switch (qt) { case 64: launch<64, 32>(P, d, splits, with_flags); return; case 128: launch<128, 32>(P, d, splits, with_flags); return; }
    printf("unsupported qt %d kt %d\n", qt, kt); exit(1);
}
static void flags_only(int kt, const Problem & P, const Dev & d) {
    if (kt == 64) fa_flags<64><<<dim3(P.n_kv/64, P.n_q/16), 256>>>(d.mask, d.idx, P.n_kv, d.flags);
    else          fa_flags<32><<<dim3(P.n_kv/32, P.n_q/16), 256>>>(d.mask, d.idx, P.n_kv, d.flags);
}

static void check(const Problem & P, const Dev & d, int qt, int kt, int splits, const std::vector<double> & ref, const char * label) {
    launch_cfg(qt, kt, P, d, splits, true); CHECK(hipDeviceSynchronize());
    std::vector<float> out((size_t) P.n_q*HQ*D); CHECK(hipMemcpy(out.data(), d.out, out.size()*4, hipMemcpyDeviceToHost));
    double max_abs = 0, max_ref = 0, max_rel = 0, sum_sq = 0, sum_ref_sq = 0; size_t worst = 0; int nan = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        if (!std::isfinite(out[i])) { ++nan; continue; }
        const double e = std::fabs(out[i] - ref[i]); max_ref = std::max(max_ref, std::fabs(ref[i]));
        if (e > max_abs) { max_abs = e; worst = i; }
        sum_sq += e*e; sum_ref_sq += ref[i]*ref[i];
        if (std::fabs(ref[i]) > 1e-2) max_rel = std::max(max_rel, e/std::fabs(ref[i]));
    }
    double near_abs = 0;
    for (size_t i = (size_t) (P.n_q - 16)*HQ*D; i < out.size(); ++i) if (std::isfinite(out[i])) near_abs = std::max(near_abs, std::fabs(out[i] - ref[i]));
    printf("  [%s] qt %d kt %d splits %d: max|err| %.3g (max|ref| %.3g, rel-to-max %.2e), rms rel %.2e, max rel (|ref|>1e-2) %.2e, "
           "last-16-queries max|err| %.3g, non-finite %d, worst at q %zu h %zu d %zu\n",
           label, qt, kt, splits, max_abs, max_ref, max_abs/max_ref, std::sqrt(sum_sq/sum_ref_sq), max_rel, near_abs, nan,
           worst/(HQ*D), (worst/D) % HQ, worst % D);
}

static float time_ms(const std::function<void()> & fn, int reps) {
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    for (int w = 0; w < 2; ++w) fn(); CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(a)); for (int r = 0; r < reps; ++r) fn(); CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b));
    float ms; CHECK(hipEventElapsedTime(&ms, a, b)); hipEventDestroy(a); hipEventDestroy(b); return ms/reps;
}

static void bench(const Problem & P, const Dev & d, int qt, int kt, int splits) {
    const int reps = P.n_kv >= 65536 ? 10 : 20;
    const float ms_all   = time_ms([&] { launch_cfg(qt, kt, P, d, splits, true); }, reps);
    const float ms_main  = time_ms([&] { launch_cfg(qt, kt, P, d, splits, false); }, reps);
    const float ms_flags = time_ms([&] { flags_only(kt, P, d); }, reps);
    const double flops = 4.0*HQ*P.n_q*(double) P.n_kv*D, kv_bytes = 2.0*P.n_kv*P.kv_pos_stride*2;
    printf("  qt %3d kt %2d splits %d: total %8.3f ms (flags %.3f, attn+combine %.3f) | attn+combine %.2f TFLOPS, unique K/V %.1f GB/s; "
           "with flags %.2f TFLOPS\n", qt, kt, splits, ms_all, ms_flags, ms_main, flops/(ms_main*1e-3)/1e12, kv_bytes/(ms_main*1e-3)/1e9,
           flops/(ms_all*1e-3)/1e12);
}

// q8_0 conversion: correctness (bit-exact vs CPU dequant) and timing of transposed vs plain output
static void conv_test(int n_kv) {
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<char> hVq((size_t) n_kv*Q8_POS);
    for (int k = 0; k < n_kv; ++k) for (int h = 0; h < HKV; ++h) for (int b = 0; b < 4; ++b) {
        char * blk = hVq.data() + (size_t) k*Q8_POS + h*Q8_ROW + b*Q8_BLK;
        float x[32], amax = 0; for (int i = 0; i < 32; ++i) { x[i] = nd(rng); amax = std::max(amax, std::fabs(x[i])); }
        const float dd = amax/127.0f, id = dd ? 1.0f/dd : 0.0f;
        *(__half *) blk = __float2half(dd);
        for (int i = 0; i < 32; ++i) ((int8_t *) (blk + 2))[i] = (int8_t) std::lrint(x[i]*id);
    }
    const int64_t vts = n_kv + 64;
    char * dVq; __half * dT, * dPl;
    CHECK(hipMalloc(&dVq, hVq.size())); CHECK(hipMalloc(&dT, (size_t) vts*HKV*D*2)); CHECK(hipMalloc(&dPl, (size_t) n_kv*HKV*D*2));
    CHECK(hipMemcpy(dVq, hVq.data(), hVq.size(), hipMemcpyHostToDevice));
    conv_vt<<<dim3(n_kv/64, HKV), 256>>>(dVq, n_kv, vts, dT); conv_plain<<<dim3(n_kv/64, HKV), 256>>>(dVq, n_kv, dPl); CHECK(hipDeviceSynchronize());
    std::vector<__half> hT((size_t) vts*HKV*D), hPl((size_t) n_kv*HKV*D);
    CHECK(hipMemcpy(hT.data(), dT, hT.size()*2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(hPl.data(), dPl, hPl.size()*2, hipMemcpyDeviceToHost));
    size_t bad_t = 0, bad_p = 0;
    for (int k = 0; k < n_kv; ++k) for (int h = 0; h < HKV; ++h) for (int d = 0; d < D; ++d) {
        const char * blk = hVq.data() + (size_t) k*Q8_POS + h*Q8_ROW + (d/32)*Q8_BLK;
        const __half ref = __float2half(__half2float(*(const __half *) blk)*((const int8_t *) (blk + 2))[d % 32]);
        bad_t += __half_as_ushort(hT[((size_t) h*D + d)*vts + k]) != __half_as_ushort(ref);
        bad_p += __half_as_ushort(hPl[((size_t) k*HKV + h)*D + d]) != __half_as_ushort(ref);
    }
    const int reps = 20;
    const float ms_t = time_ms([&] { conv_vt<<<dim3(n_kv/64, HKV), 256>>>(dVq, n_kv, vts, dT); }, reps);
    const float ms_p = time_ms([&] { conv_plain<<<dim3(n_kv/64, HKV), 256>>>(dVq, n_kv, dPl); }, reps);
    const double rd = (double) n_kv*Q8_POS, wr = (double) n_kv*HKV*D*2;
    printf("conv n_kv %d (V of 8 heads: read %.1f MB q8_0, write %.1f MB f16): transposed [head][d][key] %.3f ms (%.0f GB/s), "
           "plain [pos][head][d] %.3f ms (%.0f GB/s); mismatches vs CPU dequant: transposed %zu, plain %zu\n",
           n_kv, rd/1e6, wr/1e6, ms_t, (rd + wr)/(ms_t*1e-3)/1e9, ms_p, (rd + wr)/(ms_p*1e-3)/1e9, bad_t, bad_p);
    hipFree(dVq); hipFree(dT); hipFree(dPl);
}

int main(int argc, char ** argv) {
    const int n_kv = argc > 1 ? atoi(argv[1]) : 4096;
    const int n_q  = argc > 2 ? atoi(argv[2]) : 512;
    const std::string mode = argc > 3 ? argv[3] : "both";
    const int qt = argc > 4 ? atoi(argv[4]) : 128;
    const int splits = argc > 5 ? atoi(argv[5]) : 2;
    const int kt = argc > 6 ? atoi(argv[6]) : 64;
    const std::string maskmode = argc > 7 ? argv[7] : "causal";
    if (mode == "conv") { conv_test(n_kv); return 0; }
    if (n_kv % 64 != 0 || n_q % qt != 0 || n_q % 16 != 0) { printf("need n_kv %% 64 == 0 and n_q %% qt == 0\n"); return 1; }
    const bool do_check = mode == "check" || mode == "both", do_bench = mode == "bench" || mode == "both";
    if (do_check && n_kv > 8192) { printf("check only up to n_kv 8192 (CPU reference)\n"); return 1; }
    printf("fa_prefill2: n_kv %d, n_q %d, mask %s, K+V %.1f MB\n", n_kv, n_q, maskmode.c_str(), 2.0*n_kv*HKV*D*2/1e6);
    const int max_splits = 8;
    if (do_check) {
        for (int variant = 0; variant < 2; ++variant) {
            Problem P = make_problem(n_kv, n_q, maskmode, variant == 1, variant == 1);
            std::vector<double> ref; reference(P, ref);
            Dev d = upload(P, max_splits);
            const char * label = variant == 0 ? "clean bins, zero sinks" : "invalid bins + nonzero sinks";
            for (int k : {64, 32}) for (int sp : {1, 3}) check(P, d, qt, k, sp, ref, label);
            if (n_q % 128 == 0 && qt != 128) check(P, d, 128, 64, 2, ref, label);
            if (qt != 64) check(P, d, 64, 32, 2, ref, label);
            release(d);
        }
    }
    if (do_bench) {
        Problem P = make_problem(n_kv, n_q, maskmode, false, false);
        Dev d = upload(P, max_splits);
        if (argc > 4) bench(P, d, qt, kt, splits);
        else for (int k : {64, 32}) for (int q : {64, 128}) for (int sp : {2, 4}) bench(P, d, q, k, sp);
        release(d);
    }
    return 0;
}
