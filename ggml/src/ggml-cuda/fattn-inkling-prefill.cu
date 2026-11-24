#include "fattn-inkling.cuh"

#include <cstdlib>
#include <type_traits>

// Prefill attention for Inkling 2, ported from kernel-research/fa-prefill/fa_prefill.hip at kern 71c77209
// (the standalone prototype with a double-precision CPU reference; see its header for the design). WMMA (gfx11 wave32),
// flash-attention-2, one workgroup per (QT queries, KV head, KV split), one wave per (16 queries, Q head); the indexed bias is
// applied per (16-query, 64-key) tile from a flag: gather per element, per-row constant, or skipped (all rows -inf).
// K and V must be f16 in the per-position layout (8 heads x 128 back to back); a q8_0 cache is converted into a pooled
// buffer first. GGML_CUDA_FATTN_INKLING_PREFILL=0 turns the path off.

#if defined(GGML_USE_HIP)

namespace {

constexpr int FP_D = 128, FP_HKV = 8, FP_G = 3, FP_HQ = FP_HKV*FP_G, FP_WAVE = 32;
constexpr int FP_KT = 64, FP_NSUB = FP_KT/16, FP_LA = 2, FP_STEPS = 2, FP_SPLITS = 2;
constexpr int FP_K_LD16 = FP_D*2/16 + 1, FP_V_LD16 = FP_KT*2/16 + 1;     // padded LDS rows (uint4 units)
constexpr int FP_LDS_K = FP_KT*FP_K_LD16, FP_LDS_V = FP_D*FP_V_LD16;
static_assert((FP_LDS_K + FP_LDS_V)*16 <= 65536, "LDS");

typedef __attribute__((ext_vector_type(16))) _Float16 fp_halfx16;
typedef __attribute__((ext_vector_type(8)))  float    fp_floatx8;

static __device__ __forceinline__ fp_halfx16 fp_ld_frag(const uint4 * row, int chunk) {
    union { uint4 u[2]; fp_halfx16 h; } r; r.u[0] = row[chunk]; r.u[1] = row[chunk + 1]; return r.h;
}
static __device__ __forceinline__ int fp_vswz(int row) { return (row/8) % 8; }
static __device__ __forceinline__ fp_halfx16 fp_ld_frag_s(const uint4 * row, int chunk, int swz) {
    union { uint4 u[2]; fp_halfx16 h; } r; r.u[0] = row[chunk ^ swz]; r.u[1] = row[(chunk + 1) ^ swz]; return r.h;
}
static __device__ __forceinline__ fp_floatx8 fp_wmma(const fp_halfx16 & a, const fp_halfx16 & b, const fp_floatx8 & c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);   // lane l holds D[2e + l/16][l%16]
}
static __device__ __forceinline__ void fp_sched_pipeline() {
    __builtin_amdgcn_sched_group_barrier(0x100, 2*FP_LA, 0);
#pragma unroll
    for (int i = 0; i < FP_D/16 - FP_LA; ++i) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
    }
    __builtin_amdgcn_sched_group_barrier(0x008, FP_LA, 0);
}
static __device__ __forceinline__ float fp_bias(half mk, int bin, const float * tbl, int n_bins) {   // fattn_indexed_bias::add
    return (uint32_t) bin < (uint32_t) n_bins ? __half2float(__float2half(__half2float(mk) + tbl[bin])) : __half2float(mk);
}
static __device__ __forceinline__ uint32_t fp_xchg16(uint32_t v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false); }
static __device__ __forceinline__ float    fp_xchg16(float v)    { return __uint_as_float(fp_xchg16(__float_as_uint(v))); }
static __device__ __forceinline__ uint32_t fp_pack_half2(float a, float b) {
    return (uint32_t) __half_as_ushort(__float2half(a)) | ((uint32_t) __half_as_ushort(__float2half(b)) << 16);
}

// q8_0 cache (any position/head strides) -> f16 [pos][head][128]. One thread per 8 consecutive values, so a wave writes
// 512 contiguous bytes (a block per thread with 64 B each was 4x slower: kernel-research/fa-prefill/fa_prefill2.hip, conv_*).
static __global__ void fp_q8_0_to_f16(const char * __restrict__ x, int64_t nb1, int64_t nb2, int n_kv, half * __restrict__ y) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;   // index over [pos][head][16 groups of 8]
    if (i >= (int64_t) n_kv*FP_HKV*16) return;
    const int j = i % 16, h = (i/16) % FP_HKV; const int64_t pos = i/(16*FP_HKV);
    const char * blk = x + pos*nb1 + h*nb2 + (j/4)*34;
    const float d = __half2float(*(const half *) blk);
    const uint32_t lo = *(const uint32_t *) (blk + 2 + (j%4)*8);      // 2-byte aligned: unaligned dword loads are full speed on RDNA
    const uint32_t hi = *(const uint32_t *) (blk + 2 + (j%4)*8 + 4);
    union { half2 h2[4]; uint4 u; } o;
    o.h2[0] = __floats2half2_rn(d*(float) (int8_t) (lo),       d*(float) (int8_t) (lo >> 8));
    o.h2[1] = __floats2half2_rn(d*(float) (int8_t) (lo >> 16), d*(float) (int8_t) (lo >> 24));
    o.h2[2] = __floats2half2_rn(d*(float) (int8_t) (hi),       d*(float) (int8_t) (hi >> 8));
    o.h2[3] = __floats2half2_rn(d*(float) (int8_t) (hi >> 16), d*(float) (int8_t) (hi >> 24));
    ((uint4 *) y)[i] = o.u;
}

// flags[kt][qt]: 0 gather, 1 per-row constant, 2 constant and all rows -inf
static __global__ void fp_flags(const half * __restrict__ mask, int64_t mask_stride, const int32_t * __restrict__ idx, int n_kv,
        uint8_t * __restrict__ flags) {
    const int kt = blockIdx.x, qt = blockIdx.y, t = threadIdx.x;
    const int r = t/16, q = qt*16 + r, k = kt*FP_KT + (t % 16)*4;
    const int4  bi = *(const int4 *)  (idx  + (int64_t) q*n_kv + k);
    const uint2 mu = *(const uint2 *) (mask + (int64_t) q*mask_stride + k);
    const int   ref_i = idx[(int64_t) q*n_kv + kt*FP_KT];
    const float ref_m = __half2float(mask[(int64_t) q*mask_stride + kt*FP_KT]);
    bool uni = bi.x == ref_i && bi.y == ref_i && bi.z == ref_i && bi.w == ref_i;
    uni = uni && __half2float(__ushort_as_half(mu.x & 0xffff)) == ref_m && __half2float(__ushort_as_half(mu.x >> 16)) == ref_m
              && __half2float(__ushort_as_half(mu.y & 0xffff)) == ref_m && __half2float(__ushort_as_half(mu.y >> 16)) == ref_m;
    const bool all_uni = __syncthreads_and(uni);
    const bool all_inf = __syncthreads_and(ref_m == -INFINITY);
    if (t == 0) flags[(int64_t) kt*gridDim.y + qt] = !all_uni ? 0 : (all_inf ? 2 : 1);
}

template <int QT>
__launch_bounds__(QT/16*FP_G*FP_WAVE)
static __global__ void fp_attn(
        const float * __restrict__ Q, int64_t q_stride_q, int64_t q_stride_h,
        const half * __restrict__ K, const half * __restrict__ V, int64_t kv_pos_stride,
        const half * __restrict__ mask, int64_t mask_stride, const int32_t * __restrict__ idx, const float * __restrict__ table, int n_bins,
        const uint8_t * __restrict__ flags, int n_q, int n_kv, int kt_per_split, float scale,
        float * __restrict__ part_O, float2 * __restrict__ part_ml) {
    constexpr int NW = QT/16*FP_G, NT = NW*FP_WAVE;
    constexpr int NVT = FP_KT/4*16;
    constexpr bool ROLES = NT >= 768;
    constexpr int KT0 = ROLES ? NVT : 0, NKT = NT - KT0;
    constexpr int NKLD = (FP_KT*16 + NKT - 1)/NKT;
    constexpr int NVLD = (NVT + NT - 1)/NT;
    constexpr int SUBS = FP_NSUB/FP_STEPS;
    constexpr int LA = QT >= 128 ? FP_LA : 1;
    __shared__ uint4 lds[FP_LDS_K + FP_LDS_V];

    const int tid = threadIdx.x, lane = tid % FP_WAVE, w = tid / FP_WAVE, l16 = lane % 16, hh = lane / 16;
    const int hk = blockIdx.z, split = blockIdx.y, qbase = blockIdx.x*QT;
    const int g = w % FP_G, hq = hk*FP_G + g, q0 = qbase + (w/FP_G)*16, q = q0 + l16;
    const int n_kt = n_kv/FP_KT;
    const int kt_begin = split*kt_per_split, kt_end = min(kt_begin + kt_per_split, n_kt);
    const float   * tbl  = table + (int64_t) q*FP_HQ*n_bins + hq*n_bins;
    const half    * mrow = mask + (int64_t) q*mask_stride;
    const int32_t * irow = idx  + (int64_t) q*n_kv;
    const half * Kh = K + hk*FP_D, * Vh = V + hk*FP_D;

    fp_halfx16 Qb[FP_D/16];
    {
        const float4 * qp = (const float4 *) (Q + (int64_t) q*q_stride_q + (int64_t) hq*q_stride_h);
#pragma unroll
        for (int dt = 0; dt < FP_D/16; ++dt) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float4 x = qp[dt*4 + c];
                Qb[dt][4*c + 0] = (_Float16) (x.x*scale); Qb[dt][4*c + 1] = (_Float16) (x.y*scale);
                Qb[dt][4*c + 2] = (_Float16) (x.z*scale); Qb[dt][4*c + 3] = (_Float16) (x.w*scale);
            }
        }
    }
    fp_floatx8 O[FP_D/16];
#pragma unroll
    for (int dt = 0; dt < FP_D/16; ++dt) O[dt] = (fp_floatx8) {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -1e30f, l = 0.0f;

    const int n_qt = n_q/16;
    typedef typename std::conditional<QT == 32, uint16_t, typename std::conditional<QT == 64, uint32_t, uint64_t>::type>::type flagword_t;
    constexpr flagword_t ALL2 = (flagword_t) 0x0202020202020202ull;
    auto flagword = [&](int kt) { return *(const flagword_t *) (flags + (int64_t) kt*n_qt + qbase/16); };
    flagword_t fw = 0;
    auto next_tile = [&](int kt, flagword_t wd) {
        for (; kt < kt_end; ++kt) {
            if (wd != ALL2) break;
            if (kt + 1 < kt_end) wd = flagword(kt + 1);
        }
        fw = wd;
        return kt;
    };
    auto uni = [](int c, int limit) { return __builtin_amdgcn_readfirstlane(c) < limit; };
    uint4 pk[NKLD]; uint32_t pv[NVLD][4][4];
    half mk_nx = __float2half(0.0f); int bin_nx = -1;
    flagword_t fw_nx = ALL2;
    auto prefetch = [&](int kt) {
        if (kt + 1 < kt_end) fw_nx = flagword(kt + 1);
        const int kbase = kt*FP_KT;
        mk_nx = mrow[kbase]; bin_nx = irow[kbase];
#pragma unroll
        for (int i = 0; i < NKLD; ++i) {
            const int c = tid - KT0 + i*NKT, key = c/16, dc = c % 16;
            if (uni(c, FP_KT*16) && (!ROLES || __builtin_amdgcn_readfirstlane(tid) >= KT0))
                pk[i] = *(const uint4 *) (Kh + (int64_t) (kbase + key)*kv_pos_stride + dc*8);
        }
#pragma unroll
        for (int t = 0; t < NVLD; ++t) {
            const int c = tid + t*NT, dc = c % 16, kg = c/16;
            if (uni(c, FP_KT/4*16)) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 x = *(const uint4 *) (Vh + (int64_t) (kbase + kg*4 + i)*kv_pos_stride + dc*8);
                    pv[t][i][0] = x.x; pv[t][i][1] = x.y; pv[t][i][2] = x.z; pv[t][i][3] = x.w;
                }
            }
        }
    };
    auto store_lds = [&]() {
        uint4 * Ks = lds;
#pragma unroll
        for (int i = 0; i < NKLD; ++i) {
            const int c = tid - KT0 + i*NKT, key = c/16, dc = c % 16;
            if (uni(c, FP_KT*16) && (!ROLES || __builtin_amdgcn_readfirstlane(tid) >= KT0)) Ks[key*FP_K_LD16 + dc] = pk[i];
        }
        uint2 * dstv = (uint2 *) (Ks + FP_LDS_K);
#pragma unroll
        for (int t = 0; t < NVLD; ++t) {
            const int c = tid + t*NT, dc = c % 16, kg = c/16;
            if (uni(c, FP_KT/4*16)) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t sel = (j & 1) ? 0x07060302u : 0x05040100u;
                    const uint32_t a = __builtin_amdgcn_perm(pv[t][1][j/2], pv[t][0][j/2], sel);
                    const uint32_t b = __builtin_amdgcn_perm(pv[t][3][j/2], pv[t][2][j/2], sel);
                    const int row = dc*8 + j, chunk = (kg/2) ^ fp_vswz(row);
                    dstv[row*(FP_V_LD16*2) + chunk*2 + (kg & 1)] = make_uint2(a, b);
                }
            }
        }
    };

    int kt = next_tile(kt_begin, kt_begin < kt_end ? flagword(kt_begin) : ALL2);
    if (kt < kt_end) prefetch(kt);
    half mk_cur = __float2half(0.0f); int bin_cur = -1; float brow = fp_bias(mk_cur, bin_cur, tbl, n_bins);
    while (kt < kt_end) {
        const int flag = (int) ((fw >> (8*(w/FP_G))) & 0xff);
        const int kbase = kt*FP_KT;
        store_lds();
        const half mk = mk_nx; const int bin = bin_nx;
        __syncthreads();
        const uint4 * Ks = lds, * Vt = lds + FP_LDS_K;
        const int kt_nx = next_tile(kt + 1, fw_nx);
        if (kt_nx < kt_end) prefetch(kt_nx);
        if (flag == 1 && (bin != bin_cur || __half_as_ushort(mk) != __half_as_ushort(mk_cur))) {
            bin_cur = bin; mk_cur = mk; brow = fp_bias(mk, bin, tbl, n_bins);
        }
        if (flag != 2) {
#pragma unroll 1
            for (int step = 0; step < FP_STEPS; ++step) {
                fp_floatx8 s[SUBS];
#pragma unroll
                for (int ss = 0; ss < SUBS; ++ss) {
                    const int sub = step*SUBS + ss;
                    const uint4 * kp = Ks + (sub*16 + l16)*FP_K_LD16;
                    fp_halfx16 kf[LA];
#pragma unroll
                    for (int i = 0; i < LA; ++i) kf[i] = fp_ld_frag(kp, 2*i);
                    fp_floatx8 c = (fp_floatx8) {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                    for (int dt = 0; dt < FP_D/16; ++dt) {
                        const fp_halfx16 a = kf[dt % LA];
                        if (dt + LA < FP_D/16) kf[dt % LA] = fp_ld_frag(kp, 2*(dt + LA));
                        c = fp_wmma(a, Qb[dt], c);
                    }
                    fp_sched_pipeline();
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
                            s[ss][e] += fp_bias(mrow[k], irow[k], tbl, n_bins);
                        }
                }
                float mt = s[0][0];
#pragma unroll
                for (int ss = 0; ss < SUBS; ++ss)
#pragma unroll
                    for (int e = 0; e < 8; ++e) mt = fmaxf(mt, s[ss][e]);
                mt = fmaxf(mt, fp_xchg16(mt));
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
                    for (int dt = 0; dt < FP_D/16; ++dt) O[dt] *= alpha;
                }
#pragma unroll
                for (int ss = 0; ss < SUBS; ++ss) {
                    const int sub = step*SUBS + ss;
                    uint32_t x[4], y[4];
#pragma unroll
                    for (int e2 = 0; e2 < 4; ++e2) x[e2] = fp_pack_half2(s[ss][2*e2], s[ss][2*e2 + 1]);
#pragma unroll
                    for (int e2 = 0; e2 < 4; ++e2) y[e2] = fp_xchg16(x[e2]);
                    union { uint32_t u[8]; fp_halfx16 h; } r;
#pragma unroll
                    for (int e2 = 0; e2 < 4; ++e2) {
                        const uint32_t a = hh ? y[e2] : x[e2], b = hh ? x[e2] : y[e2];
                        r.u[2*e2]     = (a & 0xffffu) | (b << 16);
                        r.u[2*e2 + 1] = (a >> 16) | (b & 0xffff0000u);
                    }
                    const fp_halfx16 Pb = r.h;
                    const uint4 * vp = Vt + l16*FP_V_LD16;
                    fp_halfx16 vf[LA];
#pragma unroll
                    for (int i = 0; i < LA; ++i) vf[i] = fp_ld_frag_s(vp + i*16*FP_V_LD16, 2*sub, fp_vswz(i*16 + l16));
#pragma unroll
                    for (int dt = 0; dt < FP_D/16; ++dt) {
                        const fp_halfx16 a = vf[dt % LA];
                        if (dt + LA < FP_D/16) vf[dt % LA] = fp_ld_frag_s(vp + (dt + LA)*16*FP_V_LD16, 2*sub, fp_vswz((dt + LA)*16 + l16));
                        O[dt] = fp_wmma(a, Pb, O[dt]);
                    }
                    fp_sched_pipeline();
                }
            }
        }
        __syncthreads();
        kt = kt_nx;
    }

    l += fp_xchg16(l);
    float * op = part_O + (((int64_t) split*FP_HQ + hq)*n_q + q)*FP_D;
#pragma unroll
    for (int dt = 0; dt < FP_D/16; ++dt)
#pragma unroll
        for (int e = 0; e < 8; ++e) op[dt*16 + 2*e + hh] = O[dt][e];
    if (hh == 0) part_ml[((int64_t) split*FP_HQ + hq)*n_q + q] = make_float2(m, l);
}

// combine the KV splits, add the sink, normalize; grid (n_q, HQ), block D; dst [n_q][HQ][D]
static __global__ void fp_combine(const float * __restrict__ part_O, const float2 * __restrict__ part_ml, const float * __restrict__ sinks,
        int splits, int n_q, float * __restrict__ out) {
    const int q = blockIdx.x, hq = blockIdx.y, d = threadIdx.x;
    float M = sinks ? sinks[hq] : -1e30f;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, part_ml[((int64_t) s*FP_HQ + hq)*n_q + q].x);
    float L = sinks ? __expf(sinks[hq] - M) : 0.0f, acc = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float2 ml = part_ml[((int64_t) s*FP_HQ + hq)*n_q + q];
        const float wgt = __expf(ml.x - M);
        L += ml.y*wgt; acc += part_O[(((int64_t) s*FP_HQ + hq)*n_q + q)*FP_D + d]*wgt;
    }
    out[((int64_t) q*FP_HQ + hq)*FP_D + d] = acc/L;
}

template <int QT>
void fp_launch(cudaStream_t st, const float * Q, int64_t qsq, int64_t qsh, const half * K, const half * V, int64_t kvps,
        const half * mask, int64_t ms, const int32_t * idx, const float * table, int n_bins, const uint8_t * flags,
        int n_q, int n_kv, int kps, float scale, float * pO, float2 * pml) {
    fp_attn<QT><<<dim3(n_q/QT, FP_SPLITS, FP_HKV), QT/16*FP_G*FP_WAVE, 0, st>>>(Q, qsq, qsh, K, V, kvps, mask, ms, idx, table, n_bins,
        flags, n_q, n_kv, kps, scale, pO, pml);
}

int fp_env(const char * name, int dflt) {
    const char * e = getenv(name);
    return e ? atoi(e) : dflt;
}

} // namespace

#endif // defined(GGML_USE_HIP)

bool ggml_cuda_flash_attn_ext_inkling_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    static const bool enabled = fp_env("GGML_CUDA_FATTN_INKLING_PREFILL", 1) != 0;
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    const ggml_tensor * table = dst->src[6], * idx = dst->src[7];

    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));

    const int64_t n_kv = K->ne[1], n_q = Q->ne[1];
    const int qt = n_q % 128 == 0 ? 128 : n_q % 64 == 0 ? 64 : n_q % 32 == 0 ? 32 : 0;
    const bool f16_direct = K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 &&
        K->nb[2] == FP_D*2 && K->nb[1] == FP_HKV*FP_D*2 && V->nb[2] == FP_D*2 && V->nb[1] == K->nb[1] &&
        (uintptr_t) K->data % 16 == 0 && (uintptr_t) V->data % 16 == 0;
    const bool shape_ok =
        qt != 0 && n_q > 1 && n_kv % FP_KT == 0 && n_kv >= FP_KT &&
        Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && Q->ne[0] == FP_D && Q->ne[2] == FP_HQ && Q->ne[3] == 1 &&
        Q->nb[0] == 4 && Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && (uintptr_t) Q->data % 16 == 0 &&
        K->ne[0] == FP_D && K->ne[2] == FP_HKV && K->ne[3] == 1 && V->ne[0] == FP_D && V->ne[1] == n_kv && V->ne[2] == FP_HKV &&
        (f16_direct || (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0)) &&
        mask && mask->type == GGML_TYPE_F16 && mask->ne[0] == n_kv && mask->ne[1] >= n_q && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        mask->nb[1] % 8 == 0 && (uintptr_t) mask->data % 8 == 0 &&
        table && idx && !dst->src[5] && table->type == GGML_TYPE_F32 && table->ne[1] == FP_HQ && table->ne[2] == n_q &&
        ggml_is_contiguous(table) && idx->type == GGML_TYPE_I32 && idx->ne[0] == n_kv && idx->ne[1] == n_q && ggml_is_contiguous(idx) &&
        (uintptr_t) idx->data % 16 == 0 &&
        (!sinks || (sinks->type == GGML_TYPE_F32 && sinks->ne[0] == FP_HQ)) &&
        max_bias == 0.0f && softcap == 0.0f && ggml_get_op_params_i32(dst, 4) == 0 && ggml_is_contiguous(dst);
    if (!shape_ok) {
        return false;
    }

    cudaStream_t st = ctx.stream();
    const half * Kf = (const half *) K->data, * Vf = (const half *) V->data;
    ggml_cuda_pool_alloc<half> Kc(ctx.pool()), Vc(ctx.pool());
    if (!f16_direct) {
        const int64_t n = n_kv*FP_HKV*FP_D, ngrp = n_kv*FP_HKV*16;
        Kc.alloc(n); Vc.alloc(n);
        fp_q8_0_to_f16<<<(ngrp + 255)/256, 256, 0, st>>>((const char *) K->data, K->nb[1], K->nb[2], (int) n_kv, Kc.ptr);
        fp_q8_0_to_f16<<<(ngrp + 255)/256, 256, 0, st>>>((const char *) V->data, V->nb[1], V->nb[2], (int) n_kv, Vc.ptr);
        Kf = Kc.ptr; Vf = Vc.ptr;
    }
    const int n_kt = (int) (n_kv/FP_KT), kps = (n_kt + FP_SPLITS - 1)/FP_SPLITS;
    ggml_cuda_pool_alloc<uint8_t> flags(ctx.pool(), (size_t) n_kt*(n_q/16));
    ggml_cuda_pool_alloc<float>   pO(ctx.pool(), (size_t) FP_SPLITS*FP_HQ*n_q*FP_D);
    ggml_cuda_pool_alloc<float2>  pml(ctx.pool(), (size_t) FP_SPLITS*FP_HQ*n_q);
    const int64_t ms = mask->nb[1]/2;
    fp_flags<<<dim3(n_kt, n_q/16), 256, 0, st>>>((const half *) mask->data, ms, (const int32_t *) idx->data, (int) n_kv, flags.ptr);
    const int64_t qsq = Q->nb[1]/4, qsh = Q->nb[2]/4, kvps = FP_HKV*FP_D;
    const auto args = [&](auto launcher) {
        launcher(st, (const float *) Q->data, qsq, qsh, Kf, Vf, kvps, (const half *) mask->data, ms, (const int32_t *) idx->data,
                 (const float *) table->data, (int) table->ne[0], flags.ptr, (int) n_q, (int) n_kv, kps, scale, pO.ptr, pml.ptr);
    };
    switch (qt) {
        case 128: args(fp_launch<128>); break;
        case 64:  args(fp_launch<64>);  break;
        default:  args(fp_launch<32>);  break;
    }
    fp_combine<<<dim3((unsigned) n_q, FP_HQ), FP_D, 0, st>>>(pO.ptr, pml.ptr, sinks ? (const float *) sinks->data : nullptr, FP_SPLITS, (int) n_q,
        (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#endif
}
