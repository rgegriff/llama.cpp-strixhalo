#include "fattn-inkling.cuh"

#include <cstdlib>

// See fattn-inkling.cuh and kernel-research/fa-decode/fa_decode2.hip (the standalone prototype with a CPU reference).
// GGML_CUDA_FATTN_INKLING=0 turns the path off; GGML_CUDA_FATTN_INKLING_MIN_KV sets the shortest context it takes
// (default 65536: below that the vec kernel is as fast).

namespace {

constexpr int FI_D = 128, FI_HKV = 8, FI_G = 3, FI_HQ = FI_HKV*FI_G, FI_QK = 32, FI_NB = FI_D/FI_QK, FI_WAVE = 32, FI_CHUNK = 512;

template <int BLK> struct fi_type { static constexpr int blk = BLK; static constexpr bool q4 = BLK == 18; };
using fi_q8_0 = fi_type<34>;   // half d, int8 qs[32]
using fi_q4_0 = fi_type<18>;   // half d, uint8 qs[16]: low nibble = value i, high = value i+16, minus 8

static __device__ __forceinline__ uint32_t fi_word_at(const uint32_t * w, int byte) { // 4 bytes at a 2-byte aligned offset
    return (byte % 4 == 0) ? w[byte/4] : __builtin_amdgcn_alignbyte(w[byte/4 + 1], w[byte/4], byte % 4);
}
static __device__ __forceinline__ float fi_half_at(const uint32_t * w, int byte) {
    return __half2float(__ushort_as_half((unsigned short) (w[byte/4] >> (8*(byte % 4)))));
}

template <typename T>
__launch_bounds__(FI_HKV*FI_WAVE)
static __global__ void fi_partial(
        const char * __restrict__ Q, int64_t nbq2, float scale,
        const char * __restrict__ K, int64_t nbk1, int64_t nbk2,
        const char * __restrict__ V, int64_t nbv1, int64_t nbv2,
        const half * __restrict__ mask, const int32_t * __restrict__ idx, const float * __restrict__ table, int n_bins, int n_kv,
        float * __restrict__ part_acc, float2 * __restrict__ part_ml) {
    constexpr int ROW = FI_NB*T::blk;
    const int lane = threadIdx.x % FI_WAVE, h = threadIdx.x / FI_WAVE, c = blockIdx.x, k0 = c*FI_CHUNK;

    __shared__ int   q_i8[FI_HKV][FI_G][FI_D/4];
    __shared__ float q_d [FI_HKV][FI_G][FI_NB];
    __shared__ int   q_s [FI_HKV][FI_G][FI_NB];
    __shared__ float p_s [FI_HKV][FI_G][FI_WAVE];

    // Q of this wave's 3 heads, scaled, to int8 per block of 32 (as the vec kernel does)
    for (int g = 0; g < FI_G; ++g) {
        const float4 x = ((const float4 *) (Q + (h*FI_G + g)*nbq2))[lane];
        const float x0 = x.x*scale, x1 = x.y*scale, x2 = x.z*scale, x3 = x.w*scale;
        float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
        for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor(amax, o, FI_WAVE));
        const float d = amax/127.0f, id = d > 0.0f ? 1.0f/d : 0.0f;
        const int b0 = (int) lrintf(x0*id), b1 = (int) lrintf(x1*id), b2 = (int) lrintf(x2*id), b3 = (int) lrintf(x3*id);
        q_i8[h][g][lane] = (b0 & 0xff) | ((b1 & 0xff) << 8) | ((b2 & 0xff) << 16) | ((b3 & 0xff) << 24);
        int s = b0 + b1 + b2 + b3;
        for (int o = 1; o < 8; o <<= 1) s += __shfl_xor(s, o, FI_WAVE);
        if (lane % 8 == 0) { q_d[h][g][lane/8] = d; q_s[h][g][lane/8] = s; }
    }
    __syncthreads();

    float m[FI_G], l[FI_G], acc[FI_G][4];
    for (int g = 0; g < FI_G; ++g) { m[g] = -1e30f; l[g] = 0.0f; acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.0f; }

    const int kend = min(k0 + FI_CHUNK, n_kv);
    for (int t0 = k0; t0 < kend; t0 += FI_WAVE) {
        const int k = t0 + lane;
        float s[FI_G] = {-1e30f, -1e30f, -1e30f};
        if (k < kend) {
            const char * row = K + (int64_t) k*nbk1 + (int64_t) h*nbk2;
            uint32_t w[ROW/4 + 1];
#pragma unroll
            for (int i = 0; i < ROW/8; ++i) { const uint2 v = ((const uint2 *) row)[i]; w[2*i] = v.x; w[2*i + 1] = v.y; }
            w[ROW/4] = 0;
            float sum[FI_G] = {0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int b = 0; b < FI_NB; ++b) {
                const int off = b*T::blk;
                const float dk = fi_half_at(w, off);
                int isum[FI_G] = {0, 0, 0};
                if constexpr (!T::q4) {
#pragma unroll
                    for (int j = 0; j < FI_QK/4; ++j) {
                        const uint32_t kq = fi_word_at(w, off + 2 + 4*j);
#pragma unroll
                        for (int g = 0; g < FI_G; ++g) isum[g] = ggml_cuda_dp4a((int) kq, q_i8[h][g][b*8 + j], isum[g]);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < FI_QK/8; ++j) {
                        const uint32_t kq = fi_word_at(w, off + 2 + 4*j);
                        const int lo = (int) (kq & 0x0F0F0F0Fu), hi = (int) ((kq >> 4) & 0x0F0F0F0Fu);
#pragma unroll
                        for (int g = 0; g < FI_G; ++g) {
                            isum[g] = ggml_cuda_dp4a(lo, q_i8[h][g][b*8 + j],     isum[g]);
                            isum[g] = ggml_cuda_dp4a(hi, q_i8[h][g][b*8 + 4 + j], isum[g]);
                        }
                    }
#pragma unroll
                    for (int g = 0; g < FI_G; ++g) isum[g] -= 8*q_s[h][g][b];
                }
#pragma unroll
                for (int g = 0; g < FI_G; ++g) sum[g] += dk*q_d[h][g][b]*(float) isum[g];
            }
            // mask + indexed bias, rounded to half as fattn_indexed_bias::add does
            const half mk = mask[k];
            const int  bi = idx[k];
#pragma unroll
            for (int g = 0; g < FI_G; ++g) {
                const half mb = uint32_t(bi) < uint32_t(n_bins) ? __float2half(__half2float(mk) + table[(h*FI_G + g)*n_bins + bi]) : mk;
                s[g] = sum[g] + __half2float(mb);
            }
        }
#pragma unroll
        for (int g = 0; g < FI_G; ++g) {
            float mt = s[g];
            for (int o = 1; o < FI_WAVE; o <<= 1) mt = fmaxf(mt, __shfl_xor(mt, o, FI_WAVE));
            const float mn = fmaxf(m[g], mt), sc = __expf(m[g] - mn);
            m[g] = mn;
            const float p = s[g] > -1e29f ? __expf(s[g] - mn) : 0.0f;
            l[g] = l[g]*sc + p;
            acc[g][0] *= sc; acc[g][1] *= sc; acc[g][2] *= sc; acc[g][3] *= sc;
            p_s[h][g][lane] = p;
        }
        __builtin_amdgcn_wave_barrier();
        const int vb = lane/8, vi = 4*(lane % 8);
        const int vo = T::q4 ? vb*T::blk + 2 + (vi % 16) : vb*T::blk + 2 + vi;
        const int nt = min(FI_WAVE, kend - t0);
        for (int i = 0; i < nt; ++i) {
            const char * row = V + (int64_t) (t0 + i)*nbv1 + (int64_t) h*nbv2;
            const float dv = __half2float(*(const half *) (row + vb*T::blk));
            uint32_t q = *(const uint32_t *) (row + vo); // 2-byte aligned: unaligned dword loads run at full speed on RDNA
            float v0, v1, v2, v3;
            if constexpr (!T::q4) {
                v0 = dv*(float) (int8_t) q; v1 = dv*(float) (int8_t) (q >> 8); v2 = dv*(float) (int8_t) (q >> 16); v3 = dv*(float) (int8_t) (q >> 24);
            } else {
                q = (vi < 16 ? q : q >> 4) & 0x0F0F0F0Fu;
                v0 = dv*(float) ((int) (q & 0xff) - 8);         v1 = dv*(float) ((int) ((q >> 8) & 0xff) - 8);
                v2 = dv*(float) ((int) ((q >> 16) & 0xff) - 8); v3 = dv*(float) ((int) (q >> 24) - 8);
            }
#pragma unroll
            for (int g = 0; g < FI_G; ++g) {
                const float p = p_s[h][g][i];
                acc[g][0] += p*v0; acc[g][1] += p*v1; acc[g][2] += p*v2; acc[g][3] += p*v3;
            }
        }
        __builtin_amdgcn_wave_barrier();
    }
#pragma unroll
    for (int g = 0; g < FI_G; ++g) {
        float ls = l[g];
        for (int o = 1; o < FI_WAVE; o <<= 1) ls += __shfl_xor(ls, o, FI_WAVE);
        const int hq = h*FI_G + g;
        ((float4 *) (part_acc + ((int64_t) c*FI_HQ + hq)*FI_D))[lane] = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
        if (lane == 0) part_ml[(int64_t) c*FI_HQ + hq] = make_float2(m[g], ls);
    }
}

// grid.x = Q heads, block = D: combine the chunks and the sink logit of the head
static __global__ void fi_combine(const float * __restrict__ part_acc, const float2 * __restrict__ part_ml, int chunks,
        const float * __restrict__ sinks, float * __restrict__ dst) {
    const int hq = blockIdx.x, d = threadIdx.x;
    const float sink = sinks ? sinks[hq] : -INFINITY;
    float M = sinks ? sink : -1e30f;
    for (int c = 0; c < chunks; ++c) M = fmaxf(M, part_ml[(int64_t) c*FI_HQ + hq].x);
    float L = sinks ? __expf(sink - M) : 0.0f, acc = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float2 ml = part_ml[(int64_t) c*FI_HQ + hq];
        const float w = __expf(ml.x - M);
        L += ml.y*w; acc += part_acc[((int64_t) c*FI_HQ + hq)*FI_D + d]*w;
    }
    dst[hq*FI_D + d] = acc/L;
}

int fi_env(const char * name, int dflt) {
    const char * e = getenv(name);
    return e ? atoi(e) : dflt;
}

} // namespace

bool ggml_cuda_flash_attn_ext_inkling_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    static const bool enabled = fi_env("GGML_CUDA_FATTN_INKLING", 1) != 0;
    static const int  min_kv  = fi_env("GGML_CUDA_FATTN_INKLING_MIN_KV", 65536);
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    const ggml_tensor * table = dst->src[6], * idx = dst->src[7];

    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));

    const int64_t n_kv = K->ne[1];
    const bool shape_ok =
        Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && Q->ne[0] == FI_D && Q->ne[1] == 1 && Q->ne[2] == FI_HQ && Q->ne[3] == 1 &&
        K->ne[0] == FI_D && K->ne[2] == FI_HKV && K->ne[3] == 1 && V->ne[0] == FI_D && V->ne[1] == n_kv && V->ne[2] == FI_HKV &&
        K->type == V->type && (K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
        mask && mask->type == GGML_TYPE_F16 && mask->ne[0] == n_kv && table && idx && !dst->src[5] &&
        table->ne[0] > 0 && table->ne[1] == FI_HQ && idx->ne[0] == n_kv &&
        (!sinks || (sinks->type == GGML_TYPE_F32 && sinks->ne[0] == FI_HQ)) &&
        max_bias == 0.0f && softcap == 0.0f && ggml_get_op_params_i32(dst, 4) == 0 &&
        n_kv >= min_kv && Q->nb[0] == 4 && Q->nb[2] % 16 == 0 && (uintptr_t) Q->data % 16 == 0 &&
        (uintptr_t) K->data % 8 == 0 && K->nb[1] % 8 == 0 && K->nb[2] % 8 == 0 &&
        (uintptr_t) V->data % 8 == 0 && V->nb[1] % 8 == 0 && V->nb[2] % 8 == 0 &&
        ggml_is_contiguous(dst);
    if (!shape_ok) {
        return false;
    }

    const int chunks = (int) ((n_kv + FI_CHUNK - 1)/FI_CHUNK);
    ggml_cuda_pool_alloc<float>  part_acc(ctx.pool(), (size_t) chunks*FI_HQ*FI_D);
    ggml_cuda_pool_alloc<float2> part_ml (ctx.pool(), (size_t) chunks*FI_HQ);
    cudaStream_t stream = ctx.stream();

    const auto args = [&](auto kernel) {
        kernel<<<chunks, FI_HKV*FI_WAVE, 0, stream>>>((const char *) Q->data, Q->nb[2], scale,
            (const char *) K->data, K->nb[1], K->nb[2], (const char *) V->data, V->nb[1], V->nb[2],
            (const half *) mask->data, (const int32_t *) idx->data, (const float *) table->data, (int) table->ne[0], (int) n_kv,
            part_acc.ptr, part_ml.ptr);
    };
    if (K->type == GGML_TYPE_Q8_0) {
        args(fi_partial<fi_q8_0>);
    } else {
        args(fi_partial<fi_q4_0>);
    }
    fi_combine<<<FI_HQ, FI_D, 0, stream>>>(part_acc.ptr, part_ml.ptr, chunks, sinks ? (const float *) sinks->data : nullptr, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#endif
}
