// Standalone prototype: Inkling 2 Mini decode attention over a q8_0 KV cache, gfx1151.
//
// Shape: one query token, 24 Q heads, 8 KV heads (3 Q heads each), head size 128, n_kv cached positions.
// Logit(h, k) = (scale*q_h) . k_k + mask[k] + (idx[k] < n_bins ? table[h][idx[k]] : 0), plus one zero sink logit per head.
// Cache layout as llama.cpp keeps it: per position, the 8 heads' rows back to back (row = 4 q8_0 blocks = 136 bytes),
// so position stride nb_pos = 8*136 and head stride nb_head = 136. Each block of the kernel takes a chunk of positions
// for ALL 8 KV heads (one wave per KV head), so the chunk is read once, front to back; each wave serves its 3 Q heads.
// Split-K: partial (m, l, acc) per chunk and Q head, combined by a second kernel.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <type_traits>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int D = 128, HKV = 8, G = 3, HQ = HKV*G, QK = 32, NB = D/QK;   // 4 q8_0 blocks per row
constexpr int ROW = NB*(2 + QK);                                          // 136 bytes
constexpr int WAVE = 32;

struct block_q8_0 { __half d; int8_t qs[QK]; };
static_assert(sizeof(block_q8_0) == 34, "q8_0 block");

// ---------------------------------------------------------------------------------------------------------------------
// kernel 1: partials. grid.x = chunks, block = HKV waves. CHUNK positions per block, processed in tiles of WAVE.
template <int CHUNK>
__global__ __launch_bounds__(HKV*WAVE) void fa_partial(
        const float * __restrict__ Q,          // [HQ][D], already scaled
        const char  * __restrict__ K, const char * __restrict__ V, int64_t nb_pos, int64_t nb_head,
        const __half * __restrict__ mask,      // [n_kv]
        const int32_t * __restrict__ idx,      // [n_kv]
        const float * __restrict__ table, int n_bins,   // [HQ][n_bins]
        int n_kv,
        float * __restrict__ part_acc,         // [chunks][HQ][D]
        float2 * __restrict__ part_ml) {       // [chunks][HQ] (m, l)
    const int lane = threadIdx.x % WAVE;
    const int h    = threadIdx.x / WAVE;       // KV head of this wave
    const int c    = blockIdx.x;
    const int k0   = c*CHUNK;

    __shared__ int   q_i8[HKV][G][D/4];        // Q quantized to int8, packed 4 per int
    __shared__ float q_d [HKV][G][NB];
    __shared__ float p_s [HKV][G][WAVE];

    // quantize this wave's 3 Q heads: lane handles dims 4*lane..4*lane+3, 8 lanes per 32-block
    for (int g = 0; g < G; ++g) {
        const float4 x = ((const float4 *) (Q + (h*G + g)*D))[lane];
        float amax = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
        for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor(amax, o, WAVE));
        const float d  = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f/d : 0.0f;
        const int b0 = (int) lrintf(x.x*id), b1 = (int) lrintf(x.y*id), b2 = (int) lrintf(x.z*id), b3 = (int) lrintf(x.w*id);
        q_i8[h][g][lane] = (b0 & 0xff) | ((b1 & 0xff) << 8) | ((b2 & 0xff) << 16) | ((b3 & 0xff) << 24);
        if (lane % 8 == 0) q_d[h][g][lane/8] = d;
    }
    __syncthreads();

    float m[G], l[G], acc[G][4];
    for (int g = 0; g < G; ++g) { m[g] = -1e30f; l[g] = 0.0f; acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.0f; }

    const int kend = min(k0 + CHUNK, n_kv);
    for (int t0 = k0; t0 < kend; t0 += WAVE) {
        // ---- K: lane = position t0 + lane, all 3 heads
        const int k = t0 + lane;
        float s[G] = {-1e30f, -1e30f, -1e30f};
        if (k < kend) {
            const char * row = K + (int64_t) k*nb_pos + (int64_t) h*nb_head;
            uint32_t w[ROW/4 + 1];
            #pragma unroll
            for (int i = 0; i < ROW/8; ++i) {
                const uint2 v = ((const uint2 *) row)[i];
                w[2*i] = v.x; w[2*i + 1] = v.y;
            }
            w[ROW/4] = 0;
            float sum[G] = {0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int b = 0; b < NB; ++b) {
                const int off = b*34;                         // byte offset of the block
                const float dk = __half2float(__ushort_as_half((unsigned short) (w[off/4] >> (8*(off % 4)))));
                int isum[G] = {0, 0, 0};
                #pragma unroll
                for (int j = 0; j < QK/4; ++j) {
                    const int qo = off + 2 + 4*j;             // byte offset of qs word j
                    const uint32_t kq = (qo % 4 == 0) ? w[qo/4] : __builtin_amdgcn_alignbyte(w[qo/4 + 1], w[qo/4], qo % 4);
                    #pragma unroll
                    for (int g = 0; g < G; ++g) isum[g] = __builtin_amdgcn_sudot4(true, (int) kq, true, q_i8[h][g][b*8 + j], isum[g], false);
                }
                #pragma unroll
                for (int g = 0; g < G; ++g) sum[g] += dk*q_d[h][g][b]*(float) isum[g];
            }
            const float mk = __half2float(mask[k]);
            const int   bi = idx[k];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float bias = (uint32_t) bi < (uint32_t) n_bins ? table[(h*G + g)*n_bins + bi] : 0.0f;
                s[g] = sum[g] + mk + bias;
            }
        }
        // ---- online softmax over the tile
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float mt = s[g];
            for (int o = 1; o < WAVE; o <<= 1) mt = fmaxf(mt, __shfl_xor(mt, o, WAVE));
            const float mn = fmaxf(m[g], mt);
            const float sc = __expf(m[g] - mn);
            m[g] = mn;
            const float p = s[g] > -1e29f ? __expf(s[g] - mn) : 0.0f;
            l[g] = l[g]*sc + p;                                // per-lane partial sum, reduced at the end
            acc[g][0] *= sc; acc[g][1] *= sc; acc[g][2] *= sc; acc[g][3] *= sc;
            p_s[h][g][lane] = p;
        }
        __builtin_amdgcn_wave_barrier();
        // ---- V: lane = dims 4*lane..4*lane+3, loop over the tile's positions
        const int vb = lane/8, vo = vb*34 + 2 + 4*(lane % 8);
        const int nt = min(WAVE, kend - t0);
        for (int i = 0; i < nt; ++i) {
            const char * row = V + (int64_t) (t0 + i)*nb_pos + (int64_t) h*nb_head;
            const float dv = __half2float(*(const __half *) (row + vb*34));
            const uint32_t q = *(const uint32_t *) (row + vo);          // 2-byte aligned: full speed on RDNA
            const float v0 = dv*(float) (int8_t) (q), v1 = dv*(float) (int8_t) (q >> 8);
            const float v2 = dv*(float) (int8_t) (q >> 16), v3 = dv*(float) (int8_t) (q >> 24);
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = p_s[h][g][i];
                acc[g][0] += p*v0; acc[g][1] += p*v1; acc[g][2] += p*v2; acc[g][3] += p*v3;
            }
        }
        __builtin_amdgcn_wave_barrier();
    }
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        float ls = l[g];
        for (int o = 1; o < WAVE; o <<= 1) ls += __shfl_xor(ls, o, WAVE);
        const int hq = h*G + g;
        ((float4 *) (part_acc + ((int64_t) c*HQ + hq)*D))[lane] = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
        if (lane == 0) part_ml[(int64_t) c*HQ + hq] = make_float2(m[g], ls);
    }
}

// kernel 2: combine. grid.x = HQ, block = D threads. Includes the zero sink logit.
__global__ void fa_combine(const float * __restrict__ part_acc, const float2 * __restrict__ part_ml, int chunks, float * __restrict__ out) {
    const int hq = blockIdx.x, d = threadIdx.x;
    float M = 0.0f;                                   // the sink logit is 0
    for (int c = 0; c < chunks; ++c) M = fmaxf(M, part_ml[(int64_t) c*HQ + hq].x);
    float L = __expf(0.0f - M), acc = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float2 ml = part_ml[(int64_t) c*HQ + hq];
        const float w = __expf(ml.x - M);
        L += ml.y*w;
        acc += part_acc[((int64_t) c*HQ + hq)*D + d]*w;
    }
    out[hq*D + d] = acc/L;
}

// ---------------------------------------------------------------------------------------------------------------------
static void quantize_row_q8_0(const float * x, block_q8_0 * y) {
    for (int b = 0; b < NB; ++b) {
        float amax = 0; for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(x[b*QK + i]));
        const float d = amax/127.0f, id = d ? 1.0f/d : 0.0f;
        y[b].d = __float2half(d);
        for (int i = 0; i < QK; ++i) y[b].qs[i] = (int8_t) std::lrint(x[b*QK + i]*id);
    }
}
static float deq(const block_q8_0 * y, int i) { return __half2float(y[i/QK].d)*y[i/QK].qs[i%QK]; }

int main(int argc, char ** argv) {
    const int n_kv   = argc > 1 ? atoi(argv[1]) : 131072;
    const int n_bins = 64;
    const float scale = 1.0f/sqrtf((float) D);
    const int64_t nb_head = ROW, nb_pos = (int64_t) HKV*ROW;
    printf("n_kv %d, KV bytes %.1f MB\n", n_kv, 2.0*n_kv*nb_pos/1e6);

    std::mt19937 rng(42); std::normal_distribution<float> nd(0.0f, 1.0f); std::uniform_int_distribution<int> ud(0, n_bins + 7);
    std::vector<char> hK((size_t) n_kv*nb_pos), hV((size_t) n_kv*nb_pos);
    std::vector<float> tmp(D);
    for (int k = 0; k < n_kv; ++k) for (int h = 0; h < HKV; ++h) {
        // K with a few large channels, like a real clipped K; V unit RMS
        for (int i = 0; i < D; ++i) tmp[i] = nd(rng)*(i % 17 == 0 ? 3.0f : 1.0f);
        quantize_row_q8_0(tmp.data(), (block_q8_0 *) (hK.data() + k*nb_pos + h*nb_head));
        for (int i = 0; i < D; ++i) tmp[i] = nd(rng);
        quantize_row_q8_0(tmp.data(), (block_q8_0 *) (hV.data() + k*nb_pos + h*nb_head));
    }
    std::vector<float> hQ(HQ*D), hT(HQ*n_bins);
    for (auto & x : hQ) x = nd(rng)*0.4f*scale;
    for (auto & x : hT) x = nd(rng)*0.5f;
    std::vector<__half> hM(n_kv); std::vector<int32_t> hI(n_kv);
    for (int k = 0; k < n_kv; ++k) { hM[k] = __float2half(k >= n_kv - 3 ? -INFINITY : 0.0f); hI[k] = ud(rng); }

    // CPU reference in double over the dequantized cache and float Q
    std::vector<double> ref(HQ*D);
    {
        std::vector<double> s(n_kv);
        for (int hq = 0; hq < HQ; ++hq) {
            const int h = hq/G; double M = 0.0;
            for (int k = 0; k < n_kv; ++k) {
                const block_q8_0 * kr = (const block_q8_0 *) (hK.data() + k*nb_pos + h*nb_head);
                double dot = 0; for (int i = 0; i < D; ++i) dot += (double) hQ[hq*D + i]*deq(kr, i);
                const double bias = (uint32_t) hI[k] < (uint32_t) n_bins ? hT[hq*n_bins + hI[k]] : 0.0;
                s[k] = dot + __half2float(hM[k]) + bias; M = std::max(M, s[k]);
            }
            double L = std::exp(0.0 - M); std::vector<double> o(D, 0.0);
            for (int k = 0; k < n_kv; ++k) {
                const double p = std::exp(s[k] - M); L += p;
                const block_q8_0 * vr = (const block_q8_0 *) (hV.data() + k*nb_pos + h*nb_head);
                for (int i = 0; i < D; ++i) o[i] += p*deq(vr, i);
            }
            for (int i = 0; i < D; ++i) ref[hq*D + i] = o[i]/L;
        }
    }

    float *dQ, *dT, *dPA, *dO; char *dK, *dV; __half *dM; int32_t *dI; float2 *dML;
    const int max_chunks = (n_kv + 63)/64;
    CHECK(hipMalloc(&dQ, hQ.size()*4)); CHECK(hipMalloc(&dT, hT.size()*4)); CHECK(hipMalloc(&dK, hK.size())); CHECK(hipMalloc(&dV, hV.size()));
    CHECK(hipMalloc(&dM, n_kv*2)); CHECK(hipMalloc(&dI, n_kv*4)); CHECK(hipMalloc(&dPA, (size_t) max_chunks*HQ*D*4)); CHECK(hipMalloc(&dML, (size_t) max_chunks*HQ*8));
    CHECK(hipMalloc(&dO, HQ*D*4));
    CHECK(hipMemcpy(dQ, hQ.data(), hQ.size()*4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dT, hT.data(), hT.size()*4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dK, hK.data(), hK.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dV, hV.data(), hV.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dM, hM.data(), n_kv*2, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dI, hI.data(), n_kv*4, hipMemcpyHostToDevice));

    auto bench = [&](auto chunk_c) {
        constexpr int CHUNK = decltype(chunk_c)::value;
        const int chunks = (n_kv + CHUNK - 1)/CHUNK;
        auto run = [&]() {
            fa_partial<CHUNK><<<chunks, HKV*WAVE>>>(dQ, dK, dV, nb_pos, nb_head, dM, dI, dT, n_bins, n_kv, dPA, dML);
            fa_combine<<<HQ, D>>>(dPA, dML, chunks, dO);
        };
        run(); CHECK(hipDeviceSynchronize());
        std::vector<float> out(HQ*D); CHECK(hipMemcpy(out.data(), dO, out.size()*4, hipMemcpyDeviceToHost));
        double max_err = 0, max_ref = 0;
        for (int i = 0; i < HQ*D; ++i) { max_err = std::max(max_err, std::fabs(out[i] - ref[i])); max_ref = std::max(max_ref, std::fabs(ref[i])); }
        hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
        for (int w = 0; w < 3; ++w) run();
        const int reps = 50;
        hipEventRecord(a); for (int r = 0; r < reps; ++r) run(); hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); ms /= reps;
        hipEventRecord(a); for (int r = 0; r < reps; ++r) fa_combine<<<HQ, D>>>(dPA, dML, chunks, dO); hipEventRecord(b); hipEventSynchronize(b);
        float mc; hipEventElapsedTime(&mc, a, b); mc /= reps;
        printf("  chunk %4d (%5d blocks): %7.1f us (combine %6.1f), partial %6.1f GB/s, rel err %.2g\n", CHUNK, chunks, ms*1e3, mc*1e3,
               2.0*n_kv*nb_pos/((ms - mc)*1e-3)/1e9, max_err/max_ref);
    };
    bench(std::integral_constant<int, 128>{});
    bench(std::integral_constant<int, 256>{});
    bench(std::integral_constant<int, 512>{});
    bench(std::integral_constant<int, 1024>{});
    return 0;
}
