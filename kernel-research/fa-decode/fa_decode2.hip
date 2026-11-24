// Standalone prototype v2: Inkling 2 Mini decode attention over a q8_0 or q4_0 KV cache, gfx1151.
// Same design as fa_decode.hip (one block per chunk of positions for all 8 KV heads, one wave per KV head serving its
// 3 Q heads, split-K + combine, indexed bias + mask + zero sink), templated over the cache type.
//
//   ./fa_decode2 <q8_0|q4_0> <n_kv>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int D = 128, HKV = 8, G = 3, HQ = HKV*G, QK = 32, NB = D/QK, WAVE = 32;

struct q8_0_t { static constexpr int BLK = 34; static constexpr bool Q4 = false; };  // half d, int8 qs[32]
struct q4_0_t { static constexpr int BLK = 18; static constexpr bool Q4 = true;  };  // half d, uint8 qs[16]: lo = i, hi = i+16, value-8

template <typename T> constexpr int ROW_BYTES = NB*T::BLK;

__device__ __forceinline__ uint32_t word_at(const uint32_t * w, int byte) {   // 4 bytes at a 2-byte aligned offset
    return (byte % 4 == 0) ? w[byte/4] : __builtin_amdgcn_alignbyte(w[byte/4 + 1], w[byte/4], byte % 4);
}
__device__ __forceinline__ float half_at(const uint32_t * w, int byte) {
    return __half2float(__ushort_as_half((unsigned short) (w[byte/4] >> (8*(byte % 4)))));
}

template <typename T, int CHUNK>
__global__ __launch_bounds__(HKV*WAVE) void fa_partial(
        const float * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, int64_t nb_pos, int64_t nb_head,
        const __half * __restrict__ mask, const int32_t * __restrict__ idx, const float * __restrict__ table, int n_bins, int n_kv,
        float * __restrict__ part_acc, float2 * __restrict__ part_ml) {
    constexpr int ROW = ROW_BYTES<T>;
    const int lane = threadIdx.x % WAVE, h = threadIdx.x / WAVE, c = blockIdx.x, k0 = c*CHUNK;

    __shared__ int   q_i8[HKV][G][D/4];
    __shared__ float q_d [HKV][G][NB];
    __shared__ int   q_s [HKV][G][NB];        // per-block sum of the int8 Q (q4_0's -8 offset)
    __shared__ float p_s [HKV][G][WAVE];

    for (int g = 0; g < G; ++g) {
        const float4 x = ((const float4 *) (Q + (h*G + g)*D))[lane];
        float amax = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
        for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor(amax, o, WAVE));
        const float d = amax/127.0f, id = d > 0.0f ? 1.0f/d : 0.0f;
        const int b0 = (int) lrintf(x.x*id), b1 = (int) lrintf(x.y*id), b2 = (int) lrintf(x.z*id), b3 = (int) lrintf(x.w*id);
        q_i8[h][g][lane] = (b0 & 0xff) | ((b1 & 0xff) << 8) | ((b2 & 0xff) << 16) | ((b3 & 0xff) << 24);
        int s = b0 + b1 + b2 + b3;
        for (int o = 1; o < 8; o <<= 1) s += __shfl_xor(s, o, WAVE);
        if (lane % 8 == 0) { q_d[h][g][lane/8] = d; q_s[h][g][lane/8] = s; }
    }
    __syncthreads();

    float m[G], l[G], acc[G][4];
    for (int g = 0; g < G; ++g) { m[g] = -1e30f; l[g] = 0.0f; acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.0f; }

    const int kend = min(k0 + CHUNK, n_kv);
    for (int t0 = k0; t0 < kend; t0 += WAVE) {
        const int k = t0 + lane;
        float s[G] = {-1e30f, -1e30f, -1e30f};
        if (k < kend) {
            const char * row = K + (int64_t) k*nb_pos + (int64_t) h*nb_head;
            uint32_t w[ROW/4 + 1];
            #pragma unroll
            for (int i = 0; i < ROW/8; ++i) { const uint2 v = ((const uint2 *) row)[i]; w[2*i] = v.x; w[2*i + 1] = v.y; }
            w[ROW/4] = 0;
            float sum[G] = {0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int b = 0; b < NB; ++b) {
                const int off = b*T::BLK;
                const float dk = half_at(w, off);
                int isum[G] = {0, 0, 0};
                if constexpr (!T::Q4) {
                    #pragma unroll
                    for (int j = 0; j < QK/4; ++j) {
                        const uint32_t kq = word_at(w, off + 2 + 4*j);
                        #pragma unroll
                        for (int g = 0; g < G; ++g) isum[g] = __builtin_amdgcn_sudot4(true, (int) kq, true, q_i8[h][g][b*8 + j], isum[g], false);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < QK/8; ++j) {
                        const uint32_t kq = word_at(w, off + 2 + 4*j);
                        const int lo = (int) (kq & 0x0F0F0F0Fu), hi = (int) ((kq >> 4) & 0x0F0F0F0Fu);
                        #pragma unroll
                        for (int g = 0; g < G; ++g) {
                            isum[g] = __builtin_amdgcn_sudot4(false, lo, true, q_i8[h][g][b*8 + j],     isum[g], false);
                            isum[g] = __builtin_amdgcn_sudot4(false, hi, true, q_i8[h][g][b*8 + 4 + j], isum[g], false);
                        }
                    }
                    #pragma unroll
                    for (int g = 0; g < G; ++g) isum[g] -= 8*q_s[h][g][b];
                }
                #pragma unroll
                for (int g = 0; g < G; ++g) sum[g] += dk*q_d[h][g][b]*(float) isum[g];
            }
            const float mk = __half2float(mask[k]);
            const int   bi = idx[k];
            #pragma unroll
            for (int g = 0; g < G; ++g) s[g] = sum[g] + mk + ((uint32_t) bi < (uint32_t) n_bins ? table[(h*G + g)*n_bins + bi] : 0.0f);
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float mt = s[g];
            for (int o = 1; o < WAVE; o <<= 1) mt = fmaxf(mt, __shfl_xor(mt, o, WAVE));
            const float mn = fmaxf(m[g], mt), sc = __expf(m[g] - mn);
            m[g] = mn;
            const float p = s[g] > -1e29f ? __expf(s[g] - mn) : 0.0f;
            l[g] = l[g]*sc + p;
            acc[g][0] *= sc; acc[g][1] *= sc; acc[g][2] *= sc; acc[g][3] *= sc;
            p_s[h][g][lane] = p;
        }
        __builtin_amdgcn_wave_barrier();
        // V: lane = dims 4*lane..4*lane+3 of block vb
        const int vb = lane/8, vi = 4*(lane % 8);                 // first index inside the block
        const int vo = T::Q4 ? vb*T::BLK + 2 + (vi % 16) : vb*T::BLK + 2 + vi;
        const int nt = min(WAVE, kend - t0);
        for (int i = 0; i < nt; ++i) {
            const char * row = V + (int64_t) (t0 + i)*nb_pos + (int64_t) h*nb_head;
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

__global__ void fa_combine(const float * __restrict__ part_acc, const float2 * __restrict__ part_ml, int chunks, float * __restrict__ out) {
    const int hq = blockIdx.x, d = threadIdx.x;
    float M = 0.0f;
    for (int c = 0; c < chunks; ++c) M = fmaxf(M, part_ml[(int64_t) c*HQ + hq].x);
    float L = __expf(-M), acc = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float2 ml = part_ml[(int64_t) c*HQ + hq];
        const float w = __expf(ml.x - M);
        L += ml.y*w; acc += part_acc[((int64_t) c*HQ + hq)*D + d]*w;
    }
    out[hq*D + d] = acc/L;
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

template <typename T> static void run_all(int n_kv) {
    constexpr int ROW = ROW_BYTES<T>;
    const int n_bins = 64; const float scale = 1.0f/sqrtf((float) D);
    const int64_t nb_head = ROW, nb_pos = (int64_t) HKV*ROW;
    printf("%s n_kv %d, K+V %.1f MB\n", T::Q4 ? "q4_0" : "q8_0", n_kv, 2.0*n_kv*nb_pos/1e6);
    std::mt19937 rng(42); std::normal_distribution<float> nd(0.0f, 1.0f); std::uniform_int_distribution<int> ud(0, n_bins + 7);
    std::vector<char> hK((size_t) n_kv*nb_pos), hV((size_t) n_kv*nb_pos); std::vector<float> tmp(D);
    for (int k = 0; k < n_kv; ++k) for (int h = 0; h < HKV; ++h) {
        for (int i = 0; i < D; ++i) tmp[i] = nd(rng)*(i % 17 == 0 ? 3.0f : 1.0f);
        quantize_row<T>(tmp.data(), hK.data() + k*nb_pos + h*nb_head);
        for (int i = 0; i < D; ++i) tmp[i] = nd(rng);
        quantize_row<T>(tmp.data(), hV.data() + k*nb_pos + h*nb_head);
    }
    std::vector<float> hQ(HQ*D), hT(HQ*n_bins);
    for (auto & x : hQ) x = nd(rng)*0.4f*scale;
    for (auto & x : hT) x = nd(rng)*0.5f;
    std::vector<__half> hM(n_kv); std::vector<int32_t> hI(n_kv);
    for (int k = 0; k < n_kv; ++k) { hM[k] = __float2half(k >= n_kv - 3 ? -INFINITY : 0.0f); hI[k] = ud(rng); }

    std::vector<double> ref(HQ*D), s(n_kv);
    for (int hq = 0; hq < HQ; ++hq) {
        const int h = hq/G; double M = 0.0;
        for (int k = 0; k < n_kv; ++k) {
            const char * kr = hK.data() + k*nb_pos + h*nb_head;
            double dot = 0; for (int i = 0; i < D; ++i) dot += (double) hQ[hq*D + i]*deq<T>(kr, i);
            s[k] = dot + __half2float(hM[k]) + ((uint32_t) hI[k] < (uint32_t) n_bins ? hT[hq*n_bins + hI[k]] : 0.0); M = std::max(M, s[k]);
        }
        double L = std::exp(-M); std::vector<double> o(D, 0.0);
        for (int k = 0; k < n_kv; ++k) {
            const double p = std::exp(s[k] - M); L += p; const char * vr = hV.data() + k*nb_pos + h*nb_head;
            for (int i = 0; i < D; ++i) o[i] += p*deq<T>(vr, i);
        }
        for (int i = 0; i < D; ++i) ref[hq*D + i] = o[i]/L;
    }

    float *dQ, *dT, *dPA, *dO; char *dK, *dV; __half *dM; int32_t *dI; float2 *dML;
    const int max_chunks = (n_kv + 127)/128;
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
            fa_partial<T, CHUNK><<<chunks, HKV*WAVE>>>(dQ, dK, dV, nb_pos, nb_head, dM, dI, dT, n_bins, n_kv, dPA, dML);
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
        printf("  chunk %4d (%5d blocks): %7.1f us, %6.1f GB/s, rel err %.2g\n", CHUNK, chunks, ms*1e3, 2.0*n_kv*nb_pos/(ms*1e-3)/1e9, max_err/max_ref);
    };
    bench(std::integral_constant<int, 256>{});
    bench(std::integral_constant<int, 512>{});
    bench(std::integral_constant<int, 1024>{});
    bench(std::integral_constant<int, 2048>{});
    hipFree(dQ); hipFree(dT); hipFree(dK); hipFree(dV); hipFree(dM); hipFree(dI); hipFree(dPA); hipFree(dML); hipFree(dO);
}

int main(int argc, char ** argv) {
    const std::string type = argc > 1 ? argv[1] : "q8_0";
    const int n_kv = argc > 2 ? atoi(argv[2]) : 131072;
    if (type == "q4_0") run_all<q4_0_t>(n_kv); else run_all<q8_0_t>(n_kv);
    return 0;
}
