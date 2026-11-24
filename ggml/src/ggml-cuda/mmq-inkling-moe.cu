// MUL_MAT_ID (MoE expert matmul) for IQ4_NL experts x Q8_1_MMQ activations on RDNA3.5 WMMA.
// Prototype and measurements: kernel-research/moe/moe_wmma_kernel.cuh, NOTES.md.
//
// One workgroup computes a BM x BN output tile (BM expert rows x up to BN compact tokens of one expert) over the whole
// K range. The weights stay in their IQ4_NL byte layout until the fragment is built: the K loop stages raw 18-byte
// blocks in LDS (nibbles 16B-aligned per block, scales separate), each wave builds its WMMA fragments with the
// 16-entry table lookup in registers, and the int32 accumulators of one 32-wide block are scaled by d_w*d_y into float
// sums. Global->LDS loads for chunk c+1 are issued into registers before the compute of chunk c (one barrier per chunk).
// Column tiles are expert-aligned (plan_tiles), so a 16-token WMMA tile is wasted at most once per expert.
#include "mmq-inkling-moe.cuh"
#include "mmq.cuh"

#include <cstdio>
#include <cstdlib>

namespace inkling_moe {

constexpr int BM       = 128;            // expert rows per workgroup
constexpr int BN       = GGML_CUDA_INKLING_MOE_J;   // tokens (compact columns) per workgroup
constexpr int BK       = 128;            // K per chunk == QK8_1_MMQ (one block_q8_1_mmq per token per chunk)
constexpr int NTHREADS = GGML_CUDA_INKLING_MOE_NTHREADS;
constexpr int NWARPS   = NTHREADS / 32;
constexpr int ROWS_PER_WARP = BM / NWARPS;          // 32
constexpr int RT       = ROWS_PER_WARP / 16;        // 16-row WMMA tiles per wave
constexpr int TT       = BN / 16;                   // 16-token WMMA tiles
constexpr int W_STRIDE = 80;                        // bytes per staged row: 4 blocks x 16 nibble bytes + 16 pad (conflict-free b128 reads)
constexpr int W_CHUNK  = BK/32 * 18;                // 72 bytes of IQ4_NL per row per chunk
constexpr int Y_STRIDE = 144;                       // sizeof(block_q8_1_mmq)
constexpr int LDS_W_QS = BM * W_STRIDE;
constexpr int LDS_W_D  = BM * 8;                    // 4 fp16 scales per row
constexpr int LDS_Y    = BN * Y_STRIDE;
constexpr int LDS_BUF  = LDS_W_QS + LDS_W_D + LDS_Y;
constexpr int LDS_BYTES = 2*LDS_BUF;                // ids_s lives in the pad bytes of the first BN rows of buffer 0

constexpr int PLAN_THREADS = 256;
constexpr int MAX_EXPERTS  = 1024;

static_assert(BM % NWARPS == 0 && ROWS_PER_WARP % 16 == 0, "bad BM");
static_assert(NTHREADS == BM, "the W staging maps one thread to one row");
static_assert(BN % 16 == 0 && BN <= 128, "bad BN");
static_assert(sizeof(block_q8_1_mmq) == Y_STRIDE && QK8_1_MMQ == BK, "Q8_1_MMQ layout");
static_assert(LDS_BYTES <= 32768, "two workgroups per CU");

#if defined(GGML_USE_HIP)

typedef __attribute__((ext_vector_type(4))) int i32x4;
typedef __attribute__((ext_vector_type(8))) int i32x8;

__device__ __forceinline__ uint32_t perm(uint32_t a, uint32_t b, uint32_t sel) { return __builtin_amdgcn_perm(a, b, sel); }

// 8 nibbles -> 8 int8 through the 16-entry kvalues_iq4nl table: lo = even (low) nibbles, hi = odd (high) nibbles.
__device__ __forceinline__ void table16(const uint32_t q4, const uint32_t t0, const uint32_t t1, const uint32_t t2, const uint32_t t3,
                                        uint32_t & lo, uint32_t & hi) {
    const uint32_t qe = q4, qo = q4 >> 4;
    const uint32_t ve_l = perm(t1, t0, qe & 0x07070707u), vo_l = perm(t1, t0, qo & 0x07070707u);
    const uint32_t ve_h = perm(t3, t2, qe & 0x07070707u), vo_h = perm(t3, t2, qo & 0x07070707u);
    lo = perm(ve_h, ve_l, 0x03020100u | ((qe & 0x08080808u) >> 1));
    hi = perm(vo_h, vo_l, 0x03020100u | ((qo & 0x08080808u) >> 1));
}

// A = 16 tokens x K (activations), B = K x 16 expert rows (weights): D lane l holds expert row l%16 for tokens 2e + l/16.
__device__ __forceinline__ i32x8 wmma_iu8(const i32x4 a, const i32x4 b, const i32x8 c) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#else
    GGML_UNUSED_VARS(a, b);
    NO_DEVICE_CODE;
    return c;
#endif
}

// x: IQ4_NL experts [n_expert][nrows_x][ncols_x]; y: block_q8_1_mmq [ncols_x/128][ncols_y] (k-block major, compact
// rows sorted by expert); ids_dst[compact] = dst row; tile_spec[blockIdx.y] = {expert, col_low, col_high}.
__global__ __launch_bounds__(NTHREADS, 2) void mul_mat_id_iq4nl(
        const char * __restrict__ x, const char * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, float * __restrict__ dst,
        const int ncols_x, const int stride_row_x, const int64_t stride_expert_x, const int ncols_y, const int stride_col_dst) {
    extern __shared__ __align__(16) char lds[];
    auto ids_s = [&](const int j) -> int & { return *(int *) (lds + j*W_STRIDE + 64); };   // pad bytes, never staged

    const ggml_cuda_mm_id_tile_spec ts = tile_spec[blockIdx.y];
    const int ntok = ts.col_high - ts.col_low;
    if (ntok <= 0) {
        return;
    }
    const int ntt = (ntok + 15) / 16;

    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lane = tid % 32;
    const int r    = lane % 16;
    const int hl   = lane / 16;

    if (tid < BN) {
        ids_s(tid) = ids_dst[ts.col_low + tid];   // padded allocation: reads past col_high are in bounds
    }

    const int row0 = blockIdx.x * BM;
    const char * xrow = x + (int64_t) ts.expert * stride_expert_x + (int64_t) (row0 + tid) * stride_row_x;
    const char * ycol = y + (int64_t) ts.col_low * Y_STRIDE;
    const int64_t ystride_chunk = (int64_t) ncols_y * Y_STRIDE;
    const int nchunks = ncols_x / BK;

    const uint32_t * kv = (const uint32_t *) kvalues_iq4nl;
    const uint32_t t0 = kv[0], t1 = kv[1], t2 = kv[2], t3 = kv[3];

    // prefetch registers
    uint2 wreg[9];
    i32x4 yreg[3];

    auto load_chunk = [&](const int c) {
        const uint2 * src = (const uint2 *) (xrow + c * W_CHUNK);
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            wreg[i] = src[i];
        }
        const i32x4 * ysrc = (const i32x4 *) (ycol + c * ystride_chunk);
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int idx = i*NTHREADS + tid;
            if (idx < BN*Y_STRIDE/16) {
                yreg[i] = ysrc[idx];
            }
        }
    };
    auto store_chunk = [&](const int buf) {
        char * wqs = lds + buf*LDS_BUF;
        char * wd  = wqs + LDS_W_QS;
        char * yl  = wd + LDS_W_D;
        const uint32_t * w = (const uint32_t *) wreg;
        i32x4 * q = (i32x4 *) (wqs + tid * W_STRIDE);
        // block 0: bytes 0..17 (d at 0, qs at 2); block 1: 18..35 (qs at 20, dword aligned); 2: 36..53; 3: 54..71
        q[0] = i32x4{(int) __builtin_amdgcn_alignbit(w[1],  w[0],  16), (int) __builtin_amdgcn_alignbit(w[2],  w[1],  16),
                     (int) __builtin_amdgcn_alignbit(w[3],  w[2],  16), (int) __builtin_amdgcn_alignbit(w[4],  w[3],  16)};
        q[1] = i32x4{(int) w[5], (int) w[6], (int) w[7], (int) w[8]};
        q[2] = i32x4{(int) __builtin_amdgcn_alignbit(w[10], w[9],  16), (int) __builtin_amdgcn_alignbit(w[11], w[10], 16),
                     (int) __builtin_amdgcn_alignbit(w[12], w[11], 16), (int) __builtin_amdgcn_alignbit(w[13], w[12], 16)};
        q[3] = i32x4{(int) w[14], (int) w[15], (int) w[16], (int) w[17]};
        uint2 * d = (uint2 *) (wd + tid * 8);
        *d = uint2{(w[0] & 0xffffu) | (w[4] & 0xffff0000u), (w[9] & 0xffffu) | (w[13] & 0xffff0000u)};
        i32x4 * yq = (i32x4 *) yl;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int idx = i*NTHREADS + tid;
            if (idx < BN*Y_STRIDE/16) {
                yq[idx] = yreg[i];
            }
        }
    };

    float sum[RT][TT][8];
#pragma unroll
    for (int a = 0; a < RT; ++a) {
#pragma unroll
        for (int b = 0; b < TT; ++b) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                sum[a][b][e] = 0.0f;
            }
        }
    }
    const i32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};

    load_chunk(0);
    store_chunk(0);
    __syncthreads();

    for (int c = 0; c < nchunks; ++c) {
        const int buf = c & 1;
        if (c + 1 < nchunks) {
            load_chunk(c + 1);
        }
        const char * wqs = lds + buf*LDS_BUF;
        const char * wd  = wqs + LDS_W_QS;
        const char * yl  = wd + LDS_W_D;

#pragma unroll
        for (int b = 0; b < BK/32; ++b) {
            i32x4 wlo[RT], whi[RT];
            float dw[RT];
#pragma unroll
            for (int a = 0; a < RT; ++a) {
                const int row = wid*ROWS_PER_WARP + a*16 + r;
                const i32x4 q = *(const i32x4 *) (wqs + row*W_STRIDE + b*16);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t lo, hi;
                    table16((uint32_t) q[i], t0, t1, t2, t3, lo, hi);
                    wlo[a][i] = (int) lo;
                    whi[a][i] = (int) hi;
                }
                const unsigned short dh = *(const unsigned short *) (wd + row*8 + b*2);
                dw[a] = __half2float(__ushort_as_half(dh));
            }
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                if (t >= ntt) {
                    break;
                }
                const char * yt = yl + (t*16 + r)*Y_STRIDE + 16 + b*32;
                const i32x4 ylo = *(const i32x4 *) yt;
                const i32x4 yhi = *(const i32x4 *) (yt + 16);
                float dy[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    dy[e] = *(const float *) (yl + (t*16 + 2*e + hl)*Y_STRIDE + b*4);
                }
#pragma unroll
                for (int a = 0; a < RT; ++a) {
                    i32x8 acc = wmma_iu8(ylo, wlo[a], zero);
                    acc       = wmma_iu8(yhi, whi[a], acc);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        sum[a][t][e] += (float) acc[e] * (dw[a] * dy[e]);
                    }
                }
            }
        }

        if (c + 1 < nchunks) {
            store_chunk(buf ^ 1);
        }
        __syncthreads();
    }

    // D fragment: lane holds column r (expert row) for tokens 2e + hl of each 16-token tile
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        if (t >= ntt) {
            break;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int j = t*16 + 2*e + hl;
            if (j < ntok) {
                float * drow = dst + (int64_t) ids_s(j) * stride_col_dst + row0 + wid*ROWS_PER_WARP + r;
#pragma unroll
                for (int a = 0; a < RT; ++a) {
                    drow[a*16] = sum[a][t][e];
                }
            }
        }
    }
}

// Expert-aligned column tiles: ceil(n_e/BN) tiles per expert, each starting at the expert's first compact column.
// mm_ids_helper's tiles are aligned to multiples of BN in the global compact index and split at every expert
// boundary, which with ~32 tokens per expert pads the WMMA work to ~2x. The compact order (ids_src1/ids_dst from
// mm_ids_helper) is the same prefix-sum order, so only tile_spec is replaced. One block; empty tiles have
// col_low == col_high.
__global__ __launch_bounds__(PLAN_THREADS) void plan_tiles(
        const int32_t * __restrict__ ids, const int n_tokens, const int n_expert_used, const int si1, const int n_experts,
        ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, const int ntj) {
    __shared__ int count[MAX_EXPERTS];
    __shared__ int start[MAX_EXPERTS];
    __shared__ int tbase[MAX_EXPERTS + 1];
    for (int e = threadIdx.x; e < n_experts; e += PLAN_THREADS) {
        count[e] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_tokens*n_expert_used; i += PLAN_THREADS) {
        const int e = ids[(i / n_expert_used) * si1 + i % n_expert_used];
        atomicAdd(&count[e], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {   // serial scan: n_experts is small
        int s = 0, t = 0;
        for (int e = 0; e < n_experts; ++e) {
            start[e] = s; tbase[e] = t;
            s += count[e]; t += (count[e] + BN - 1) / BN;
        }
        tbase[n_experts] = t;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_experts; e += PLAN_THREADS) {
        for (int i = 0; i < (count[e] + BN - 1) / BN; ++i) {
            tile_spec[tbase[e] + i] = {e, start[e] + i*BN, min(start[e] + (i + 1)*BN, start[e] + count[e]), 0};
        }
    }
    for (int t = tbase[n_experts] + threadIdx.x; t < ntj; t += PLAN_THREADS) {
        tile_spec[t] = {0, 0, 0, 0};
    }
}

#endif // defined(GGML_USE_HIP)

} // namespace inkling_moe

bool ggml_cuda_mmq_inkling_moe_applies(ggml_type type, int cc, int64_t ne00, int64_t ne01, int64_t ne02, ggml_prec prec_src1, bool has_fusion) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_INKLING_MOE");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled && !has_fusion && type == GGML_TYPE_IQ4_NL && prec_src1 == GGML_PREC_Q8 &&
        GGML_CUDA_CC_IS_RDNA3_5(cc) && amd_wmma_available(cc) &&
        ne00 % inkling_moe::BK == 0 && ne01 % inkling_moe::BM == 0 && ne00 > 0 && ne02 <= inkling_moe::MAX_EXPERTS;
}

void ggml_cuda_mmq_inkling_moe(
        const char * x, const char * y_q8, const int32_t * ids_dst, const int32_t * ids, int si1,
        ggml_cuda_mm_id_tile_spec * tile_spec, float * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t n_tokens, int64_t n_expert_used,
        int64_t nb01, int64_t nb02, int64_t s1, int ntj, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    using namespace inkling_moe;
    static const bool debug = [] { const char * e = getenv("GGML_CUDA_INKLING_MOE_DEBUG"); return e != nullptr && atoi(e) != 0; }();
    if (debug) {
        fprintf(stderr, "inkling moe: taken ne00 %lld ne01 %lld ne02 %lld n_tokens %lld n_expert_used %lld\n",
            (long long) ne00, (long long) ne01, (long long) ne02, (long long) n_tokens, (long long) n_expert_used);
    }
    GGML_ASSERT(ne00 % BK == 0 && ne01 % BM == 0 && ne02 <= MAX_EXPERTS);
    GGML_ASSERT(ntj >= (n_tokens*n_expert_used + BN - 1) / BN + ne02);   // enough tile_spec entries for the expert-aligned plan
    GGML_ASSERT(ne01 / BM <= 65535 && ntj <= 65535);

    plan_tiles<<<1, PLAN_THREADS, 0, stream>>>(ids, (int) n_tokens, (int) n_expert_used, si1, (int) ne02, tile_spec, ntj);
    CUDA_CHECK(cudaGetLastError());

    CUDA_SET_SHARED_MEMORY_LIMIT(mul_mat_id_iq4nl, LDS_BYTES);
    const dim3 grid(ne01 / BM, ntj, 1);
    mul_mat_id_iq4nl<<<grid, NTHREADS, LDS_BYTES, stream>>>(
        x, y_q8, ids_dst, tile_spec, dst, (int) ne00, (int) nb01, nb02, (int) (n_tokens*n_expert_used), (int) s1);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(x, y_q8, ids_dst, ids, si1, tile_spec, dst, ne00, ne01, ne02, n_tokens, n_expert_used, nb01, nb02, s1, ntj, stream);
    GGML_ABORT("ggml_cuda_mmq_inkling_moe is HIP-only");
#endif
}
