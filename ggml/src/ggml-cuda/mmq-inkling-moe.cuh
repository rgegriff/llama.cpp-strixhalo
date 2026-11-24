#pragma once

// Dedicated MUL_MAT_ID kernel for IQ4_NL experts on RDNA3.5 WMMA (Inkling 2 Mini prefill).
// Takes over the matmul of ggml_cuda_mul_mat_q's ids branch after mm_ids_helper and the Q8_1_MMQ quantization:
// same inputs, same numerics (int8 WMMA per 32-block, d_w*d_y scaling in f32; identical to MMQ up to summation order).
// Prototype, ablations and measurements: kernel-research/moe/ (moe_wmma_kernel.cuh, NOTES.md).
// GGML_CUDA_INKLING_MOE=0 disables it (the MMQ J=16 path is then taken); GGML_CUDA_INKLING_MOE_DEBUG=1 logs every taken call.

#include "common.cuh"
#include "mmid.cuh"

// column tile width the ids helper must use when this kernel is taken
constexpr int GGML_CUDA_INKLING_MOE_J        = 32;
constexpr int GGML_CUDA_INKLING_MOE_NTHREADS = 128;

bool ggml_cuda_mmq_inkling_moe_applies(ggml_type type, int cc, int64_t ne00, int64_t ne01, int64_t ne02, ggml_prec prec_src1, bool has_fusion);

// x: IQ4_NL experts; y_q8: block_q8_1_mmq [ne00/128][ne_get_rows] (as quantize_mmq_q8_1 writes it for MUL_MAT_ID);
// ids_dst/tile_spec: the mm_ids_helper outputs (tile_spec is overwritten with expert-aligned tiles, ntj entries).
void ggml_cuda_mmq_inkling_moe(
        const char * x, const char * y_q8, const int32_t * ids_dst, const int32_t * ids, int si1,
        ggml_cuda_mm_id_tile_spec * tile_spec, float * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t n_tokens, int64_t n_expert_used,
        int64_t nb01, int64_t nb02, int64_t s1, int ntj, cudaStream_t stream);
