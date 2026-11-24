#pragma once

#include "common.cuh"

// Decode attention for Inkling 2: one query, D = 128, 24 Q heads over 8 KV heads, an indexed bias,
// K and V both q8_0 or both q4_0, at a long context. One block per chunk of positions covers all 8 KV heads, so the
// cache (whose rows are per position, all heads back to back) is read once front to back; split-K plus a combine.
// Returns false when the op is not that shape and the generic kernels must handle it.
bool ggml_cuda_flash_attn_ext_inkling_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Small-batch decode attention for Inkling 2: 2 <= n_q <= 16 queries (speculative-decoding verify),
// n_kv a multiple of 32 at a long context, D = 128, 24/8 heads, indexed bias, K and V both q8_0 or both q4_0; f16 WMMA on
// RDNA3, one block per chunk of positions for a group of KV heads, split-K plus a combine. Returns false when it does not apply.
bool ggml_cuda_flash_attn_ext_inkling_smallbatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Prefill attention for Inkling 2: n_q a multiple of 32, n_kv a multiple of 64, D = 128, 24/8 heads,
// indexed bias, f16 (per-position layout) or q8_0 K/V; WMMA on RDNA3. Returns false when it does not apply.
bool ggml_cuda_flash_attn_ext_inkling_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
