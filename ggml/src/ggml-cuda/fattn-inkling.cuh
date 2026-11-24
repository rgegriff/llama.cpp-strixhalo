#pragma once

#include "common.cuh"

// Decode attention for Inkling 2: one query, D = 128, 24 Q heads over 8 KV heads, an indexed bias,
// K and V both q8_0 or both q4_0, at a long context. One block per chunk of positions covers all 8 KV heads, so the
// cache (whose rows are per position, all heads back to back) is read once front to back; split-K plus a combine.
// Returns false when the op is not that shape and the generic kernels must handle it.
bool ggml_cuda_flash_attn_ext_inkling_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
