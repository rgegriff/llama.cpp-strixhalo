# llama.cpp for AMD Strix Halo (gfx1151, ROCm)

This is a personal fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), tuned for one machine class:
AMD Strix Halo (Radeon 8060S, `gfx1151`, RDNA 3.5) with the ROCm/HIP backend and unified memory. It is rebased on
upstream master from time to time. Everything not listed here is upstream's.

**Read this first**

- The kernels and fusions added here were written by AI agents (Claude) under the direction of the fork's owner. They
  have **not** been reviewed by the llama.cpp maintainers.
- They are measured on gfx1151 only. Paths gated on RDNA 3 / RDNA 3.5 are untested on other RDNA generations, and none
  of this was tried on NVIDIA.
- **Do not report issues with this fork to ggml-org/llama.cpp.** If something breaks, turn the relevant switch off
  (below); if it still breaks on an upstream build, it is an upstream issue, otherwise it is this fork's.

## What is added

Commit subjects carry a scope prefix:

| prefix | what |
|---|---|
| `hip/rdna3.5:` | changes that help any model on gfx1151 (e.g. J=16 MMQ tiles for `MUL_MAT_ID`) |
| `inkling2:` | Inkling 2 / Inkling 2 Mini: fused glue-op chains, the indexed-bias flash-attention kernels (decode, small-batch verify, prefill), the IQ4_NL MoE expert matmul |
| `kern-tools:` | measurement tooling (`GGML_CUDA_OP_STATS`, timing in the speculative stats line, perf test cases) |
| `kernel-research:` | standalone prototypes, harnesses and notes under `kernel-research/`; not built by CMake |

The Inkling 2 model graph differs from upstream in two places, both order/layout only: the depth mix keeps the
`[n_embd, n_queries, n_tokens]` layout (the HIP backend runs it as one fused kernel), and independent nodes are expanded
in an order that keeps each glue chain contiguous so the backend can fuse it.

## Build

The usual HIP build, with rocWMMA for the WMMA kernels:

```
cmake -B build-rocm -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 \
      -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_HIP_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA_FA_ALL_QUANTS=OFF "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16"
cmake --build build-rocm --target llama-server llama-bench test-backend-ops
```

## Runtime switches

All are environment variables read once per process. Unset means the default.

| variable | default | effect | to disable |
|---|---|---|---|
| `GGML_CUDA_INKLING2_FUSION` | on | fuse the Inkling 2 glue-op chains (clip, short conv + state store, SiTU, depth mix, loop controller, bias table) into single kernels | `=0` |
| `GGML_CUDA_INKLING2_SITU_PLAIN` | on | fuse the SiTU chain of models without a per-expert scale (Inkling 2 Mini): 8 dispatches per layer visit become 1 | `=0` |
| `GGML_CUDA_TOPK_MOE_NORM_EPS` | on | the top-k MoE router kernel also takes the weight normalization by `sum + eps` (`expert_weights_norm_eps`), not only the clamped sum | `=0` |
| `GGML_CUDA_MMVQ_Q8_CACHE` | on | consecutive quantized matrix-vector products that read the same activations (the projections of an attention block) quantize them to Q8_1 once; bit-identical | `=0` |
| `GGML_CUDA_INKLING2_CONV_MT` | `1` | bit mask for the short-conv fusion on 2..32-token batches (speculative-decoding verify): bit 0 `CONV_STATE` (bit-identical), bit 1 `CONV_DYN` (opt-in: changes the logits slightly, KL ~0.004, cause not identified). `3` enables both | `=0` |
| `GGML_CUDA_FATTN_INKLING` | on | one-query decode attention over a q8_0/q4_0 KV cache with the indexed bias, contexts of `GGML_CUDA_FATTN_INKLING_MIN_KV` (65536) keys and more | `=0` |
| `GGML_CUDA_FATTN_INKLING_SMALL` | on | the same for 2..16 queries (speculative-decoding verify), from `GGML_CUDA_FATTN_INKLING_SMALL_MIN_KV` (32768) keys | `=0` |
| `GGML_CUDA_FATTN_INKLING_PREFILL` | on | WMMA prefill attention with the indexed bias; `GGML_CUDA_FATTN_INKLING_BAND_MAX_KV` (16384) bounds the band bias path | `=0` |
| `GGML_CUDA_FATTN_VEC_GQA` | on | GQA variant of the vector flash-attention kernel when decoding with an indexed bias | `=0` |
| `GGML_CUDA_INKLING_MOE` | on | dedicated IQ4_NL `MUL_MAT_ID` WMMA kernel on RDNA 3.5 | `=0` |
| `LLAMA_DSPARK_GREEDY_CHAIN` | off | Inkling 2 DSpark draft: chain the rnn Markov head on the argmax and build no in-graph draws (fewer ops per draft). Probabilistic drafting (`--spec-draft-sampling probabilistic`) is then ignored | leave unset |
| `GGML_CUDA_MMID_Q8_MMQ_MIN` | `4` | RDNA 3.5: Q8_0 `MUL_MAT_ID` goes to MMQ from this many tokens (MMVQ below) | `=0` (stock table) |
| `LLAMA_INKLING_KV_ROT` | off | experiment: Hadamard rotation of Q/K (bit 1) and V (bit 2) ahead of the KV cache | leave unset |
| `GGML_CUDA_OP_STATS` | off | `1`: count dispatched ops and fused chains per graph evaluation; `2`: also synchronize and time each kind of op (slow) | leave unset |
| `GGML_CUDA_OP_STATS_SHAPES`, `GGML_CUDA_OP_STATS_TRACE` | off | with `GGML_CUDA_OP_STATS`: key the matmul and attention ops by shape at level 1 too; `TRACE=<n>` prints one line per dispatch of graph evaluation n. While a HIP graph is captured the stats also count the kernels each op launches | leave unset |
| `GGML_CUDA_STEP_TIMING` | off | `1`: one line per graph evaluation with its mode (direct, capture, replay), the host time to enqueue it and the time the stream needs to drain; synchronizes around every evaluation | leave unset |
| `GGML_CUDA_MMID_ROUTE_STATS` | off | `1`: per-shape histogram of experts touched by small `MUL_MAT_ID` batches, printed at exit (synchronizes the stream) | leave unset |
| `GGML_CUDA_FATTN_INKLING_DEBUG`, `GGML_CUDA_INKLING_MOE_DEBUG`, `GGML_CUDA_INKLING2_CONV_MT_DEBUG` | off | log which calls the dedicated kernels take or reject | leave unset |

`GGML_CUDA_DISABLE_FUSION=1` (upstream) turns off every fusion, including this fork's. The J=16 `MUL_MAT_ID` tile choice
on RDNA 3.5 has no switch. With every switch above at `0` the fork's dedicated kernels are bypassed; the indexed-bias
support in the stock tile/vec flash-attention kernels remains, since upstream has no CUDA/HIP implementation of it.

## Tests

`test-backend-ops -o FLASH_ATTN_EXT`, `-o MUL_MAT_ID` and `-o INKLING2_<chain>` compare each added path against the CPU
backend.
