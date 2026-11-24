#!/bin/bash
# Burst 2: real routing + per-op wall time in the model. llama-perplexity on the IQ4_NL model with
# ubatch = n_tokens (consecutive tokens of one sequence = the verify-batch shape), GGML_CUDA_OP_STATS=2 (per-op time,
# synchronizes after every op) and GGML_CUDA_MMID_ROUTE_STATS=1 (distinct experts per MUL_MAT_ID call).
cd "$(dirname "$(readlink -f "$0")")"
K=../../build-rocm-moesb/bin; M=${MODEL:?path to Inkling-2-Mini-IQ4_NL.gguf}
TXT=${TXT:?a text file of a few thousand tokens}
Q=${QUIET:-}   # optional wrapper that serializes GPU runs
export GGML_CUDA_INKLING2_FUSION=1 GGML_CUDA_FATTN_VEC_GQA=0 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_OP_STATS=2 GGML_CUDA_MMID_ROUTE_STATS=1
echo "== burst2 start $(date +%T)"
for ub in ${UBS:-1 2 4 6 8}; do
  echo "== ub $ub $(date +%T)"
  $Q $K/llama-perplexity -m $M -f $TXT -ngl 99 -fa on -c 2048 --chunks 1 -b $ub -ub $ub -ctk q8_0 -ctv q8_0 > logs/ppl-ub$ub.log 2>&1 || echo FAIL ub $ub
  grep -E "mmid-route|MUL_MAT_ID|^\[.*estimate|graphs" logs/ppl-ub$ub.log | head -40
done
echo "== burst2 end $(date +%T)"
