#!/bin/bash
# Burst 4: test-backend-ops for the ops touched: Inkling conv chains (multi-token kernel on, then off) and MUL_MAT_ID
# (Q8_0 >= 4 tokens on MMQ by default).
cd "$(dirname "$(readlink -f "$0")")"
T=../../build-rocm-moesb/bin/test-backend-ops; Q=${QUIET:-}
echo "== burst4 start $(date +%T)"
for ops in INKLING2_CONV_STATE INKLING2_CONV_DYN INKLING2_CONV MUL_MAT_ID; do
  echo "== $ops"; $Q $T test -b ROCm0 -o $ops 2>&1 | grep -E "FAIL|tests passed|Backend .*: " | tail -8
done
echo "== INKLING2_CONV_STATE/DYN with GGML_CUDA_INKLING2_CONV_MT=0"
for ops in INKLING2_CONV_STATE INKLING2_CONV_DYN; do GGML_CUDA_INKLING2_CONV_MT=0 $Q $T test -b ROCm0 -o $ops 2>&1 | grep -E "FAIL|tests passed" | tail -4; done
echo "== burst4 end $(date +%T)"
