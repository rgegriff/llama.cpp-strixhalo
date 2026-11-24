#!/bin/bash
# Burst 18: the ggml port in build-rocm-moe: test-backend-ops MUL_MAT_ID (dedicated kernel enabled), then moe_bench through libggml with the kernel on/off.
cd "$(dirname "$(readlink -f "$0")")"
B=../../build-rocm-moe/bin
echo "== test-backend-ops -o MUL_MAT_ID (GGML_CUDA_INKLING_MOE default on)"
$B/test-backend-ops test -b ROCm0 -o MUL_MAT_ID 2>&1 | grep -E "iq4_nl.*(FAIL|OK)" | head -5
$B/test-backend-ops test -b ROCm0 -o MUL_MAT_ID 2>&1 | grep -E "FAIL|tests passed|Backend ROCm0" | head -20
echo "== moe_bench through libggml, 512 tokens"
for env in 1 0; do
  for S in "--m 2304 --k 2304 --bcast 1" "--m 2304 --k 1152 --bcast 0"; do
    GGML_CUDA_INKLING_MOE=$env ./moe_bench_moe --type iq4_nl $S --n 512 --reps 12 $([ $env = 1 ] && echo --check) 2>&1 | grep -v "^ggml_\|^load_backend\|^register" | sed "s/^\[x4\]/INKLING_MOE=$env [x4]/" | cut -c1-200
  done
done
