#!/bin/bash
# Burst 19: test-backend-ops MUL_MAT_ID with the new IQ4_NL 128-multiple cases; GGML_CUDA_INKLING_MOE_DEBUG=1 shows which tests the dedicated kernel takes.
cd "$(dirname "$(readlink -f "$0")")"
B=../../build-rocm-moe/bin
GGML_CUDA_INKLING_MOE_DEBUG=1 $B/test-backend-ops test -b ROCm0 -o MUL_MAT_ID > logs/burst19_tbo.txt 2>&1
echo "taken calls: $(grep -c 'inkling moe: taken' logs/burst19_tbo.txt)"; grep 'inkling moe: taken' logs/burst19_tbo.txt | sort | uniq -c | sort -rn | head -12
grep -E "iq4_nl.*m=(512|256),n=" logs/burst19_tbo.txt | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-110
grep -E "FAIL|tests passed|Backend ROCm0" logs/burst19_tbo.txt | sed 's/\x1b\[[0-9;]*m//g'
echo "== with the kernel disabled"
GGML_CUDA_INKLING_MOE=0 $B/test-backend-ops test -b ROCm0 -o MUL_MAT_ID 2>&1 | grep -E "FAIL|tests passed" | sed 's/\x1b\[[0-9;]*m//g'
