#!/bin/bash
# Burst 3: Q8_0 MUL_MAT_ID, MMVQ-moe (table) vs MMQ (GGML_CUDA_MMID_Q8_MMQ_MIN=2), n 2..8, pool 32 (realistic
# sharing: ~30 distinct experts at n=8, weights > MALL) and pool 0 (random).
cd "$(dirname "$(readlink -f "$0")")"
Q=${QUIET:-}   # optional wrapper that serializes GPU runs
echo "== burst3 start $(date +%T)"
for shape in "2304 2304 1" "2304 1152 0"; do set -- $shape
  for n in 2 3 4 5 6 7 8; do
    for pool in 32 0; do
      for mm in 0 2; do
        echo -n "q8mmq_min=$mm "
        GGML_CUDA_MMID_Q8_MMQ_MIN=$mm $Q ./moe_bench_sb --type q8_0 --m $1 --k $2 --bcast $3 --n $n --pool $pool --reps 30 --graph-reps 8 --quiet 2>/dev/null | grep "^\[x"
      done
    done
  done
done
echo "== burst3 end $(date +%T)"
