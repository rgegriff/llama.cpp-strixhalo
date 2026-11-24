#!/bin/bash
# Burst 16: NTT-templated body (v4), partial b-unroll (u2, u1), 8 waves (w8, w8 with VGPR cap): correctness + interleaved timing, all with --plan.
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1 --reps 15"; D="--m 2304 --k 1152 --n 512 --bcast 0 --reps 15"
STRIP='s/nused=8 nmat=128 bcast=. skew=0.00 | J=[0-9]* (I=[0-9]*, [0-9]* thr)//; s/ms, mmq alone/|/; s/ TFLOPS, weights [0-9]* MB ->/ TF/; s/| tiles.*//'
echo "== correctness"
for v in moe_wmma_v4 moe_wmma_v4u2 moe_wmma_v4u1 moe_wmma_v4w8 moe_wmma_v4w8o8; do
  echo "-- $v"; ./$v --kernel wmma $F --plan --reps 2 --check --cmp 2>&1 | grep -v "^device\|Signals leaked\|^wmma" | cut -c1-120
done
for round in 1 2 3; do
  echo "== round $round"
  for S in "$F" "$D"; do
    ./moe_wmma --kernel mmq --J 16 $S 2>&1 | grep -v "^device\|Signals leaked" | sed "$STRIP"
    for v in moe_wmma moe_wmma_v4 moe_wmma_v4u2 moe_wmma_v4u1 moe_wmma_v4w8 moe_wmma_v4w8o8; do
      ./$v --kernel wmma $S --plan 2>&1 | grep -v "^device\|Signals leaked" | sed "s/^wmma/${v#moe_wmma}+plan/; $STRIP"
    done
  done
done
