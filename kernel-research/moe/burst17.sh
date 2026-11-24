#!/bin/bash
# Burst 17: MMQ J=16 baseline vs the WMMA kernel (+plan) at 256 / 512 / 2048 tokens, both shapes, 2 interleaved rounds.
cd "$(dirname "$(readlink -f "$0")")"
STRIP='s/nused=8 nmat=128 bcast=. skew=0.00 | J=[0-9]* (I=[0-9]*, [0-9]* thr)//; s/ms, mmq alone/|/; s/ TFLOPS, weights [0-9]* MB ->/ TF/; s/| tiles.*//'
for round in 1 2; do
  echo "== round $round"
  for n in 256 512 2048; do
    for S in "--m 2304 --k 2304 --bcast 1" "--m 2304 --k 1152 --bcast 0"; do
      ./moe_wmma --kernel mmq --J 16 $S --n $n --reps 12 2>&1 | grep -v "^device\|Signals leaked" | sed "$STRIP"
      ./moe_wmma --kernel wmma $S --n $n --reps 12 --plan 2>&1 | grep -v "^device\|Signals leaked" | sed "s/^wmma/wmma+plan/; $STRIP"
    done
  done
done
echo "== skewed routing (expert popularity ~ (e+1)^-0.5), 512 tokens"
for S in "--m 2304 --k 2304 --bcast 1" "--m 2304 --k 1152 --bcast 0"; do
  ./moe_wmma --kernel mmq --J 16 $S --n 512 --skew 0.5 --reps 12 2>&1 | grep -v "^device\|Signals leaked" | sed "$STRIP"
  ./moe_wmma --kernel wmma $S --n 512 --skew 0.5 --reps 12 --plan --check 2>&1 | grep -v "^device\|Signals leaked" | sed "s/^wmma/wmma+plan/; $STRIP"
done
