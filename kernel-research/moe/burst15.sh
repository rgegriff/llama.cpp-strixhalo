#!/bin/bash
# Burst 15: expert-aligned tile planner (--plan) and the half2 variant: correctness, then interleaved timing vs MMQ J=16.
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1 --reps 15"; D="--m 2304 --k 1152 --n 512 --bcast 0 --reps 15"
STRIP='s/nused=8 nmat=128 bcast=. skew=0.00 | J=[0-9]* (I=[0-9]*, 128 thr)//; s/ms, mmq alone/|/; s/ TFLOPS, weights [0-9]* MB ->/ TF/'
echo "== correctness"
./moe_wmma --kernel wmma $F --plan --reps 3 --check --cmp 2>&1 | grep -v "^device\|Signals leaked" | sed "$STRIP"
./moe_wmma --kernel wmma $D --plan --reps 3 --check --cmp 2>&1 | grep -v "^device\|Signals leaked" | sed "$STRIP"
./moe_wmma_half2 --kernel wmma $D --reps 3 --check --cmp 2>&1 | grep -v "^device\|Signals leaked" | sed "$STRIP"
for round in 1 2 3; do
  echo "== round $round"
  for S in "$F" "$D"; do
    ./moe_wmma --kernel mmq --J 16 $S 2>&1 | grep -v "^device\|Signals leaked" | sed "$STRIP"
    ./moe_wmma --kernel wmma $S 2>&1 | grep -v "^device\|Signals leaked" | sed "s/^wmma/v1/; $STRIP"
    ./moe_wmma --kernel wmma $S --plan 2>&1 | grep -v "^device\|Signals leaked" | sed "s/^wmma/v1+plan/; $STRIP"
    ./moe_wmma_half2 --kernel wmma $S --plan 2>&1 | grep -v "^device\|Signals leaked" | sed "s/^wmma/half2+plan/; $STRIP"
  done
done
