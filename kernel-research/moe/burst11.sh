#!/bin/bash
# Burst 11: interleaved A/B rounds (no clock sampler): MMQ J=16 baseline vs WMMA v1 and its ablations, both shapes.
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1 --reps 15"; D="--m 2304 --k 1152 --n 512 --bcast 0 --reps 15"
for round in 1 2 3; do
  echo "== round $round"
  for S in "$F" "$D"; do
    ./moe_wmma --kernel mmq --J 16 $S 2>&1 | grep -v "^device\|Signals leaked" | sed 's/nused=8 nmat=128 bcast=. skew=0.00 | J=[0-9]* (I=[0-9]*, 128 thr) tiles [0-9]* x [0-9]* row-tiles = [0-9]* blocks ([0-9]* non-empty) |//; s/ms, mmq alone/|/'
    for v in moe_wmma moe_wmma_MOE_NO_LOAD moe_wmma_MOE_NO_MMA moe_wmma_MOE_NO_DEQ; do
      ./$v --kernel wmma $S 2>&1 | grep -v "^device\|Signals leaked" | sed "s/^wmma/${v#moe_wmma}/; s/nused=8 nmat=128 bcast=. skew=0.00 | J=[0-9]* (I=[0-9]*, 128 thr) tiles [0-9]* x [0-9]* row-tiles = [0-9]* blocks ([0-9]* non-empty) |//; s/ms, mmq alone/|/"
    done
  done
done
