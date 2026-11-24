#!/bin/bash
# Burst 13: transposed Y scales (MOE_DY_T, 2 b128 reads per token tile instead of 16 b32) on top of HALF_DEQ: correctness, then interleaved timing.
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1 --reps 15"; D="--m 2304 --k 1152 --n 512 --bcast 0 --reps 15"
STRIP='s/nused=8 nmat=128 bcast=. skew=0.00 | J=[0-9]* (I=[0-9]*, 128 thr) tiles [0-9]* x [0-9]* row-tiles = [0-9]* blocks ([0-9]* non-empty) |//; s/ms, mmq alone/|/'
echo "== correctness v3"
./moe_wmma_v3 --kernel wmma $F --reps 3 --check --cmp 2>&1 | grep -v "^device\|Signals leaked" | grep -v "^wmma"
./moe_wmma_v3 --kernel wmma $D --reps 3 --check --cmp 2>&1 | grep -v "^device\|Signals leaked" | grep -v "^wmma"
for round in 1 2 3; do
  echo "== round $round"
  for S in "$F" "$D"; do
    ./moe_wmma --kernel mmq --J 16 $S 2>&1 | grep -v "^device\|Signals leaked" | sed "$STRIP"
    for v in moe_wmma_MOE_HALF_DEQ moe_wmma_v3; do
      ./$v --kernel wmma $S 2>&1 | grep -v "^device\|Signals leaked" | sed "s/^wmma/${v#moe_wmma}/; $STRIP"
    done
  done
done
