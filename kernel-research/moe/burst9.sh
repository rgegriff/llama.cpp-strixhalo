#!/bin/bash
# Burst 9: first correctness run of the dedicated WMMA kernel, both WMMA operand orders, gate_up and down shapes.
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1 --reps 5 --warm 1"; D="--m 2304 --k 1152 --n 512 --bcast 0 --reps 5 --warm 1"
for v in moe_wmma moe_wmma_MOE_WMMA_SWAP_AB; do
  echo "== $v gate_up"; ./$v --kernel wmma $F --check --cmp 2>&1 | grep -v "^device\|Signals leaked"
  echo "== $v down";    ./$v --kernel wmma $D --check --cmp 2>&1 | grep -v "^device\|Signals leaked"
done
echo "== mmq J=16 baseline"; ./moe_wmma --kernel mmq --J 16 $F --check 2>&1 | grep -v "^device\|Signals leaked"; ./moe_wmma --kernel mmq --J 16 $D --check 2>&1 | grep -v "^device\|Signals leaked"
