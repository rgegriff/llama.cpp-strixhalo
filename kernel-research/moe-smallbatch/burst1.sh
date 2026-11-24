#!/bin/bash
# Burst 1: read-bandwidth ceiling, then MUL_MAT_ID per-call time at Inkling 2 Mini shapes for
# n_tokens 1..16, iq4_nl and q8_0, gate_up (k=2304, bcast x) and down (k=1152, per-expert x), with all tokens routed
# to the same 8 experts (pool 8), to 16 experts (pool 16), and random routing (pool 0, ~8n distinct experts).
cd "$(dirname "$(readlink -f "$0")")"
Q=${QUIET:-}   # optional wrapper that serializes GPU runs
echo "== burst1 start $(date +%T)"
$Q ./stream
for ty in iq4_nl q8_0; do
  for shape in "2304 2304 1" "2304 1152 0"; do set -- $shape
    for n in 1 2 3 4 6 7 8 12 16; do
      for pool in 8 16 0; do
        $Q ./moe_bench_sb --type $ty --m $1 --k $2 --bcast $3 --n $n --pool $pool --reps 30 --graph-reps 8
      done
    done
  done
done
echo "== burst1 end $(date +%T)"
