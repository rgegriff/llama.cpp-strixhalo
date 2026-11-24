#!/bin/bash
# Burst 8: is IQ4_NL itself slow? Same harness, same tiles, four weight types, J=16 and J=32, two interleaved rounds.
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1 --reps 12"; D="--m 2304 --k 1152 --n 512 --bcast 0 --reps 12"
for round in 1 2; do
  for J in 16 32; do
    echo "== round $round gate_up J=$J"; for t in iq4_nl q4_0 q4_K q8_0; do ./moe_mmq --type $t $F --J $J | grep -v ^device | cut -c1-40,110-; done
    echo "== round $round down J=$J";    for t in iq4_nl q4_0 q8_0;      do ./moe_mmq --type $t $D --J $J | grep -v ^device | cut -c1-40,110-; done
  done
done
