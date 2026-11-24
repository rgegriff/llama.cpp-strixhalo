#!/bin/bash
# Burst 5: time decomposition of the MMQ kernel: no global->LDS loads vs no MMA, at J=32 and J=16.
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1 --reps 15"; D="--m 2304 --k 1152 --n 512 --bcast 0 --reps 15"
for J in 32 16; do
  echo "== gate_up J=$J"
  for v in "" _V_NOLOAD _V_NOVECDOT; do printf "%-12s " "${v:-full}"; ./moe_mmq$v $F --J $J | grep -v ^device | cut -c90-; done
  echo "== down J=$J"
  for v in "" _V_NOLOAD _V_NOVECDOT; do printf "%-12s " "${v:-full}"; ./moe_mmq$v $D --J $J | grep -v ^device | cut -c90-; done
done
