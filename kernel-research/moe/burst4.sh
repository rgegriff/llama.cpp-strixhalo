#!/bin/bash
# Burst 4: J=32 kernel variants (register spill fix via occupancy hint, tile shapes, unroll) vs baseline and J=16,
# interleaved twice against clock drift; q4_K vs iq4_nl interleaved; n x J matrix.
cd "$(dirname "$(readlink -f "$0")")"
( for i in $(seq 1 600); do sleep 0.5; echo "$(date +%s.%N | cut -c1-14) $(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk)"; done ) > logs/burst4.sclk &
SCLK=$!
F="--m 2304 --k 2304 --n 512 --bcast 1 --reps 15"; D="--m 2304 --k 1152 --n 512 --bcast 0 --reps 15"
for round in 1 2; do
  echo "== round $round: gate_up"
  for v in "" _V_OCC8 _V_OCC4 _V_J32_256 _V_J32_I32 _V_UNROLL1; do printf "%-11s " "${v:-base}"; ./moe_mmq$v $F --J 32 | grep -v ^device | cut -c90-; done
  printf "%-11s " "base J16"; ./moe_mmq $F --J 16 | grep -v ^device | cut -c90-
  printf "%-11s " "OCC8 J16"; ./moe_mmq_V_OCC8 $F --J 16 | grep -v ^device | cut -c90-
  echo "== round $round: down"
  for v in "" _V_OCC8 _V_OCC4 _V_J32_256 _V_J32_I32 _V_UNROLL1; do printf "%-11s " "${v:-base}"; ./moe_mmq$v $D --J 32 | grep -v ^device | cut -c90-; done
  printf "%-11s " "base J16"; ./moe_mmq $D --J 16 | grep -v ^device | cut -c90-
done
echo "== OCC8 correctness"
./moe_mmq_V_OCC8 $F --J 32 --check | grep check; ./moe_mmq_V_OCC8 $D --J 32 --check | grep check
echo "== OCC8 J=32 with skew 1.0 vs base J16"
./moe_mmq_V_OCC8 $F --J 32 --skew 1.0 | grep -v ^device | cut -c90-; ./moe_mmq $F --J 16 --skew 1.0 | grep -v ^device | cut -c90-
echo "== q4_K vs iq4_nl (ggml path), interleaved"
for r in 1 2; do for t in q4_K iq4_nl; do ./moe_bench --type $t --m 2304 --k 2304 --n 512 --bcast 1 --reps 15 2>&1 | grep -v '^ggml_\|load_backend\|Device 0' | cut -c1-120; done; done
echo "== n x J matrix (base harness, gate_up): n in 256 1024 2048, J in 16 32 64; and OCC8 J32"
for n in 256 1024 2048; do for J in 16 32 64; do printf "n=%-5s J=%-3s " $n $J; ./moe_mmq --m 2304 --k 2304 --n $n --bcast 1 --J $J --reps 8 | grep -v ^device | cut -c90-; done; printf "n=%-5s OCC8 J32 " $n; ./moe_mmq_V_OCC8 --m 2304 --k 2304 --n $n --bcast 1 --J 32 --reps 8 | grep -v ^device | cut -c90-; done
kill $SCLK 2>/dev/null
echo; echo "sclk histogram:"; awk '{print $3}' logs/burst4.sclk | sort | uniq -c | sort -rn | head -8
