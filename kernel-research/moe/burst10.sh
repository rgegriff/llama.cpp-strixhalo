#!/bin/bash
# Burst 10: ablations of the WMMA kernel v1 (no global loads / no MMA+scaling / no dequant) + sclk during the base run.
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1 --reps 12"; D="--m 2304 --k 1152 --n 512 --bcast 0 --reps 12"
( while sleep 0.3; do grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | tr -d ' *\n'; echo -n ' '; done ) > logs/burst10.sclk &
SCLK=$!
for v in moe_wmma moe_wmma_MOE_NO_LOAD moe_wmma_MOE_NO_MMA moe_wmma_MOE_NO_DEQ; do
  echo "== $v"; ./$v --kernel wmma $F 2>&1 | grep -v "^device\|Signals leaked" | cut -c1-60,150-260; ./$v --kernel wmma $D 2>&1 | grep -v "^device\|Signals leaked" | cut -c1-60,150-260
done
kill $SCLK
echo "sclk samples:"; tr ' ' '\n' < logs/burst10.sclk | sort | uniq -c | sort -rn | head -6
