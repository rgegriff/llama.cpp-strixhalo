#!/bin/bash
# Burst 3: standalone harness vs the ggml path, correctness, and the J (column tile) sweep; GPU clock sampled.
cd "$(dirname "$(readlink -f "$0")")"
( while sleep 0.5; do grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | tr -d ' *\n'; echo -n ' '; done ) > logs/burst3.sclk &
SCLK=$!
echo "== harness correctness + baseline (auto J)"
./moe_mmq --m 2304 --k 2304 --n 512 --bcast 1 --check
./moe_mmq --m 2304 --k 1152 --n 512 --bcast 0 --check
echo "== J sweep, gate_up"
for J in 16 32 48 64 96 128; do ./moe_mmq --m 2304 --k 2304 --n 512 --bcast 1 --J $J; done
echo "== J sweep, down"
for J in 16 32 48 64 96 128; do ./moe_mmq --m 2304 --k 1152 --n 512 --bcast 0 --J $J; done
echo "== J sweep, gate_up, skew 1.0"
for J in 16 32 64; do ./moe_mmq --m 2304 --k 2304 --n 512 --bcast 1 --J $J --skew 1.0; done
echo "== ggml path again for the same moment"
./moe_bench --type iq4_nl --m 2304 --k 2304 --n 512 --bcast 1 2>&1 | grep -v '^ggml_\|load_backend\|Device 0'
kill $SCLK
echo; echo "sclk samples:"; tr ' ' '\n' < logs/burst3.sclk | sort | uniq -c | sort -rn | head -5
