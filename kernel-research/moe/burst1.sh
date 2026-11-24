#!/bin/bash
# Burst 1: ceilings (WMMA iu8/f16/dp4a peak, DRAM stream) and the baseline IQ4_NL MUL_MAT_ID at Inkling's shapes.
cd "$(dirname "$(readlink -f "$0")")"
echo "== wmma peak"; ./wmma_iu8_peak
echo "== stream"; ../bw/stream
echo "== baseline iq4_nl, uniform routing"
./moe_bench --type iq4_nl --m 2304 --k 2304 --n 512 --bcast 1 --check
./moe_bench --type iq4_nl --m 2304 --k 1152 --n 512 --bcast 0 --check
echo "== skewed routing"
for s in 0.5 1.0; do
./moe_bench --type iq4_nl --m 2304 --k 2304 --n 512 --bcast 1 --skew $s
./moe_bench --type iq4_nl --m 2304 --k 1152 --n 512 --bcast 0 --skew $s
done
