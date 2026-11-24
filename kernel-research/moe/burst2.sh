#!/bin/bash
# Burst 2: other weight types at the same shapes, and the token-count sweep (J tile scales with n/128).
cd "$(dirname "$(readlink -f "$0")")"
echo "== baseline again: 4 ops per graph vs 1"
./moe_bench --type iq4_nl --m 2304 --k 2304 --n 512 --bcast 1 --graph-reps 4 2>&1 | grep -v '^ggml_\|load_backend'
./moe_bench --type iq4_nl --m 2304 --k 2304 --n 512 --bcast 1 --graph-reps 1 2>&1 | grep -v '^ggml_\|load_backend'
./moe_bench --type iq4_nl --m 2304 --k 1152 --n 512 --bcast 0 --graph-reps 4 2>&1 | grep -v '^ggml_\|load_backend'
echo "== types, gate_up shape (m=2304 k=2304 n=512 bcast)"
for t in iq4_nl q4_0 q4_1 q4_K q5_0 q8_0 mxfp4 iq4_xs f16; do ./moe_bench --type $t --m 2304 --k 2304 --n 512 --bcast 1 --reps 15 2>&1 | grep -v '^ggml_\|load_backend'; done
echo "== types, down shape (m=2304 k=1152 n=512)"
for t in iq4_nl q4_0 q4_K q8_0 mxfp4 f16; do ./moe_bench --type $t --m 2304 --k 1152 --n 512 --bcast 0 --reps 15 2>&1 | grep -v '^ggml_\|load_backend'; done
echo "== n sweep iq4_nl gate_up"
for n in 64 128 256 512 1024 2048 4096; do ./moe_bench --type iq4_nl --m 2304 --k 2304 --n $n --bcast 1 --reps 10 2>&1 | grep -v '^ggml_\|load_backend'; done
echo "== n sweep iq4_nl down"
for n in 128 256 512 1024 2048 4096; do ./moe_bench --type iq4_nl --m 2304 --k 1152 --n $n --bcast 0 --reps 10 2>&1 | grep -v '^ggml_\|load_backend'; done
echo "== rocprofv3 (ROCm 10.1 wheel) kernel trace attempt"
source "${VENV:-$HOME/.venvs/rocm}/bin/activate"
rocprofv3 --kernel-trace --stats -d logs/prof -o gateup -- ./moe_bench --type iq4_nl --m 2304 --k 2304 --n 512 --bcast 1 --reps 5 --warm 1 2>&1 | tail -5
deactivate
