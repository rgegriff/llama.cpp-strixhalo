#!/bin/bash
# Build the small-batch MoE benches against build-rocm-moesb. No GPU use.
set -e
cd "$(dirname "$(readlink -f "$0")")"
ROOT=../..
B=$ROOT/build-rocm-moesb/bin
g++ -O2 -std=c++17 -I$ROOT/ggml/include ../moe/moe_bench.cpp -o moe_bench_sb -L$B -lggml -lggml-base -Wl,-rpath,"$(readlink -f $B)"
[ -x stream ] || hipcc --offload-arch=gfx1151 -O3 ../bw/stream.hip -o stream
ls -la moe_bench_sb stream
