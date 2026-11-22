#!/bin/bash
# Build the inkling-kern worktree. Same recipe as ~/src/llama.cpp-inkling-fa/build-all.sh, but -j 8 (house rule:
# the host CPU is throttled under GPU load) and with the test binaries. Building does not touch the GPU.
#   usage: kernel-research/build-kern.sh [cpu] [rocm] [vulkan]      (default: all three)
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
J=${J:-8}
T="--target llama-server llama-bench llama-perplexity test-flash-attn-indexed-bias test-backend-ops"
ARGS=("$@")
want() { [ ${#ARGS[@]} -eq 0 ] && return 0; for a in "${ARGS[@]}"; do [ "$a" = "$1" ] && return 0; done; return 1; }
if want cpu; then
  echo "== cpu configure $(date +%T)"
  cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON > build-cpu.cfg.log 2>&1 || { echo "cpu configure FAILED"; tail -20 build-cpu.cfg.log; }
  nice -n 10 cmake --build build-cpu -j "$J" $T > build-cpu.log 2>&1 && echo "cpu OK $(date +%T)" || { echo "cpu build FAILED"; grep -m8 -E "error|fatal" build-cpu.log; }
fi
if want rocm; then
  echo "== rocm configure $(date +%T)"
  cmake -B build-rocm -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGPU_TARGETS=gfx1151 \
    -DCMAKE_HIP_COMPILER=/usr/lib/llvm-21/bin/clang++ -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_HIP_NO_VMM=ON \
    -DCMAKE_HIP_FLAGS="-I${ROCWMMA_DIR:-$HOME/src/rocWMMA}/library/include" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON > build-rocm.cfg.log 2>&1 || { echo "rocm configure FAILED"; tail -25 build-rocm.cfg.log; }
  nice -n 10 cmake --build build-rocm -j "$J" $T > build-rocm.log 2>&1 && echo "rocm OK $(date +%T)" || { echo "rocm build FAILED"; grep -m8 -i -E "error|fatal" build-rocm.log; }
fi
if want vulkan; then
  echo "== vulkan configure $(date +%T)"
  cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON > build-vulkan.cfg.log 2>&1 || { echo "vulkan configure FAILED"; tail -20 build-vulkan.cfg.log; }
  nice -n 10 cmake --build build-vulkan -j "$J" $T > build-vulkan.log 2>&1 && echo "vulkan OK $(date +%T)" || { echo "vulkan build FAILED"; grep -m5 -i error build-vulkan.log; }
fi
ls -la build-*/bin/llama-bench 2>&1
