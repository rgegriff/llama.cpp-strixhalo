#!/bin/bash
# Build the MoE expert-matmul benches. Needs build-rocm-mix (libggml*.so). Does not touch the GPU.
set -e
cd "$(dirname "$(readlink -f "$0")")"
ROOT=../..
B=$ROOT/build-rocm-mix/bin
g++ -O2 -std=c++17 -I$ROOT/ggml/include moe_bench.cpp -o moe_bench -L$B -lggml -lggml-base -Wl,-rpath,"$(readlink -f $B)"
hipcc --offload-arch=gfx1151 -O3 -Wno-unused-value -Wno-unused-result wmma_iu8_peak.hip -o wmma_iu8_peak
ls -la moe_bench wmma_iu8_peak
# standalone MMQ harness: kernels from the copies in mmq/, the rest of the headers from ggml/src/ggml-cuda (read-only)
G=$ROOT/ggml/src/ggml-cuda
/usr/lib/llvm-21/bin/clang++ -x hip --offload-arch=gfx1151 -O3 -DNDEBUG -std=gnu++17 \
  -DGGML_USE_HIP -DGGML_HIP_NO_VMM -DGGML_HIP_GRAPHS -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -D_GNU_SOURCE \
  -I$G -I$ROOT/ggml/include -I$ROOT/ggml/src -I${ROCWMMA_DIR:-$HOME/src/rocWMMA}/library/include \
  -Wno-unused-result -Wno-unused-value -Rpass-analysis=kernel-resource-usage ${VARIANT:+-D$VARIANT} \
  moe_mmq.hip -o moe_mmq${VARIANT:+_$VARIANT} 2> logs/build_moe_mmq${VARIANT:+_$VARIANT}.log || { grep -m5 -B2 -A6 'error' logs/build_moe_mmq${VARIANT:+_$VARIANT}.log; exit 1; }
ls -la moe_mmq${VARIANT:+_$VARIANT}
