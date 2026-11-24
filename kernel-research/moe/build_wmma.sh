#!/bin/bash
# Build the moe_wmma harness: the dedicated IQ4_NL MUL_MAT_ID WMMA kernel next to the MMQ path. Does not touch the GPU.
# VARIANT=MOE_WMMA_SWAP_AB etc. builds moe_wmma_<VARIANT>; extra -D flags via DEFS.
set -e
cd "$(dirname "$(readlink -f "$0")")"
ROOT=../..
G=$ROOT/ggml/src/ggml-cuda
OUT=moe_wmma${VARIANT:+_$VARIANT}${SUFFIX}
/usr/lib/llvm-21/bin/clang++ -x hip --offload-arch=gfx1151 -O3 -DNDEBUG -std=gnu++17 \
  -DGGML_USE_HIP -DGGML_HIP_NO_VMM -DGGML_HIP_GRAPHS -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -D_GNU_SOURCE \
  -I$G -I$ROOT/ggml/include -I$ROOT/ggml/src -I${ROCWMMA_DIR:-$HOME/src/rocWMMA}/library/include \
  -Wno-unused-result -Wno-unused-value -Rpass-analysis=kernel-resource-usage ${VARIANT:+-D$VARIANT} $DEFS \
  moe_wmma.hip -o $OUT 2> logs/build_$OUT.log || { grep -m5 -B2 -A6 'error' logs/build_$OUT.log; exit 1; }
grep -A12 "mul_mat_id_iq4nl" logs/build_$OUT.log | grep -E "Function Name|VGPRs:|SGPRs|Occupancy|ScratchSize|LDS" | head -8
ls -la $OUT
