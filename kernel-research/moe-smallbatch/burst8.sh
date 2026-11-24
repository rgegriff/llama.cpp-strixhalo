#!/bin/bash
# Burst 8: in-place test cases (debug log shows the aliasing each launch saw), then KL at ub 8 against the unfused base
# with only CONV_STATE (mask 1) and only CONV_DYN (mask 2) on the multi-token kernel.
cd "$(dirname "$(readlink -f "$0")")"
K=../../build-rocm-moesb/bin; M=${MODEL:?path to Inkling-2-Mini-IQ4_NL.gguf}
TXT=${TXT:?a text file of a few thousand tokens}
Q=${QUIET:-}   # optional wrapper that serializes GPU runs
echo "== burst8 start $(date +%T)"
for ops in INKLING2_CONV_STATE INKLING2_CONV_DYN; do
  GGML_CUDA_INKLING2_CONV_MT_DEBUG=1 $Q $K/test-backend-ops test -b ROCm0 -o $ops > logs/b8-test-$ops.log 2>&1
  grep -E "tests passed|FAIL" logs/b8-test-$ops.log | head -5; grep "inkling2_conv_mt:" logs/b8-test-$ops.log | sort | uniq -c
done
export GGML_CUDA_INKLING2_FUSION=1 GGML_CUDA_FATTN_VEC_GQA=0 GGML_CUDA_MMID_Q8_MMQ_MIN=0
PPL="$K/llama-perplexity -m $M -f $TXT -ngl 99 -fa on -c 2048 --chunks 1 -ctk q8_0 -ctv q8_0"
KLD=${TMPDIR:-/tmp}/tmp-moesb-b8.kld
GGML_CUDA_INKLING2_CONV_MT=0 $Q $PPL -b 8 -ub 8 --kl-divergence-base $KLD > logs/b8-ref.log 2>&1 || echo FAIL ref
for mask in 1 2 3; do
  GGML_CUDA_INKLING2_CONV_MT=$mask GGML_CUDA_INKLING2_CONV_MT_DEBUG=1 $Q $PPL -b 8 -ub 8 --kl-divergence-base $KLD --kl-divergence > logs/b8-kl-mask$mask.log 2>&1 || echo FAIL
  echo "mask $mask: $(grep -E 'Mean    KLD' logs/b8-kl-mask$mask.log) | $(grep -E 'Maximum KLD' logs/b8-kl-mask$mask.log) | $(grep -E '^Same top' logs/b8-kl-mask$mask.log)"
  grep "inkling2_conv_mt:" logs/b8-kl-mask$mask.log | sort | uniq -c
done
rm -f $KLD
echo "== burst8 end $(date +%T)"
