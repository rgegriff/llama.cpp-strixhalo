#!/bin/bash
# Burst 7: which multi-token conv path agrees with one-token decode? Reference logits from -b 1 -ub 1 (base config),
# then KL of ub 8 and ub 2 with the conv fusion off / on against it.
cd "$(dirname "$(readlink -f "$0")")"
K=../../build-rocm-moesb/bin; M=${MODEL:?path to Inkling-2-Mini-IQ4_NL.gguf}
TXT=${TXT:?a text file of a few thousand tokens}
Q=${QUIET:-}   # optional wrapper that serializes GPU runs
export GGML_CUDA_INKLING2_FUSION=1 GGML_CUDA_FATTN_VEC_GQA=0 GGML_CUDA_MMID_Q8_MMQ_MIN=0
PPL="$K/llama-perplexity -m $M -f $TXT -ngl 99 -fa on -c 2048 --chunks 1 -ctk q8_0 -ctv q8_0"
KLD=${TMPDIR:-/tmp}/tmp-moesb-b7.kld
echo "== burst7 start $(date +%T)"
GGML_CUDA_INKLING2_CONV_MT=0 $Q $PPL -b 1 -ub 1 --kl-divergence-base $KLD > logs/b7-ref-ub1.log 2>&1 || echo FAIL ref
for ub in 8 2 ${EXTRA_UBS}; do for mt in 0 1; do
  GGML_CUDA_INKLING2_CONV_MT=$mt $Q $PPL -b $ub -ub $ub --kl-divergence-base $KLD --kl-divergence > logs/b7-ub$ub-mt$mt.log 2>&1 || echo FAIL
  echo "ub $ub conv_mt=$mt vs ub1: $(grep -E 'Mean    KLD' logs/b7-ub$ub-mt$mt.log) | $(grep -E 'Maximum KLD' logs/b7-ub$ub-mt$mt.log) | $(grep -E '^Same top' logs/b7-ub$ub-mt$mt.log)"
done; done
rm -f $KLD
echo "== burst7 end $(date +%T)"
