#!/bin/bash
# Burst 6: three more interleaved timing rounds at ub 2 4 8 (seconds per pass / 2048 tokens * ub), then KL at ub 8
# against one base run: base again (noise floor), conv fusion only, Q8_0 MMQ only, both.
cd "$(dirname "$(readlink -f "$0")")"
K=../../build-rocm-moesb/bin; M=${MODEL:?path to Inkling-2-Mini-IQ4_NL.gguf}
TXT=${TXT:?a text file of a few thousand tokens}
Q=${QUIET:-}   # optional wrapper that serializes GPU runs
export GGML_CUDA_INKLING2_FUSION=1 GGML_CUDA_FATTN_VEC_GQA=0
PPL="$K/llama-perplexity -m $M -f $TXT -ngl 99 -fa on -c 2048 --chunks 1 -ctk q8_0 -ctv q8_0"
cfg() { case $1 in base) echo "GGML_CUDA_INKLING2_CONV_MT=0 GGML_CUDA_MMID_Q8_MMQ_MIN=0";; conv) echo "GGML_CUDA_INKLING2_CONV_MT=1 GGML_CUDA_MMID_Q8_MMQ_MIN=0";;
  q8) echo "GGML_CUDA_INKLING2_CONV_MT=0 GGML_CUDA_MMID_Q8_MMQ_MIN=4";; new) echo "GGML_CUDA_INKLING2_CONV_MT=1 GGML_CUDA_MMID_Q8_MMQ_MIN=4";; esac; }
pass() { grep -h 'seconds per pass' $1 | sed -E 's/.*perplexity: ([0-9.]+) seconds.*/\1/' | awk -v ub=$2 '{printf "%.2f ms/ubatch", $1/2048*ub*1000}'; }
echo "== burst6 start $(date +%T)"
for round in 3 4 5; do
  for ub in 2 4 8; do
    for side in new base; do
      L=logs/b6-$side-ub$ub-r$round.log
      env $(cfg $side) $Q $PPL -b $ub -ub $ub > $L 2>&1 || echo FAIL $L
      echo "$side ub $ub round $round: $(pass $L $ub)  $(date +%T)"
    done
  done
done
echo "== KL at ub 8 vs one base run"
KLD=${TMPDIR:-/tmp}/tmp-moesb-b6.kld
env $(cfg base) $Q $PPL -b 8 -ub 8 --kl-divergence-base $KLD > logs/b6-kl-ref.log 2>&1 || echo FAIL kl ref
for side in base conv q8 new; do
  env $(cfg $side) $Q $PPL -b 8 -ub 8 --kl-divergence-base $KLD --kl-divergence > logs/b6-kl-$side.log 2>&1 || echo FAIL kl $side
  echo "$side: $(grep -E 'Mean    KLD' logs/b6-kl-$side.log) | $(grep -E 'Maximum KLD' logs/b6-kl-$side.log) | $(grep -E 'Same top' logs/b6-kl-$side.log)"
done
rm -f $KLD
echo "== burst6 end $(date +%T)"
