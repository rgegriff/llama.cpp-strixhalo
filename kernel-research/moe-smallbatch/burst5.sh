#!/bin/bash
# Burst 5: in-model A/B, unsynchronized. llama-perplexity -c 2048 --chunks 1 -b N -ub N; ms per ubatch from
# llama_perf "prompt eval time". base = GGML_CUDA_INKLING2_CONV_MT=0 GGML_CUDA_MMID_Q8_MMQ_MIN=0, new = defaults.
# Two interleaved rounds. Then OP_STATS=1 (counts, no sync) at ub 8 for both, and KL of new vs base at ub 8.
# EXTRA_BASE / EXTRA_NEW: further env assignments for the two sides; TAG names the logs.
cd "$(dirname "$(readlink -f "$0")")"
K=../../build-rocm-moesb/bin; M=${MODEL:?path to Inkling-2-Mini-IQ4_NL.gguf}
TXT=${TXT:?a text file of a few thousand tokens}
Q=${QUIET:-}   # optional wrapper that serializes GPU runs; TAG=${TAG:-b5}
export GGML_CUDA_INKLING2_FUSION=1 GGML_CUDA_FATTN_VEC_GQA=0
BASE="GGML_CUDA_INKLING2_CONV_MT=0 GGML_CUDA_MMID_Q8_MMQ_MIN=0 $EXTRA_BASE"; NEW="$EXTRA_NEW"
PPL="$K/llama-perplexity -m $M -f $TXT -ngl 99 -fa on -c 2048 --chunks 1 -ctk q8_0 -ctv q8_0"
echo "== $TAG start $(date +%T)  base: $BASE | new: $NEW"
for round in 1 2; do
  for ub in ${UBS:-1 2 4 8}; do
    for side in base new; do
      [ $side = base ] && E="$BASE" || E="$NEW"
      L=logs/$TAG-$side-ub$ub-r$round.log
      env $E $Q $PPL -b $ub -ub $ub > $L 2>&1 || echo FAIL $L
      awk -v ub=$ub -v s=$side -v r=$round '/prompt eval time/ { for (i=1;i<=NF;i++) if ($i=="ms" && $(i+1)=="/") { printf "%s ub %d round %d: %.2f ms/token, %.2f ms/ubatch\n", s, ub, r, $(i-1)/$(i+2), $(i-1)/$(i+2)*ub; exit } }' $L
    done
  done
done
echo "== op counts at ub 8"
for side in base new; do
  [ $side = base ] && E="$BASE" || E="$NEW"
  env $E GGML_CUDA_OP_STATS=1 $Q $PPL -b 8 -ub 8 > logs/$TAG-$side-ub8-counts.log 2>&1
  grep -E "graph evaluations" logs/$TAG-$side-ub8-counts.log | sed "s/^/$side: /"
done
echo "== KL new vs base, ub 8"
KLD=${TMPDIR:-/tmp}/tmp-moesb-$TAG.kld
env $BASE $Q $PPL -b 8 -ub 8 --kl-divergence-base $KLD > logs/$TAG-kl-base.log 2>&1 || echo FAIL kl base
env $NEW $Q $PPL -b 8 -ub 8 --kl-divergence-base $KLD --kl-divergence > logs/$TAG-kl-new.log 2>&1 || echo FAIL kl new
grep -E "Mean    KLD|Maximum KLD|Same top|Mean PPL\(Q\)|Mean PPL\(base\)" logs/$TAG-kl-new.log
rm -f $KLD
echo "== $TAG end $(date +%T)"
