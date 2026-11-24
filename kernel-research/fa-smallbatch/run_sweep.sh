#!/bin/bash
# Timing sweep: {q8_0,q4_0} x {32768,131072,262144} x n_q {1,2,4,8,16}; one quiet.sh burst per (type, n_kv).
cd "$(dirname "$(readlink -f "$0")")"
out=results/sweep-$(date +%Y%m%d-%H%M).log
for t in q8_0 q4_0; do
  for n in 32768 131072 262144; do
    ch="512 1024"; [ $n = 32768 ] && ch="256 512 1024"
    ${QUIET:-env} bash -c "for nq in 1 2 4 8 16; do REPS=20 ./fa_smallbatch $t $n \$nq $ch 2>&1 | grep -v Warning; done" >> "$out" 2>&1
  done
done
echo DONE >> "$out"
