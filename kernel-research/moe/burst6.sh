#!/bin/bash
# Burst 6: J=16 correctness; hardware counters on the MMQ kernel (rocprofv3 ROCm 10.1 wheel, if the counters exist).
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1"
echo "== J=16 correctness"; ./moe_mmq $F --J 16 --check --reps 3 | grep check; ./moe_mmq --m 2304 --k 1152 --n 512 --bcast 0 --J 16 --check --reps 3 | grep check
source "${VENV:-$HOME/.venvs/rocm}/bin/activate"
rocprofv3-avail list > logs/pmc_avail.txt 2>&1
WANT="GRBM_GUI_ACTIVE SQ_WAVES SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_LDS SQ_INSTS_VMEM_RD SQ_INSTS_SALU SQ_INST_CYCLES_VMEM_RD SQ_WAIT_INST_LDS SQ_WAIT_INST_ANY SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_LDS_DATA_FIFO_FULL SQ_ACTIVE_INST_LDS SQ_ACTIVE_INST_VMEM SQ_ACTIVE_INST_VALU SQ_INSTS_WMMA SQ_ACTIVE_INST_ANY SQ_INST_LEVEL_LDS SQ_INST_LEVEL_VMEM TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum TCP_TOTAL_CACHE_ACCESSES_sum SQ_INSTS_SMEM"
HAVE=""; for c in $WANT; do grep -qw "$c" logs/pmc_avail.txt && HAVE="$HAVE $c"; done
echo "counters available of the wanted set:$HAVE"
# one or two counters per pass keeps every pass valid on RDNA (few SQ counters per pass)
for J in 32 16; do
  echo "== counters J=$J"
  i=0; set -- $HAVE
  while [ $# -gt 0 ]; do
    c1=$1; shift; c2=${1:-}; [ -n "$c2" ] && shift
    rocprofv3 --pmc $c1 $c2 --kernel-include-regex 'mul_mat_q' -d logs/pmc -o J${J}_$i --output-format csv -- ./moe_mmq $F --J $J --reps 2 --warm 1 > /dev/null 2> logs/pmc_J${J}_$i.err || { echo "pass $c1 $c2 failed: $(tail -1 logs/pmc_J${J}_$i.err)"; }
    i=$((i+1))
  done
done
deactivate
