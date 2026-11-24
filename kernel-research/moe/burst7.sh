#!/bin/bash
# Burst 7: hardware counters on the MMQ kernel, J=32 vs J=16, one counter group per pass (rocprofv3-avail is broken in the wheel, so try names).
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1"
source "${VENV:-$HOME/.venvs/rocm}/bin/activate"
for J in 32 16; do
  i=0
  for grp in "GRBM_GUI_ACTIVE SQ_WAVES" "SQ_BUSY_CYCLES SQ_INSTS_LDS" "SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE" "SQ_WAIT_INST_LDS SQ_WAIT_INST_ANY" "SQ_INSTS_VMEM_RD SQ_INSTS_VALU" "SQ_ACTIVE_INST_LDS SQ_ACTIVE_INST_VMEM" "SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_ANY" "SQ_INST_LEVEL_VMEM SQ_INST_LEVEL_LDS" "SQ_INSTS_WMMA" "TCC_HIT_sum TCC_MISS_sum" "SQ_WAVE_CYCLES SQ_INSTS_SALU" "SQ_INST_CYCLES_VMEM_RD SQ_INSTS_SMEM"; do
    rocprofv3 --pmc $grp --kernel-include-regex 'mul_mat_q' -d logs/pmc -o J${J}_$i --output-format csv -- ./moe_mmq $F --J $J --reps 2 --warm 1 > /dev/null 2> logs/pmc_J${J}_$i.err || echo "J=$J pass '$grp' failed: $(grep -m1 -i 'error\|invalid\|not found' logs/pmc_J${J}_$i.err | cut -c1-160)"
    i=$((i+1))
  done
done
deactivate
ls logs/pmc | head -40
