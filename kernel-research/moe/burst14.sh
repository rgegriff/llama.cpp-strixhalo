#!/bin/bash
# Burst 14: hardware counters on the WMMA kernel (v1 and HALF_DEQ), one counter group per pass, gate_up shape.
cd "$(dirname "$(readlink -f "$0")")"
F="--m 2304 --k 2304 --n 512 --bcast 1 --reps 2 --warm 1"
source "${VENV:-$HOME/.venvs/rocm}/bin/activate"
rocprofv3 -L 2>/dev/null | grep -oE "^(SQ|GRBM|TCC|TCP|TA|TD)_[A-Z0-9_a-z]+" | sort -u > logs/pmc_list.txt; echo "counters listed: $(wc -l < logs/pmc_list.txt)"
for v in moe_wmma moe_wmma_MOE_HALF_DEQ; do
  i=0
  for grp in "GRBM_GUI_ACTIVE SQ_WAVES" "SQ_BUSY_CYCLES SQ_INSTS_LDS" "SQ_WAIT_INST_LDS SQ_WAIT_INST_ANY" "SQ_INSTS_VALU" "SQ_INSTS_SALU SQ_WAVE_CYCLES" "SQ_ACTIVE_INST_VALU" "SQ_ACTIVE_INST_LDS" "SQ_INST_CYCLES_VMEM_RD" "SQ_LDS_BANK_CONFLICT" "SQ_INSTS_WMMA" "SQ_VALU_MFMA_BUSY_CYCLES" "SQ_INSTS_VALU_MFMA_MOPS" "SQ_ACTIVE_INST_ANY" "SQ_INSTS_VMEM_RD" "TCC_HIT_sum TCC_MISS_sum" "SQ_INST_LEVEL_VMEM" "SQ_LDS_IDX_ACTIVE" "SQ_LDS_DATA_FIFO_FULL" "SQ_INST_CYCLES_LDS" "SQ_ACTIVE_INST_VMEM"; do
    rocprofv3 --pmc $grp --kernel-include-regex 'mul_mat_id_iq4nl' -d logs/pmc -o ${v}_$i --output-format csv -- ./$v --kernel wmma $F > /dev/null 2> logs/pmc_${v}_$i.err || echo "$v pass '$grp' failed"
    f=logs/pmc/${v}_${i}_counter_collection.csv
    if [ -f $f ]; then python3 - "$f" "$grp" <<'PY'
import csv,sys
rows=[r for r in csv.DictReader(open(sys.argv[1])) if 'mul_mat_id' in r.get('Kernel_Name','')]
n=len(set(r['Dispatch_Id'] for r in rows)); agg={}
for r in rows: agg[r['Counter_Name']]=agg.get(r['Counter_Name'],0)+float(r['Counter_Value'])
print(sys.argv[1].split('/')[-1][:-23], 'x%d'%n, ' '.join('%s=%.4g'%(k,v/max(n,1)) for k,v in agg.items()))
PY
    else echo "$v '$grp': no csv"; fi
    i=$((i+1))
  done
done
deactivate
