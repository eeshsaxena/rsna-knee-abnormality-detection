#!/bin/bash
# gpu_commit_hw.sh (10-16, Claude): one GPU commit of rsna-sub-v224r6hws6fj, fetch, gate. NEVER submits. Run from the project root.
set -u
cd "$(dirname "$0")/../.." || exit 1
D=session_2026-10-16/hw; TAG=v224r6hws6fj; OUT=$D/out_${TAG}_v1
python session_2026-10-15/v224f/running.py 6 | tee $D/running_before_push.txt
N=$(grep RUNNING_COUNT $D/running_before_push.txt | awk '{print $2}')
if [ "${N:-9}" -gt 1 ]; then echo "RUNNING_COUNT $N > 1: not pushing"; exit 1; fi
kaggle kernels push -p $D/kernel_$TAG 2>&1 | tee $D/push_$TAG.log
grep -q "successfully pushed" $D/push_$TAG.log || { echo "PUSH FAILED"; exit 1; }
python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$TAG $OUT 50 | tee $OUT.log
py -3.12 -I $D/hw_check.py $OUT | tee $D/gate_${TAG}_v1.txt
tail -1 $D/gate_${TAG}_v1.txt
