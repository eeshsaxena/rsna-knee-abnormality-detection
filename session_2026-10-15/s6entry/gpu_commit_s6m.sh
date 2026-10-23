#!/bin/bash
# gpu_commit_s6m.sh (10-15, Claude): one GPU commit of an S6 entry kernel built by patch_s6_model.py, fetch, gate. NEVER submits.
# Run from the project root: bash session_2026-10-15/s6entry/gpu_commit_s6m.sh TAG BASE_OUT [--control]
set -u
cd "$(dirname "$0")/../.." || exit 1
TAG=$1; BASE_OUT=$2; CTL=${3:-}
D=session_2026-10-15/s6entry
OUT=$D/out_${TAG}_v1
python session_2026-10-15/v224f/running.py 6 | tee $D/running_before_push_$TAG.txt
N=$(grep RUNNING_COUNT $D/running_before_push_$TAG.txt | awk '{print $2}')
if [ "${N:-9}" -gt 1 ]; then echo "RUNNING_COUNT $N > 1: not pushing"; exit 1; fi
kaggle kernels push -p $D/kernel_$TAG 2>&1 | tee $D/push_$TAG.log
grep -q "successfully pushed" $D/push_$TAG.log || { echo "PUSH FAILED"; exit 1; }
python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$TAG $OUT 50 | tee $OUT.log
py -3.12 -I $D/s6m_check.py $OUT --tag $TAG --base-out $BASE_OUT $CTL | tee $D/gate_${TAG}_v1.txt
tail -1 $D/gate_${TAG}_v1.txt
