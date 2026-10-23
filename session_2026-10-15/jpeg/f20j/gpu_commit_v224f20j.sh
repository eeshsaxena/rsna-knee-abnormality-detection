#!/bin/bash
# gpu_commit_v224f20j.sh (2026-10-14, Claude subagent): PREPARED, NOT RUN. The one GPU commit that proves V224-F20-J end to end
# on the 3 public studies (all uncompressed): the DEC cell works on the GPU image (same docker pin) and every output is the
# V224-F20 v1 commit's output. ~11 min of a T4x2 session (V224-F20 v1 commit: 586 s notebook time). NEVER submits.
# Run from the project root (C:/Users/eeshs/Downloads/rsna-knee) after 19:30 IST, when a GPU slot is free:
#   bash session_2026-10-15/jpeg/f20j/gpu_commit_v224f20j.sh
set -u
cd "$(dirname "$0")/../../.." || exit 1
D=session_2026-10-15/jpeg/f20j
DT=$(date +%Y%m%d%H%M)   # laptop clock = IST
if [ "$DT" -lt 202610141930 ]; then echo "too early ($DT IST): the GPU slots are reserved until 19:30 IST 10-14"; exit 1; fi
# 0. the kernel on disk is the built one (sha in the build receipt) and the CPU proof passed
python - <<'EOF' || exit 1
import json, hashlib, sys
r = json.load(open('session_2026-10-15/jpeg/f20j/build_receipt_v224f20j.json'))
h = hashlib.sha256(open('session_2026-10-15/jpeg/f20j/kernel_v224f20j/rsna_sub_v224f20j.ipynb', 'rb').read()).hexdigest()
assert h == r['notebook_sha256'] == '2a480ce894ee9f2f5c2f10da57343c052ff1fd1f60039ccd57cb99953790dcb5', h
v = open('session_2026-10-15/jpeg/f20j/out_jj_v1/v224f20j_cpu_table.md').read()
assert 'V224F20J_CPU PASS' in v, 'CPU check did not pass'
print('kernel sha OK, CPU check PASS')
EOF
# 1. at most 1 GPU session running (2 concurrent max)
python session_2026-10-15/v224f/running.py 6 | tee $D/running_before_push.txt
N=$(grep RUNNING_COUNT $D/running_before_push.txt | awk '{print $2}')
if [ "${N:-9}" -gt 1 ]; then echo "RUNNING_COUNT $N > 1: not pushing"; exit 1; fi
# 2. push (private, GPU T4, internet off, same docker pin as V224-F20; metadata in kernel_v224f20j/kernel-metadata.json)
kaggle kernels push -p $D/kernel_v224f20j 2>&1 | tee $D/push_v224f20j.log
grep -q "successfully pushed" $D/push_v224f20j.log || { echo "PUSH FAILED"; exit 1; }
# 3. wait (<= 40 min) and fetch
python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-v224f20j $D/out_v224f20j_v1 40 | tee $D/out_v224f20j_v1.log
# 4. gate (V224-F20 gate 28/28 + DEC checks + identity with the V224-F20 v1 commit)
python $D/v224f20j_check.py $D/out_v224f20j_v1 | tee $D/gate_v224f20j_v1.txt
tail -1 $D/gate_v224f20j_v1.txt
