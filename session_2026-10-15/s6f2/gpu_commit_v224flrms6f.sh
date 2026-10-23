#!/bin/bash
# gpu_commit_v224flrms6f.sh (10-15, Claude subagent): PREPARED, NOT RUN (the main session does every GPU commit). The one GPU
# commit of FLRM-AO + S6fall (eeshsaxena/rsna-sub-v224flrms6f, private, T4 x2, internet off, docker pin 2757e0c7) on the 3 public
# studies, then the gate. ~10-11 min of a T4x2 session (V224-FLIP-RM commit log end 432 s + S6F ~2.5 min MODEL). NEVER submits.
# Pattern of s6f/gpu_commit_v224flips6f.sh (rfix form: step 0 re-runs every offline test and accepts only exact verdict lines).
# Run from the project root (C:/Users/eeshs/Downloads/rsna-knee):  bash session_2026-10-15/s6f2/gpu_commit_v224flrms6f.sh
# A re-run pushes a NEW version of the same slug (first rename the _v1 out / log / gate paths below to _v2).
set -u
cd "$(dirname "$0")/../.." || exit 1
D=session_2026-10-15/s6f2
TAG=v224flrms6f
verdict_ok() { tail -1 | tr -d '\r' | grep -qx "$1 PASS \([0-9][0-9]*\)/\1"; }
# 0. the kernel on disk is the built one, the cell-diff proof and every offline test pass (re-run now, ~7 min laptop CPU)
py -3.12 -I - <<'EOF' || exit 1
import json, hashlib
r = json.load(open('session_2026-10-15/s6f2/build_receipt_v224flrms6f.json'))
h = hashlib.sha256(open('session_2026-10-15/s6f2/kernel_v224flrms6f/rsna_sub_v224flrms6f.ipynb', 'rb').read()).hexdigest()
assert h == r['notebook_sha256'], ('notebook differs from the build receipt', h)
m = json.load(open('session_2026-10-15/s6f2/kernel_v224flrms6f/kernel-metadata.json'))
assert m['id'] == 'eeshsaxena/rsna-sub-v224flrms6f' and m['is_private'] is True and m['enable_internet'] is False, m
assert r['base_kind'] == 'flrm' and r['policy']['g_start_h'] == 6.0 and 'control' not in r['policy'], r['policy']
print('kernel sha OK', h[:16])
EOF
py -3.12 -I $D/celldiff_s6f2_proof.py --kernel flrm | verdict_ok S6F2_CELLDIFF || { echo "cell-diff proof FAILED"; exit 1; }
py -3.12 -I $D/run_offline_tests_s6f2.py | tee $D/offline_tests_before_push_$TAG.txt | verdict_ok OFFLINE_TESTS_S6F2 || {
  echo "offline tests not all PASS (see $D/offline_tests_before_push_$TAG.txt)"; exit 1; }
# 1. at most 1 GPU session running (2 concurrent max)
python session_2026-10-15/v224f/running.py 6 | tee $D/running_before_push_$TAG.txt
N=$(grep RUNNING_COUNT $D/running_before_push_$TAG.txt | awk '{print $2}')
if [ "${N:-9}" -gt 1 ]; then echo "RUNNING_COUNT $N > 1: not pushing"; exit 1; fi
# 2. push (a commit run on the 3 public studies; never a competition submission)
kaggle kernels push -p $D/kernel_$TAG 2>&1 | tee $D/push_$TAG.log
grep -q "successfully pushed" $D/push_$TAG.log || { echo "PUSH FAILED"; exit 1; }
# 3. wait (<= 45 min) and fetch into a NEW dir
python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$TAG $D/out_${TAG}_v1 45 | tee $D/out_${TAG}_v1.log
# 4. gate: (0) notebook, (A) base rows = the UNCHANGED V224-FLIP-RM gate on the frozen output + stage csvs vs the FLRM v1 commit,
#    (B) parity vs nartaa, (C) blend + base canaries, (D) coverage, (E) receipts / env / policy, (F) canary, (G) markers, (H) format
py -3.12 -I $D/v224s6f2_check.py $D/out_${TAG}_v1 --build $D/build_receipt_$TAG.json | tee $D/gate_${TAG}_v1.txt
tail -1 $D/gate_${TAG}_v1.txt
