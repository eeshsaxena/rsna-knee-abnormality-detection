#!/bin/bash
# gpu_commit_v224flips6f.sh (10-15, Claude subagent): PREPARED, NOT RUN (the main session does every GPU commit). The one GPU
# commit of V224-FLIP-AO + S6fall (eeshsaxena/rsna-sub-v224flips6f, private, T4 x2, internet off, docker pin 2757e0c7) on the 3
# public studies, then the gate. ~10-11 min of a T4x2 session (V224-FLIP commit log end 481 s + S6F ~2.5 min MODEL). NEVER submits.
# Run from the project root (C:/Users/eeshs/Downloads/rsna-knee):  bash session_2026-10-15/s6f/gpu_commit_v224flips6f.sh
# [rfix 10-15] main ran this once (03:30-03:45 IST 10-15): v1 pushed, gate PASS 31/31 (gate_v224flips6f_v1.txt). A new run
# pushes a NEW version of the same slug (rebuilds only; first rename the _v1 out / log / gate paths below to _v2). Step 0 now
# re-runs every offline test (finding S6F-R4).
set -u
cd "$(dirname "$0")/../.." || exit 1
D=session_2026-10-15/s6f
# 0. the kernel on disk is the built one, the cell-diff proof and every offline test pass
#    [rfix S6F-R4] every offline test is RE-RUN on the current files (run_offline_tests.py, ~5 min laptop CPU) and only an exact
#    verdict line counts (last line '<NAME> PASS n/n'); recorded tests/*.txt are never grepped (" PASS " matched non-verdict lines)
verdict_ok() { tail -1 | tr -d '\r' | grep -qx "$1 PASS \([0-9][0-9]*\)/\1"; }
py -3.12 -I - <<'EOF' || exit 1
import json, hashlib
r = json.load(open('session_2026-10-15/s6f/build_receipt_v224flips6f.json'))
h = hashlib.sha256(open('session_2026-10-15/s6f/kernel_v224flips6f/rsna_sub_v224flips6f.ipynb', 'rb').read()).hexdigest()
assert h == r['notebook_sha256'], ('notebook differs from the build receipt', h)
print('kernel sha OK', h[:16])
EOF
py -3.12 -I $D/celldiff_v224flips6f_proof.py | verdict_ok S6F_CELLDIFF || { echo "cell-diff proof FAILED"; exit 1; }
py -3.12 -I $D/run_offline_tests.py | tee $D/offline_tests_before_push.txt | verdict_ok OFFLINE_TESTS || {
  echo "offline tests not all PASS (see $D/offline_tests_before_push.txt)"; exit 1; }
# 1. at most 1 GPU session running (2 concurrent max)
python session_2026-10-15/v224f/running.py 6 | tee $D/running_before_push.txt
N=$(grep RUNNING_COUNT $D/running_before_push.txt | awk '{print $2}')
if [ "${N:-9}" -gt 1 ]; then echo "RUNNING_COUNT $N > 1: not pushing"; exit 1; fi
# 2. push
kaggle kernels push -p $D/kernel_v224flips6f 2>&1 | tee $D/push_v224flips6f.log
grep -q "successfully pushed" $D/push_v224flips6f.log || { echo "PUSH FAILED"; exit 1; }
# 3. wait (<= 45 min) and fetch into a NEW dir
python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-v224flips6f $D/out_v224flips6f_v1 45 | tee $D/out_v224flips6f_v1.log
# 4. gate (A base rows, B parity vs nartaa, C blend, D coverage, E receipts / env / policy, F canary, G markers, H format)
py -3.12 -I $D/v224s6f_check.py $D/out_v224flips6f_v1 | tee $D/gate_v224flips6f_v1.txt
tail -1 $D/gate_v224flips6f_v1.txt
