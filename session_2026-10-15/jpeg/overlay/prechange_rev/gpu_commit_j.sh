#!/bin/bash
# gpu_commit_j.sh TAG (2026-10-15, Claude subagent): PREPARED, NOT RUN (the main session does every GPU commit). The one GPU commit of a
# J kernel built by make_j.py (private, T4 x2, internet off, docker pin 2757e0c7 = its base's) on the 3 public studies, then the gate
# against the BASE commit output. NEVER submits. TAG in: v224flipj | v224flips6fj | v224flips20j.
# Cost (MODEL): the base commit's time + ~6 s DEC (base log ends: FLIP 481 s, S6F 591 s, FLIPS20 527 s).
# Run from the project root (C:/Users/eeshs/Downloads/rsna-knee):  bash session_2026-10-15/jpeg/overlay/gpu_commit_j.sh v224flipj
# A re-run pushes a NEW version of the same slug: rename the _v1 out / log / gate paths first.
set -u
cd "$(dirname "$0")/../../.." || exit 1
D=session_2026-10-15/jpeg/overlay
T=${1:?usage: gpu_commit_j.sh TAG}
case "$T" in
  v224flipj)    BASE_OUT=session_2026-10-14/v224/out_v224flip_v1 ;;
  v224flips6fj) BASE_OUT=session_2026-10-15/s6f/out_v224flips6f_v1 ;;
  v224flips20j) BASE_OUT=session_2026-10-15/v224f/out_v224flips20_v1 ;;
  *) echo "unknown tag $T"; exit 1 ;;
esac
verdict_ok() { tail -1 | tr -d '\r' | grep -q "^$1 PASS \([0-9][0-9]*\)/\1\b"; }
# 0. the kernel on disk is the built one; every offline J check re-run now on the current files (exact verdict lines only)
py -3.12 -I - "$T" <<'EOF' || exit 1
import json, hashlib, sys
t = sys.argv[1]
r = json.load(open('session_2026-10-15/jpeg/overlay/j_build_receipt_%s.json' % t))
h = hashlib.sha256(open('session_2026-10-15/jpeg/overlay/kernel_%s/rsna_sub_%s.ipynb' % (t, t), 'rb').read()).hexdigest()
assert h == r['notebook_sha256'], ('notebook differs from the build receipt', h)
m = json.load(open('session_2026-10-15/jpeg/overlay/kernel_%s/kernel-metadata.json' % t))
assert m['id'] == 'eeshsaxena/rsna-sub-' + t and m['is_private'] is True and m['enable_internet'] is False and 'id_no' not in m, m
print('kernel sha OK', h[:16], m['id'])
EOF
py -3.12 -I $D/run_offline_j.py "$T" | tee $D/offline_before_push_$T.txt | verdict_ok J_OFFLINE || { echo "offline J checks not all PASS"; exit 1; }
# 1. at most 1 GPU session running (2 concurrent max)
python session_2026-10-15/v224f/running.py 6 | tee $D/running_before_push_$T.txt
N=$(grep RUNNING_COUNT $D/running_before_push_$T.txt | awk '{print $2}')
if [ "${N:-9}" -gt 1 ]; then echo "RUNNING_COUNT $N > 1: not pushing"; exit 1; fi
# 2. push
kaggle kernels push -p $D/kernel_$T 2>&1 | tee $D/push_$T.log
grep -q "successfully pushed" $D/push_$T.log || { echo "PUSH FAILED"; exit 1; }
# 3. wait (<= 45 min) and fetch into a NEW dir
python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$T $D/out_${T}_v1 45 | tee $D/out_${T}_v1.log
# 4. gate: DEC loaded + self-test + versions, no J_UNAVAILABLE, outputs == the base commit's
py -3.12 -I $D/j_check.py $D/out_${T}_v1 --base-out $BASE_OUT | tee $D/gate_${T}_v1.txt
tail -1 $D/gate_${T}_v1.txt
