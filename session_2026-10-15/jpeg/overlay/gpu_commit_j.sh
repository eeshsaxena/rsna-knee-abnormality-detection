#!/bin/bash
# gpu_commit_j.sh TAG (2026-10-15, Claude subagent): PREPARED, NOT RUN (the main session does every GPU commit). The one GPU commit of a
# J kernel built by make_j.py (private, T4 x2, internet off, docker pin 2757e0c7 = its base's) on the 3 public studies, then the gate
# against the BASE commit output. NEVER submits. TAG in: v224flipj | v224flips6fj | v224flips20j.
# Cost (MODEL): the base commit's time + ~6 s DEC + ~2 s preflight scan (base log ends: FLIP 481 s, S6F 591 s, FLIPS20 527 s).
# Run from the project root (C:/Users/eeshs/Downloads/rsna-knee):  bash session_2026-10-15/jpeg/overlay/gpu_commit_j.sh v224flipj
# A re-run pushes a NEW version of the same slug: rename the _v1 out / log / gate paths first.
# Review rebuild (10-15): pushes r2/kernel_TAG (DEC cell with the J1 test-data preflight and the J2 /proc envwatch). The r1
# kernel_TAG dirs next to this script are superseded: do not push them. The gate now also requires dec_envwatch.json.
set -u
cd "$(dirname "$0")/../../.." || exit 1
D=session_2026-10-15/jpeg/overlay
K=$D/r2
T=${1:?usage: gpu_commit_j.sh TAG}
case "$T" in
  v224flipj)    BASE_OUT=session_2026-10-14/v224/out_v224flip_v1 ;;
  v224flips6fj) BASE_OUT=session_2026-10-15/s6f/out_v224flips6f_v1 ;;
  v224flips20j) BASE_OUT=session_2026-10-15/v224f/out_v224flips20_v1 ;;
  *) echo "unknown tag $T"; exit 1 ;;
esac
verdict_ok() { tail -1 | tr -d '\r' | grep -q "^$1 PASS \([0-9][0-9]*\)/\1\b"; }
# 0. the kernel on disk is the built one (review-fixed DEC cell); every offline J check re-run now on the current files
py -3.12 -I - "$T" <<'EOF' || exit 1
import json, hashlib, sys
t = sys.argv[1]
K = 'session_2026-10-15/jpeg/overlay/r2'
r = json.load(open('%s/j_build_receipt_%s.json' % (K, t)))
raw = open('%s/kernel_%s/rsna_sub_%s.ipynb' % (K, t, t), 'rb').read()
h = hashlib.sha256(raw).hexdigest()
assert h == r['notebook_sha256'], ('notebook differs from the build receipt', h)
src = raw.decode('utf-8')
assert 'PRE_TIMEOUT_S, WATCH_EVERY_S, PROC' in src and 'dec_envwatch' in src, 'not the review-fixed DEC cell'
m = json.load(open('%s/kernel_%s/kernel-metadata.json' % (K, t)))
assert m['id'] == 'eeshsaxena/rsna-sub-' + t and m['is_private'] is True and m['enable_internet'] is False and 'id_no' not in m, m
print('kernel sha OK', h[:16], m['id'])
EOF
py -3.12 -I $D/run_offline_j.py "$T" | tee $K/offline_before_push_$T.txt | verdict_ok J_OFFLINE || { echo "offline J checks not all PASS"; exit 1; }
# 1. at most 1 GPU session running (2 concurrent max)
python session_2026-10-15/v224f/running.py 6 | tee $K/running_before_push_$T.txt
N=$(grep RUNNING_COUNT $K/running_before_push_$T.txt | awk '{print $2}')
if [ "${N:-9}" -gt 1 ]; then echo "RUNNING_COUNT $N > 1: not pushing"; exit 1; fi
# 2. push
kaggle kernels push -p $K/kernel_$T 2>&1 | tee $K/push_$T.log
grep -q "successfully pushed" $K/push_$T.log || { echo "PUSH FAILED"; exit 1; }
# 3. wait (<= 45 min) and fetch into a NEW dir
python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$T $K/out_${T}_v1 45 | tee $K/out_${T}_v1.log
# 4. gate: DEC loaded + self-test + versions + preflight, envwatch (every stage process had /tmp/_dec on PYTHONPATH, no -I/-E/-S),
#    S6F receipts (S6F kernels), no J_UNAVAILABLE, outputs == the base commit's
py -3.12 -I $D/j_check.py $K/out_${T}_v1 --base-out $BASE_OUT | tee $K/gate_${T}_v1.txt
tail -1 $K/gate_${T}_v1.txt
