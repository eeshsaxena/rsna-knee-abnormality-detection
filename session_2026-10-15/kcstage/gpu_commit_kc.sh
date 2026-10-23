#!/bin/bash
# gpu_commit_kc.sh (10-15, Claude subagent, kcstage): PREPARED, NOT RUN (the main session does every dataset, push and GPU commit).
# One GPU commit of an S6F+KC kernel built by build_v224s6fkc.py on the 3 public studies, then the box CPU parity and the gate.
# KCS2 fix: while Kaggle runs the commit, the box runs A6 G3 C9 on the SERVED path (box/c9_served.py = kc_prob_parity.py ca9cd0ca with
# the kc_serve substitution; the SAME member files, sha-checked vs KC_MANIFEST; 20 box train studies incl. weak-zone + SagPD-absent)
# -> OUT/c9_served.json. The gate REQUIRES it for a real KC1 build. KC_C9=skip skips it (stand-in commits only).
# NEVER submits. ~12-13 min of a T4x2 session (S6F kernel commit ~10 min + KC ~1.5-2 min MODEL: setup sha, env probe, side pass,
# model load, 3 studies, dump). Run from the project root (C:/Users/eeshs/Downloads/rsna-knee):
#   bash session_2026-10-15/kcstage/gpu_commit_kc.sh TAG BOX_MEMBER_1 [BOX_MEMBER_2 ...]
#   stand-in: bash session_2026-10-15/kcstage/gpu_commit_kc.sh v224s6fkcstandin \
#               kcstage_1015/standin/kc_standin_m1_s11_raw_fp32.pt kcstage_1015/standin/kc_standin_m2_s12_raw_fp16.pt
# BOX_MEMBER_k = the box paths (relative to the box home) of the SAME member files, in the manifest order (sha-checked by the tool).
# Before step 2 the KC dataset must exist on Kaggle (MAIN: kaggle datasets create -p <build receipt kc_dataset.local_dir>; for the real
# KC1 pack, run the OAI-string scan first and build the dir with make_kc_dataset.py --kind kc1).
set -u
cd "$(dirname "$0")/../.." || exit 1
TAG=$1; shift
D=session_2026-10-15/kcstage
OUT=$D/out_${TAG}_v1
BOX=training@192.168.10.30
verdict_ok() { tail -1 | tr -d '\r' | grep -qx "$1 PASS \([0-9][0-9]*\)/\1.*"; }
# 0. the kernel on disk is the built one; the cell-diff proof and every offline test pass
py -3.12 -I - "$TAG" <<'EOF' || exit 1
import sys, json, hashlib
tag = sys.argv[1]
r = json.load(open('session_2026-10-15/kcstage/build_receipt_%s.json' % tag))
m = json.load(open('session_2026-10-15/kcstage/kernel_%s/kernel-metadata.json' % tag))
h = hashlib.sha256(open('session_2026-10-15/kcstage/kernel_%s/%s' % (tag, m['code_file']), 'rb').read()).hexdigest()
assert h == r['notebook_sha256'], ('notebook differs from the build receipt', h)
print('kernel sha OK', h[:16], '| KC dataset', r['kc_dataset']['slug'], 'manifest', r['kc_dataset']['manifest_sha256'][:16],
      '| members', r['kc_dataset']['members'], '| standin' if r['kc_dataset']['standin'] else '| REAL KC1', '| never_submit', r['never_submit'])
EOF
py -3.12 -I $D/celldiff_v224s6fkc_proof.py $D/kernel_$TAG | verdict_ok KC_CELLDIFF || { echo "cell-diff proof FAILED"; exit 1; }
py -3.12 -I $D/run_offline_tests.py | tee $D/offline_tests_before_push_$TAG.txt | verdict_ok OFFLINE_TESTS || {
  echo "offline tests not all PASS (see $D/offline_tests_before_push_$TAG.txt)"; exit 1; }
# 1. at most 1 GPU session running (2 concurrent max)
python session_2026-10-15/v224f/running.py 6 | tee $D/running_before_push_$TAG.txt
N=$(grep RUNNING_COUNT $D/running_before_push_$TAG.txt | awk '{print $2}')
if [ "${N:-9}" -gt 1 ]; then echo "RUNNING_COUNT $N > 1: not pushing"; exit 1; fi
# 2. push
kaggle kernels push -p $D/kernel_$TAG 2>&1 | tee $D/push_$TAG.log
grep -q "successfully pushed" $D/push_$TAG.log || { echo "PUSH FAILED"; exit 1; }
# 2b. C9 on the served path (box CPU, in the background while Kaggle runs the commit; A6 G3, KCS2 fix)
ssh -o BatchMode=yes $BOX "rm -rf kcstage_1015/out_$TAG && mkdir -p kcstage_1015/out_$TAG" && \
scp -q -o BatchMode=yes $(py -3.12 -I -c "import json; print(json.load(open('$D/build_receipt_$TAG.json'))['kc_dataset']['local_dir'])")/KC_MANIFEST.json \
    $BOX:kcstage_1015/out_$TAG/KC_MANIFEST.json || { echo "box dir / manifest copy FAILED"; exit 1; }
C9PID=
if [ "${KC_C9:-run}" != "skip" ]; then
  ssh -o BatchMode=yes $BOX "cd ~ && taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
      ~/miniconda3/envs/kmimic/bin/python -u kcstage_1015/tools/c9_served.py --code kcstage_1015/code --serve kcstage_1015/kc_serve.py \
      --kpp kc/code/kc_prob_parity.py --members $* --manifest kcstage_1015/out_$TAG/KC_MANIFEST.json --store kc_store/store \
      --dicom eda_data --n 20 --threads 12 --out kcstage_1015/out_$TAG/c9_served.json" > $D/c9_served_$TAG.txt 2>&1 < /dev/null &
  C9PID=$!
fi
# 3. wait (<= 50 min) and fetch into a NEW dir
python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$TAG $OUT 50 | tee $OUT.log
# 4. box CPU parity of the SAME members on the dumped visible studies (RSNA public test data only goes to the box)
scp -q -r -o BatchMode=yes $OUT/kc $BOX:kcstage_1015/out_$TAG/ && \
ssh -o BatchMode=yes $BOX "cd ~ && taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
    ~/miniconda3/envs/kmimic/bin/python -u kcstage_1015/tools/kc_box_parity.py --out-dir kcstage_1015/out_$TAG --code kcstage_1015/code \
    --members $* --manifest kcstage_1015/out_$TAG/KC_MANIFEST.json --json kcstage_1015/out_$TAG/kc_box_parity.json --threads 12 --per-member" \
    | tee $D/box_parity_$TAG.txt
scp -q -o BatchMode=yes $BOX:kcstage_1015/out_$TAG/kc_box_parity.json $OUT/kc_box_parity.json
if [ -n "$C9PID" ]; then
  wait $C9PID; tail -1 $D/c9_served_$TAG.txt
  scp -q -o BatchMode=yes $BOX:kcstage_1015/out_$TAG/c9_served.json $OUT/c9_served.json
fi
# 5. gate (A base rows with KC removed, B box parity + side table, C blend, D coverage, E receipts / env / policy, F canary, G markers, H format)
py -3.12 -I $D/v224s6fkc_check.py $OUT | tee $D/gate_${TAG}_v1.txt
tail -1 $D/gate_${TAG}_v1.txt
