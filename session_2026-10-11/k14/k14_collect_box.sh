#!/bin/bash
# k14_collect_box.sh (2026-10-11, Claude): the BOX steps 1-3 of k12_collect.sh for the T123v00 DT set (PREREG_K14).
# k14_collect_box.sh (laptop, Git Bash; 2026-10-09, Claude; from k11_collect.sh). ONE collect for the K12 kits:
#   C + D + E + E2 = the FINAL FN11 members (box ~/k11_test/members.json, read only; NOT copied to the laptop: the K12
#                    kernel reads them from the live k11w dataset)
#   DO = D-OAI f{0..4}_s1043_ttr_frac75oaiT123v00 (box runs): k12/box/k5_pick.py (K10 checks, labels_frac75oai_T123v00.csv,
#        min epochs 14, box mtime rule) -> k12/box/k14_oai_clean.py (K12-OAI-CLEAN on the box: ckpt dict + OOF vs set D's
#        OOF) -> k12/box/k14_merge.py (bar DO_F >= D_F - .005, PREREG_K12 sec 3). A DO checkpoint is copied to the laptop
#        ONLY after it passed K12-OAI-CLEAN on the box.
#   bash "$SP/k14/k14_collect_box.sh"
# Env: K12_INTERIM=1 (TEST builds: DO < 5/5), K12_SIM=1 (TEST: simulated DO folds 1-4 from set D weights, box/k12_sim_setup.py;
#      implies K12_INTERIM). Exit 0 = k12/members.json written; 1 = a check failed; 3 = DO not 5/5 (no K12_INTERIM).
# Box: CPU only (CUDA hidden, nice 19, taskset 34-39); writes only ~/k14_test; ~/fleet_runs, ~/k10_test, ~/k11_test read only.
set -uo pipefail
SP=C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-11
K12=$SP/k14
B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
PYK='~/miniconda3/envs/kmimic/bin/python'
SSH="ssh -o BatchMode=yes $B"
SCP="scp -q -o BatchMode=yes"
T=k14_test
SIM=${K12_SIM:-0}; INTERIM=${K12_INTERIM:-$SIM}
die(){ echo "K14COLLECT FAIL: $*"; exit 1; }
mkdir -p "$K12/ckpt" "$K12/ckpt_old"


echo "== 1. sync box tools -> ~/$T/k12"
$SSH "mkdir -p ~/$T/k14/box ~/$T/ckpt ~/$T/oof ~/$T/gold ~/$T/clean" < /dev/null || die "ssh"
$SCP "$K12/box/k5_pick.py" "$K12/box/k14_oai_clean.py" "$K12/box/k14_merge.py" "$K12/box/k14_gate.py" \
     "$K12/box/k14_gate_bg.sh" $B:$T/k14/box/ || die "scp tools"

echo "== 2. box pick DO (s1043 TTR224 frac75+OAI T123v00, box runs, min epochs 14)"
PK="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 taskset -c 34-39 nice -n 19 $PYK -W ignore ~/$T/k14/box/k5_pick.py --copy --dest ~/$T --tag raw"
$SSH "rm -f ~/$T/members.json ~/$T/members_DO.json; $PK --root ~/fleet_runs --seed 1043 --suffix ttr_frac75oaiT123v00 --folds 0,1,2,3,4 --ttr_size 224 --train_labels labels_frac75oai_T123v00.csv --key_tag f75oaiT123 --no_kaggle_receipt --min_epochs 14; rc=\$?; mv ~/$T/members.json ~/$T/members_DO.json 2>/dev/null; exit \$rc" < /dev/null; rc=$?
echo "picker rc: set DO $rc (0 ok; 3 none final; 1 a fold FAILED a check: excluded by the merge)"
[ $rc -ne 0 ] && [ $rc -ne 1 ] && [ $rc -ne 3 ] && die "DO picker rc $rc"
SIMA=""
if [ "$SIM" = "1" ]; then
  $SSH "cd ~ && CUDA_VISIBLE_DEVICES= taskset -c 34-39 nice -n 19 $PYK -W ignore ~/$T/k14/box/k12_sim_setup.py" < /dev/null | tail -n 1 | grep -q "K12SIM OK" || die "sim setup"
  SIMA="--sim_do ~/$T/sim/members_DO_sim.json"
fi

echo "== 3. K12-OAI-CLEAN on the box (every DO ckpt dict; real folds also their OOF vs set D's OOF)"
SPEC=$($SSH "$PYK - <<'EOF'
import json, os
H = os.path.expanduser('~')
D = {r['fold']: r for r in json.load(open(f'{H}/k11_test/members.json'))['members'] if r['set'] == 'D'}
ms = json.load(open(f'{H}/k14_test/members_DO.json'))['members'] if os.path.isfile(f'{H}/k14_test/members_DO.json') else []
sim = f'{H}/k14_test/sim/members_DO_sim.json'
have = {r['fold'] for r in ms}
ss = [r for r in json.load(open(sim))['members'] if r['fold'] not in have] if ('$SIM' == '1' and os.path.isfile(sim)) else []
print(' '.join([f'--ckpt {r[\"src\"]}={r[\"oof_src\"]}={D[r[\"fold\"]][\"oof\"]}' for r in ms] + [f'--ckpt {r[\"src\"]}' for r in ss]))
EOF" < /dev/null) || die "clean spec"
[ -n "$SPEC" ] || die "no DO checkpoint to scan"
$SSH "cd ~ && CUDA_VISIBLE_DEVICES= taskset -c 34-39 nice -n 19 $PYK -W ignore ~/$T/k14/box/k14_oai_clean.py --out ~/$T/clean/clean_DO.json $SPEC" < /dev/null | tee "$K12/logs/clean_$(date +%H%M).log" | tail -n 1
MA="--dest ~/$T"; [ "$INTERIM" = "1" ] && MA="$MA --allow_interim"; [ -n "$SIMA" ] && MA="$MA $SIMA"
$SSH "taskset -c 34-39 nice -n 19 $PYK ~/$T/k14/box/k14_merge.py $MA" < /dev/null; rcM=$?
[ $rcM -eq 3 ] && { echo "K14COLLECT NOT FINAL: DO is not 5/5. K12_INTERIM=1 / K12_SIM=1 for TEST builds only."; exit 3; }
[ $rcM -ne 0 ] && die "merge rc $rcM"
$SCP $B:$T/members.json "$K12/members_box.json" || die "fetch members.json"
echo "K14COLLECT BOX OK (members on the box: ~/$T/members.json; laptop fetch only at build time)"
exit 0
