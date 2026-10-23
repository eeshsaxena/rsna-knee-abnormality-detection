#!/bin/bash
# k12_collect.sh (laptop, Git Bash; 2026-10-09, Claude; from k11_collect.sh). ONE collect for the K12 kits:
#   C + D + E + E2 = the FINAL FN11 members (box ~/k11_test/members.json, read only; NOT copied to the laptop: the K12
#                    kernel reads them from the live k11w dataset)
#   DO = D-OAI f{0..4}_s1043_ttr_frac75oaiT12v00 (box runs): k12/box/k5_pick.py (K10 checks, labels_frac75oai_T12v00.csv,
#        min epochs 14, box mtime rule) -> k12/box/k12_oai_clean.py (K12-OAI-CLEAN on the box: ckpt dict + OOF vs set D's
#        OOF) -> k12/box/k12_merge.py (bar DO_F >= D_F - .005, PREREG_K12 sec 3). A DO checkpoint is copied to the laptop
#        ONLY after it passed K12-OAI-CLEAN on the box.
#   bash "$SP/k12/k12_collect.sh"
# Env: K12_INTERIM=1 (TEST builds: DO < 5/5), K12_SIM=1 (TEST: simulated DO folds 1-4 from set D weights, box/k12_sim_setup.py;
#      implies K12_INTERIM). Exit 0 = k12/members.json written; 1 = a check failed; 3 = DO not 5/5 (no K12_INTERIM).
# Box: CPU only (CUDA hidden, nice 19, taskset 34-39); writes only ~/k12_test; ~/fleet_runs, ~/k10_test, ~/k11_test read only.
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K12=$SP/k12
B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
PYK='~/miniconda3/envs/kmimic/bin/python'
SSH="ssh -o BatchMode=yes $B"
SCP="scp -q -o BatchMode=yes"
T=k12_test
SIM=${K12_SIM:-0}; INTERIM=${K12_INTERIM:-$SIM}
die(){ echo "K12COLLECT FAIL: $*"; exit 1; }
mkdir -p "$K12/ckpt" "$K12/ckpt_old"

echo "== 0. generated tools current"
$PY -B "$K12/make_k12_tools.py" --check | tail -n 1 | grep -q "MAKE_K12_TOOLS OK" || die "make_k12_tools --check"

echo "== 1. sync box tools -> ~/$T/k12"
$SSH "mkdir -p ~/$T/k12/box ~/$T/ckpt ~/$T/oof ~/$T/gold ~/$T/clean" < /dev/null || die "ssh"
$SCP "$K12/box/k5_pick.py" "$K12/box/k12_oai_clean.py" "$K12/box/k12_merge.py" "$K12/box/k12_gate.py" "$K12/box/k12_sim_setup.py" \
     "$K12/box/k12_gate_bg.sh" $B:$T/k12/box/ || die "scp tools"

echo "== 2. box pick DO (s1043 TTR224 frac75+OAI T12v00, box runs, min epochs 14)"
PK="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 taskset -c 34-39 nice -n 19 $PYK -W ignore ~/$T/k12/box/k5_pick.py --copy --dest ~/$T --tag raw"
$SSH "rm -f ~/$T/members.json ~/$T/members_DO.json; $PK --root ~/fleet_runs --seed 1043 --suffix ttr_frac75oaiT12v00 --folds 0,1,2,3,4 --ttr_size 224 --train_labels labels_frac75oai_T12v00.csv --key_tag f75oai --no_kaggle_receipt --min_epochs 14; rc=\$?; mv ~/$T/members.json ~/$T/members_DO.json 2>/dev/null; exit \$rc" < /dev/null; rc=$?
echo "picker rc: set DO $rc (0 ok; 3 none final; 1 a fold FAILED a check: excluded by the merge)"
[ $rc -ne 0 ] && [ $rc -ne 1 ] && [ $rc -ne 3 ] && die "DO picker rc $rc"
SIMA=""
if [ "$SIM" = "1" ]; then
  $SSH "cd ~ && CUDA_VISIBLE_DEVICES= taskset -c 34-39 nice -n 19 $PYK -W ignore ~/$T/k12/box/k12_sim_setup.py" < /dev/null | tail -n 1 | grep -q "K12SIM OK" || die "sim setup"
  SIMA="--sim_do ~/$T/sim/members_DO_sim.json"
fi

echo "== 3. K12-OAI-CLEAN on the box (every DO ckpt dict; real folds also their OOF vs set D's OOF)"
SPEC=$($SSH "$PYK - <<'EOF'
import json, os
H = os.path.expanduser('~')
D = {r['fold']: r for r in json.load(open(f'{H}/k11_test/members.json'))['members'] if r['set'] == 'D'}
ms = json.load(open(f'{H}/k12_test/members_DO.json'))['members'] if os.path.isfile(f'{H}/k12_test/members_DO.json') else []
sim = f'{H}/k12_test/sim/members_DO_sim.json'
have = {r['fold'] for r in ms}
ss = [r for r in json.load(open(sim))['members'] if r['fold'] not in have] if ('$SIM' == '1' and os.path.isfile(sim)) else []
print(' '.join([f'--ckpt {r[\"src\"]}={r[\"oof_src\"]}={D[r[\"fold\"]][\"oof\"]}' for r in ms] + [f'--ckpt {r[\"src\"]}' for r in ss]))
EOF" < /dev/null) || die "clean spec"
[ -n "$SPEC" ] || die "no DO checkpoint to scan"
$SSH "cd ~ && CUDA_VISIBLE_DEVICES= taskset -c 34-39 nice -n 19 $PYK -W ignore ~/$T/k12/box/k12_oai_clean.py --out ~/$T/clean/clean_DO.json $SPEC" < /dev/null | tee "$K12/logs/clean_$(date +%H%M).log" | tail -n 1
MA="--dest ~/$T"; [ "$INTERIM" = "1" ] && MA="$MA --allow_interim"; [ -n "$SIMA" ] && MA="$MA $SIMA"
$SSH "taskset -c 34-39 nice -n 19 $PYK ~/$T/k12/box/k12_merge.py $MA" < /dev/null; rcM=$?
[ $rcM -eq 3 ] && { echo "K12COLLECT NOT FINAL: DO is not 5/5. K12_INTERIM=1 / K12_SIM=1 for TEST builds only."; exit 3; }
[ $rcM -ne 0 ] && die "merge rc $rcM"
$SCP $B:$T/members.json "$K12/members_box.json" || die "fetch members.json"

echo "== 4. DO checkpoints -> k12/ckpt (only members that passed K12-OAI-CLEAN; sha256 == the box)"
$PY - "$K12/members_box.json" > "$K12/.k12_want.txt" <<'EOF' || die "parse members_box.json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m.get('k12') and not m['failed']
for r in m['members']:
    if r['set'] == 'DO':
        print(r['fold'], r['src'], r['sha256'], r['file'])
EOF
while read -r F src s f; do
  f=$(printf '%s' "$f" | tr -d '[:cntrl:]'); s=$(printf '%s' "$s" | tr -d '[:cntrl:]'); src=$(printf '%s' "$src" | tr -d '[:cntrl:]')
  [ -z "$f" ] && continue
  dst="$K12/ckpt/DO_f${F}__$f"
  have=""; [ -f "$dst" ] && have=$(sha256sum "$dst" | cut -d' ' -f1)
  if [ "$have" != "$s" ]; then
    [ -n "$have" ] && mv "$dst" "$K12/ckpt_old/$(basename "$dst").$(date +%H%M%S)"
    echo "  fetching DO f$F ($src)"; $SCP "$B:$src" "$dst" || die "scp $src"; have=$(sha256sum "$dst" | cut -d' ' -f1)
  fi
  [ "$have" = "$s" ] || die "DO f$F laptop sha256 $have != box $s"
  echo "  OK DO f$F $(basename "$dst") sha256 ${s:0:16}"
done < "$K12/.k12_want.txt"
for p in "$K12"/ckpt/DO_f*__*.pt; do
  [ -e "$p" ] || continue
  grep -q "$(basename "$p" | sed 's/^DO_f[0-4]__//')" "$K12/.k12_want.txt" || { echo "  stale $(basename "$p") -> ckpt_old/"; mv "$p" "$K12/ckpt_old/"; }
done

echo "== 5. members.json"
$PY - "$K12/members_box.json" "$K12/ckpt" "$K12/members.json" <<'EOF' || die "write members.json"
import hashlib, json, sys, time
m = json.load(open(sys.argv[1])); ck = sys.argv[2]
for r in m['members']:
    if r['set'] == 'DO':
        p = f'{ck}/DO_f{r["fold"]}__{r["file"]}'
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        assert h == r['sha256'], (p, h, r['sha256'])
        r['laptop_path'] = p.replace('\\', '/')
m['collected'] = time.strftime('%Y-%m-%d %H:%M:%S')
json.dump(m, open(sys.argv[3], 'w'), indent=1)
do = [r for r in m['members'] if r['set'] == 'DO']
for r in do:
    print(f'  DO f{r["fold"]} {"SIMULATED " if r.get("simulated") else ""}epoch {r["epoch"]}/{r.get("epochs")} cv {r["cv"]:.6f} '
          f'{r["file"]} sha256 {r["sha256"][:16]}')
tag = 'FINAL' if not m.get('do_not_submit') else ('SIMULATION + INTERIM: TEST ONLY' if m.get('simulated') else 'INTERIM: TEST ONLY')
print(f'K12COLLECT OK DO {len(do)}/5 + the FN11 members ({tag})')
EOF
exit 0
