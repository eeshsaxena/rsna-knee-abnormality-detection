#!/bin/bash
# k9_collect.sh (laptop, Git Bash; 2026-10-08, Claude; from k8_collect.sh). K9 member set (NOVEL_PLAN_10-07 sec 4 row 10-09):
#   set C = TTR288 f{0..4}_s1044_ttr288mix50   mix50 labels (box chain ~/auto_after_ttr288.sh; f4 ends ~09:45 10-08)
#   set D = TTR224 f{0..4}_s1043_ttr_frac75    frac75 labels (Kaggle: swapiy f01 E17 -> receipt kaggle_frac75_f01_run_receipt.json;
#           kmrsuman f23 + f4x E~21 -> kaggle_frac75_{f23,f4x}_run_receipt.json; every D fold needs its receipt)
#   set A (the K5 mix50 s1043 members) is REPLACED by set D; set B is dropped. Equal members, RAW best checkpoints, TTA A+B.
# One k9/box/k5_pick.py run per set (set D: --train_labels labels_frac75_logit.csv --key_tag f75 --require_receipt,
# K5_MIN_EPOCHS 14 for the E17 folds, approved 10-06), merged by k9/box/k9_merge.py (per-fold bars, 5/5 rule for both sets).
#   bash "$SP/k9/k9_collect.sh"
# Exit 0 = k9/members.json written (FINAL; or INTERIM with K9_INTERIM=1, marked do_not_submit);
#      1 = a check failed (STOP); 3 = a set is not 5/5 yet and K9_INTERIM unset.
# Env: K9T (box test area, default k9_test; must be k9_test or a k9_test/ subdir), K9_ROOT_C / K9_ROOT_D (box run roots,
#      default ~/fleet_runs; the simulation uses ~/k9_test/sim/root), K9_INTERIM (1 = TEST build of the finished folds;
#      NEVER submit it), K9_C_MARGIN (.002) / K9_D_MARGIN (.004) (bar_F = set A f{F} cv - margin), K9_C_BARS / K9_D_BARS
#      ("F:bar,..": explicit per-fold bars, the operator's call), K9_C_MIN_EPOCHS (16) / K9_D_MIN_EPOCHS (14),
#      K9_LAPTOP_DIR (default k9: where members.json + ckpt/ land).
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K9=$SP/k9
L=${K9_LAPTOP_DIR:-$K9}
B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
SSH="ssh -o BatchMode=yes $B"
SCP="scp -q -o BatchMode=yes"
K9T=${K9T:-k9_test}
RC=${K9_ROOT_C:-'~/fleet_runs'}; RD=${K9_ROOT_D:-'~/fleet_runs'}
INTERIM=${K9_INTERIM:-0}
die(){ echo "K9COLLECT FAIL: $*"; exit 1; }
case "$K9T" in k9_test|k9_test/*) ;; *) die "K9T=$K9T: the box area must be ~/k9_test (or a k9_test/ subdir)";; esac
mkdir -p "$L/ckpt" "$L/ckpt_old" "$L/_real/store"

echo "== 1. sync box tools -> ~/$K9T/k5"
$SSH "mkdir -p ~/$K9T/k5/box ~/$K9T/k5/code ~/$K9T/ckpt ~/$K9T/oof ~/$K9T/runs ~/$K9T/ids ~/$K9T/kernel ~/$K9T/ds" < /dev/null || die "ssh"
$SCP "$K9/box/k5_pick.py" "$K9/box/k9_merge.py" "$K9/box/k5_box_launch.sh" "$K9/box/k2_box_stage.py" "$K9/box/k5_ids.py" $B:$K9T/k5/box/ || die "scp tools"
$SCP "$K9"/code/*.py $B:$K9T/k5/code/ || die "scp code"
$SCP "$K9/k2_check.py" $B:$K9T/k5/ || die "scp k2_check"

echo "== 2. box pick: set C (s1044 TTR288 mix50 folds 0-4, raw), set D (s1043 TTR224 frac75 folds 0-4, raw, receipts)"
PK="CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/$K9T/k5/box/k5_pick.py --copy --dest ~/$K9T --tag raw"
REC="--kaggle_receipt $RD/kaggle_frac75_f01_run_receipt.json --kaggle_receipt $RD/kaggle_frac75_f23_run_receipt.json --kaggle_receipt $RD/kaggle_frac75_f4x_run_receipt.json"
$SSH "rm -f ~/$K9T/members*.json" < /dev/null
$SSH "K5_MIN_EPOCHS=${K9_C_MIN_EPOCHS:-16} $PK --root $RC --seed 1044 --suffix ttr288mix50 --folds 0,1,2,3,4 --ttr_size 288 --no_kaggle_receipt; rc=\$?; mv ~/$K9T/members.json ~/$K9T/members_C.json 2>/dev/null; exit \$rc" < /dev/null; rcC=$?
$SSH "K5_MIN_EPOCHS=${K9_D_MIN_EPOCHS:-14} $PK --root $RD --seed 1043 --suffix ttr_frac75 --folds 0,1,2,3,4 --ttr_size 224 --train_labels labels_frac75_logit.csv --key_tag f75 --require_receipt $REC; rc=\$?; mv ~/$K9T/members.json ~/$K9T/members_D.json 2>/dev/null; exit \$rc" < /dev/null; rcD=$?
echo "picker rc: set C $rcC, set D $rcD (3 = no fold of that set final yet; 1 = a check FAILED)"
[ $rcC -eq 1 ] && die "set C (s1044 TTR288) picker check FAILED on a finished fold: read the FAIL lines above (do not ship)"
[ $rcD -eq 1 ] && die "set D (s1043 frac75) picker check FAILED on a finished fold: read the FAIL lines above (do not ship)"
[ $rcC -ne 0 ] && [ $rcC -ne 3 ] && die "set C picker rc $rcC"
[ $rcD -ne 0 ] && [ $rcD -ne 3 ] && die "set D picker rc $rcD"
MA="--dest ~/$K9T --c_margin ${K9_C_MARGIN:-0.002} --d_margin ${K9_D_MARGIN:-0.004}"
[ -n "${K9_C_BARS:-}" ] && MA="$MA --c_bars $K9_C_BARS"
[ -n "${K9_D_BARS:-}" ] && MA="$MA --d_bars $K9_D_BARS"
[ "$INTERIM" = "1" ] && MA="$MA --allow_interim"
$SSH "~/miniconda3/envs/kmimic/bin/python ~/$K9T/k5/box/k9_merge.py $MA" < /dev/null; rcM=$?
[ $rcM -eq 3 ] && { echo "K9COLLECT NOT FINAL: a set is not 5/5 (see above). Nothing to build for a submission; K9_INTERIM=1 for a TEST build only."; exit 3; }
[ $rcM -ne 0 ] && die "merge rc $rcM (read the K9MERGE line above)"
$SCP $B:$K9T/members.json "$L/members_box.json" || die "fetch members.json"

echo "== 3. fetch + verify checkpoints"
$PY - "$L/members_box.json" > "$L/.k9_want.txt" <<'EOF' || die "parse members_box.json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m['members'] and not m['failed'], m.get('failed')
for r in m['members']:
    print(r['file'], r['sha256'])
EOF
while read -r f s; do
  s=$(printf '%s' "$s" | tr -d '[:cntrl:]'); f=$(printf '%s' "$f" | tr -d '[:cntrl:]')
  [ -z "$f" ] && continue
  have=""
  [ -f "$L/ckpt/$f" ] && have=$(sha256sum "$L/ckpt/$f" | cut -d' ' -f1)
  if [ "$have" != "$s" ]; then
    [ -n "$have" ] && { echo "  $f on the laptop has other bytes ($have): moving it to ckpt_old/"; mv "$L/ckpt/$f" "$L/ckpt_old/$f.$(date +%H%M%S)"; }
    echo "  fetching $f"
    $SCP $B:$K9T/ckpt/$f "$L/ckpt/" || die "scp $f"
    have=$(sha256sum "$L/ckpt/$f" | cut -d' ' -f1)
  fi
  [ "$have" = "$s" ] || die "$f laptop sha256 $have != box source $s"
  echo "  OK $f sha256 ${s:0:16}"
done < "$L/.k9_want.txt"
for p in "$L"/ckpt/f*_s104[34]_*_best.pt; do
  [ -e "$p" ] || continue
  b=$(basename "$p")
  grep -q "^$b " "$L/.k9_want.txt" || { echo "  stale $b -> ckpt_old/"; mv "$p" "$L/ckpt_old/$b.$(date +%H%M%S)"; }
done

echo "== 4. parity npz (decode_probe studies)"
SIDS=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); print(' '.join(sorted({s for r in m['members'] for s in r['decode_probe']})))" "$L/members_box.json" | tr -d '[:cntrl:]') || die "probe list"
BOXSHA=$($SSH "cd ~/fleet_store && sha256sum $(for s in $SIDS; do printf '%s.npz ' "$s"; done)" < /dev/null) || die "box npz sha"
for s in $SIDS; do
  want=$(echo "$BOXSHA" | grep " $s.npz$" | cut -d' ' -f1)
  have=""; [ -f "$L/_real/store/$s.npz" ] && have=$(sha256sum "$L/_real/store/$s.npz" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then echo "  fetching $s.npz"; $SCP $B:fleet_store/$s.npz "$L/_real/store/" || die "scp $s.npz"
    have=$(sha256sum "$L/_real/store/$s.npz" | cut -d' ' -f1); fi
  [ -n "$want" ] && [ "$have" = "$want" ] || die "parity npz $s: laptop $have != box $want"
  echo "  OK $s.npz ${want:0:16}"
done

echo "== 5. members.json"
$PY - "$L/members_box.json" "$L/ckpt" "$L/members.json" <<'EOF' || die "write members.json"
import hashlib, json, os, sys, time
m = json.load(open(sys.argv[1])); ck = sys.argv[2]
for r in m['members']:
    p = f'{ck}/{r["file"]}'
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    assert h == r['sha256'], (r['file'], h, r['sha256'])
    r['laptop_path'] = p.replace('\\', '/'); r['laptop_sha256_ok'] = True
m['collected'] = time.strftime('%Y-%m-%d %H:%M:%S')
json.dump(m, open(sys.argv[3], 'w'), indent=1)
print(f'members ({len(m["members"])}):')
for r in m['members']:
    print(f'  set {r["set"]} s{r["seed"]} fold {r["fold"]} ttr{r["ttr_size"]} {r["train_labels"]} {r["tag"]} epoch {r["epoch"]}/{r.get("epochs")} '
          f'cv {r["cv"]:.6f} {r["file"]} sha256 {r["sha256"][:16]} source {r.get("source")} OOF {r.get("oof_rescored_cv")}')
for s in m['skipped']:
    print(f'  NOT FINISHED set {s["set"]} f{s["fold"]}_s{s["seed"]}_{s["suffix"]}: {s["reason"][:100]}')
print(f'  set C: {m["set_c"]["status"]} | set D: {m["set_d"]["status"]}')
n = {k: sum(r['set'] == k for r in m['members']) for k in 'CD'}
assert set(r['set'] for r in m['members']) <= {'C', 'D'}, 'K9 = sets C + D only'
sim = any('/k9_test/sim/' in str(r.get('run', '')) for r in m['members'])
tail = '  [SIMULATION root: TEST ONLY, never version / push / submit]' if sim else ''
if m.get('do_not_submit'):
    print(f'K9COLLECT OK {len(m["members"])} member(s): C {n["C"]}/5, D {n["D"]}/5  (INTERIM: TEST ONLY, do_not_submit, NEVER SUBMIT){tail}')
else:
    assert n['C'] == 5 and n['D'] == 5, n
    print(f'K9COLLECT OK {len(m["members"])} member(s): C 5/5, D 5/5  (FINAL){tail}')
EOF
exit 0
