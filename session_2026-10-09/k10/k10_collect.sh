#!/bin/bash
# k10_collect.sh (laptop, Git Bash; 2026-10-09, Claude; from k9_collect.sh). ONE collect for every K10 kernel:
#   set E = TTR288 f{0..4}_s1045_ttr288frac75  frac75, 21 ep (f0 box V100; f1-f4 Vast, receipts vast320_<run>_receipt.json)
#   set D = TTR224 f{0..4}_s1043_ttr_frac75    frac75 (Kaggle, receipts kaggle_frac75_{f01,f23,f4x}_run_receipt.json)
#   set C = TTR288 f{0..4}_s1044_ttr288mix50   mix50 (box)            C and D must equal the K9 picks byte for byte
# One k10/box/k5_pick.py run per set (E: --key_tag t288f75 --vast_receipt_folds $K10_E_VAST_FOLDS), merged by
# k10/box/k10_merge.py (E bars = set C cv - .004; D / C bars as K9), then the checkpoints are fetched + sha-verified
# (C / D are copied from SP/k9/ckpt when the bytes match, read only; E is scp'd from the box), parity npz, k10/members.json.
#   bash "$SP/k10/k10_collect.sh"
# Exit 0 = members.json written (FINAL; or INTERIM with K10_INTERIM=1, marked do_not_submit); 1 = a check failed (STOP);
#      3 = a set is not 5/5 yet and K10_INTERIM unset.
# Env: K10T (box area, default k10_test; must be k10_test or a k10_test/ subdir), K10_ROOT_C / _D / _E (box run roots,
#      default ~/fleet_runs; the simulation uses ~/k10_test/sim/root), K10_INTERIM, K10_E_MARGIN (.004) / K10_D_MARGIN
#      (.004) / K10_C_MARGIN (.002), K10_E_BARS / K10_D_BARS / K10_C_BARS ("F:bar,.."), K10_E_MIN_EPOCHS (16) /
#      K10_D_MIN_EPOCHS (14) / K10_C_MIN_EPOCHS (16), K10_E_VAST_FOLDS (default 1,2,3,4), K10_LAPTOP_DIR (default k10).
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K10=$SP/k10
L=${K10_LAPTOP_DIR:-$K10}
B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
SSH="ssh -o BatchMode=yes $B"
SCP="scp -q -o BatchMode=yes"
K10T=${K10T:-k10_test}
RC=${K10_ROOT_C:-'~/fleet_runs'}; RD=${K10_ROOT_D:-'~/fleet_runs'}; RE=${K10_ROOT_E:-'~/fleet_runs'}
INTERIM=${K10_INTERIM:-0}
VF=${K10_E_VAST_FOLDS-1,2,3,4}
die(){ echo "K10COLLECT FAIL: $*"; exit 1; }
case "$K10T" in k10_test|k10_test/*) ;; *) die "K10T=$K10T: the box area must be ~/k10_test (or a k10_test/ subdir)";; esac
mkdir -p "$L/ckpt" "$L/ckpt_old" "$L/_real/store"

echo "== 1. sync box tools -> ~/$K10T/k10"
$SSH "mkdir -p ~/$K10T/k10/box ~/$K10T/ckpt ~/$K10T/oof ~/$K10T/runs ~/$K10T/ids ~/$K10T/kernel ~/$K10T/ds" < /dev/null || die "ssh"
$SCP "$K10/box/k5_pick.py" "$K10/box/k10_merge.py" "$K10/box/k10_gold_read.py" "$K10/box/k10_gold_bg.sh" $B:$K10T/k10/box/ || die "scp tools"

echo "== 2. box pick: E (s1045 TTR288 frac75, Vast receipts f$VF), D (s1043 TTR224 frac75, Kaggle receipts), C (s1044 TTR288 mix50)"
PK="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 taskset -c 34-39 nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/$K10T/k10/box/k5_pick.py --copy --dest ~/$K10T --tag raw"
REC="--kaggle_receipt $RD/kaggle_frac75_f01_run_receipt.json --kaggle_receipt $RD/kaggle_frac75_f23_run_receipt.json --kaggle_receipt $RD/kaggle_frac75_f4x_run_receipt.json"
$SSH "rm -f ~/$K10T/members*.json" < /dev/null
$SSH "K5_MIN_EPOCHS=${K10_E_MIN_EPOCHS:-16} $PK --root $RE --seed 1045 --suffix ttr288frac75 --folds 0,1,2,3,4 --ttr_size 288 --train_labels labels_frac75_logit.csv --key_tag t288f75 --no_kaggle_receipt --vast_receipt_folds '$VF'; rc=\$?; mv ~/$K10T/members.json ~/$K10T/members_E.json 2>/dev/null; exit \$rc" < /dev/null; rcE=$?
$SSH "K5_MIN_EPOCHS=${K10_D_MIN_EPOCHS:-14} $PK --root $RD --seed 1043 --suffix ttr_frac75 --folds 0,1,2,3,4 --ttr_size 224 --train_labels labels_frac75_logit.csv --key_tag f75 --require_receipt $REC; rc=\$?; mv ~/$K10T/members.json ~/$K10T/members_D.json 2>/dev/null; exit \$rc" < /dev/null; rcD=$?
$SSH "K5_MIN_EPOCHS=${K10_C_MIN_EPOCHS:-16} $PK --root $RC --seed 1044 --suffix ttr288mix50 --folds 0,1,2,3,4 --ttr_size 288 --no_kaggle_receipt; rc=\$?; mv ~/$K10T/members.json ~/$K10T/members_C.json 2>/dev/null; exit \$rc" < /dev/null; rcC=$?
echo "picker rc: set E $rcE, set D $rcD, set C $rcC (3 = no fold of that set final yet; 1 = a check FAILED)"
for x in "E $rcE" "D $rcD" "C $rcC"; do set -- $x
  [ $2 -eq 1 ] && die "set $1 picker check FAILED on a finished fold: read the FAIL lines above (do not ship)"
  [ $2 -ne 0 ] && [ $2 -ne 3 ] && die "set $1 picker rc $2"
done
MA="--dest ~/$K10T --e_margin ${K10_E_MARGIN:-0.004} --d_margin ${K10_D_MARGIN:-0.004} --c_margin ${K10_C_MARGIN:-0.002} --e_vast_folds '$VF'"
[ -n "${K10_E_BARS:-}" ] && MA="$MA --e_bars $K10_E_BARS"
[ -n "${K10_D_BARS:-}" ] && MA="$MA --d_bars $K10_D_BARS"
[ -n "${K10_C_BARS:-}" ] && MA="$MA --c_bars $K10_C_BARS"
[ "$INTERIM" = "1" ] && MA="$MA --allow_interim"
$SSH "taskset -c 34-39 nice -n 19 ~/miniconda3/envs/kmimic/bin/python ~/$K10T/k10/box/k10_merge.py $MA" < /dev/null; rcM=$?
[ $rcM -eq 3 ] && { echo "K10COLLECT NOT FINAL: a set is not 5/5 (see above). K10_INTERIM=1 for a TEST build only."; exit 3; }
[ $rcM -ne 0 ] && die "merge rc $rcM (read the K10MERGE lines above)"
$SCP $B:$K10T/members.json "$L/members_box.json" || die "fetch members.json"

echo "== 3. fetch + verify checkpoints (C / D from SP/k9/ckpt when the bytes match; E from the box)"
$PY - "$L/members_box.json" > "$L/.k10_want.txt" <<'EOF' || die "parse members_box.json"
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
    k9=""; [ -f "$SP/k9/ckpt/$f" ] && k9=$(sha256sum "$SP/k9/ckpt/$f" | cut -d' ' -f1)
    if [ "$k9" = "$s" ]; then echo "  copying $f from SP/k9/ckpt (K9 pick, same sha256)"; cp -p "$SP/k9/ckpt/$f" "$L/ckpt/" || die "cp $f"
    else echo "  fetching $f"; $SCP $B:$K10T/ckpt/$f "$L/ckpt/" || die "scp $f"; fi
    have=$(sha256sum "$L/ckpt/$f" | cut -d' ' -f1)
  fi
  [ "$have" = "$s" ] || die "$f laptop sha256 $have != box source $s"
  echo "  OK $f sha256 ${s:0:16}"
done < "$L/.k10_want.txt"
for p in "$L"/ckpt/f*_s104[345]_*_best.pt; do
  [ -e "$p" ] || continue
  b=$(basename "$p")
  grep -q "^$b " "$L/.k10_want.txt" || { echo "  stale $b -> ckpt_old/"; mv "$p" "$L/ckpt_old/$b.$(date +%H%M%S)"; }
done

echo "== 4. parity npz (decode_probe studies)"
SIDS=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); print(' '.join(sorted({s for r in m['members'] for s in r['decode_probe']})))" "$L/members_box.json" | tr -d '[:cntrl:]') || die "probe list"
BOXSHA=$($SSH "cd ~/fleet_store && sha256sum $(for s in $SIDS; do printf '%s.npz ' "$s"; done)" < /dev/null) || die "box npz sha"
for s in $SIDS; do
  want=$(echo "$BOXSHA" | grep " $s.npz$" | cut -d' ' -f1)
  have=""; [ -f "$L/_real/store/$s.npz" ] && have=$(sha256sum "$L/_real/store/$s.npz" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then
    k9=""; [ -f "$SP/k9/_real/store/$s.npz" ] && k9=$(sha256sum "$SP/k9/_real/store/$s.npz" | cut -d' ' -f1)
    if [ -n "$want" ] && [ "$k9" = "$want" ]; then cp -p "$SP/k9/_real/store/$s.npz" "$L/_real/store/"
    else echo "  fetching $s.npz"; $SCP $B:fleet_store/$s.npz "$L/_real/store/" || die "scp $s.npz"; fi
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
print(f'  set E: {m["set_e"]["status"]} | set D: {m["set_d"]["status"]} | set C: {m["set_c"]["status"]}')
n = {k: sum(r['set'] == k for r in m['members']) for k in 'EDC'}
assert set(r['set'] for r in m['members']) <= {'C', 'D', 'E'}, 'K10 = sets C / D / E only'
sim = any('/k10_test/sim/' in str(r.get('run', '')) for r in m['members'])
tail = '  [SIMULATION root: TEST ONLY, never version / push / submit]' if sim else ''
if m.get('do_not_submit'):
    print(f'K10COLLECT OK {len(m["members"])} member(s): E {n["E"]}/5, D {n["D"]}/5, C {n["C"]}/5  (INTERIM: TEST ONLY, do_not_submit, NEVER SUBMIT){tail}')
else:
    assert n == {'E': 5, 'D': 5, 'C': 5}, n
    print(f'K10COLLECT OK {len(m["members"])} member(s): E 5/5, D 5/5, C 5/5  (FINAL){tail}')
EOF
exit 0
