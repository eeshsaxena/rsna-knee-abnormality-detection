#!/bin/bash
# k8_collect.sh (laptop, Git Bash; 2026-10-07, Claude; from k7_collect.sh). K8 member set (NOVEL_PLAN_10-07 sec 2 item 2):
#   set A = TTR224 f{0..4}_s1043_ttr_mix50     (the K5 members; folds 3-4 Kaggle-filled, receipt ~/fleet_runs/kaggle_f34_run_receipt.json)
#   set C = TTR288 f{0..4}_s1044_ttr288mix50   (box chain ~/auto_after_ttr288.sh; f0 done 10-07 00:09, f1 ~13:40,
#           f2 ~20:20, f3 ~03:00 10-08, f4 ~09:40-10:10 10-08)
#   set B (teammate s1044 TTR224) is DROPPED. Equal members, RAW best checkpoints (picker --tag raw), TTA A+B.
# One k8/box/k5_pick.py run per set, merged by k8/box/k8_merge.py (per-fold sanity bar for set C, 5/5 rule).
#   bash "$SP/k8/k8_collect.sh"
# Exit 0 = k8/members.json written (FINAL; or INTERIM with K8_INTERIM=1, marked do_not_submit);
#      1 = a check failed (STOP); 3 = set C not 5/5 yet (or set A not final) and K8_INTERIM unset.
# Env: K8T (box test area, default k8_test), K8_ROOT_A / K8_ROOT_C (box run roots, default ~/fleet_runs; the simulation
#      uses a scratch root), K8_INTERIM (1 = allow an INTERIM test build of the finished C folds; NEVER submit it),
#      K8_C_MARGIN (default .002: bar_F = set A f{F} cv - margin), K8_C_BARS ("F:bar,..": explicit per-fold bars, the
#      operator's call), K5_MIN_EPOCHS (default 16), K8_LAPTOP_DIR (default k8: where members.json + ckpt/ land).
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K8=$SP/k8
L=${K8_LAPTOP_DIR:-$K8}
B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
SSH="ssh -o BatchMode=yes $B"
SCP="scp -q -o BatchMode=yes"
K8T=${K8T:-k8_test}
RA=${K8_ROOT_A:-'~/fleet_runs'}; RC=${K8_ROOT_C:-'~/fleet_runs'}
MARGIN=${K8_C_MARGIN:-0.002}; BARS=${K8_C_BARS:-}
INTERIM=${K8_INTERIM:-0}
die(){ echo "K8COLLECT FAIL: $*"; exit 1; }
case "$K8T" in k8_test|k8_test/*) ;; *) die "K8T=$K8T: the box area must be ~/k8_test (or a k8_test/ subdir)";; esac
mkdir -p "$L/ckpt" "$L/ckpt_old" "$L/_real/store"

echo "== 1. sync box tools -> ~/$K8T/k5"
$SSH "mkdir -p ~/$K8T/k5/box ~/$K8T/k5/code ~/$K8T/ckpt ~/$K8T/oof ~/$K8T/runs ~/$K8T/ids ~/$K8T/kernel ~/$K8T/ds" < /dev/null || die "ssh"
$SCP "$K8/box/k5_pick.py" "$K8/box/k8_merge.py" "$K8/box/k5_box_launch.sh" "$K8/box/k2_box_stage.py" "$K8/box/k5_ids.py" $B:$K8T/k5/box/ || die "scp tools"
$SCP "$K8"/code/*.py $B:$K8T/k5/code/ || die "scp code"
$SCP "$K8/k2_check.py" $B:$K8T/k5/ || die "scp k2_check"

echo "== 2. box pick: set A (s1043 TTR224, raw), set C (s1044 TTR288 folds 0-4, raw)"
PK="CUDA_VISIBLE_DEVICES= K5_MIN_EPOCHS=${K5_MIN_EPOCHS:-16} nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/$K8T/k5/box/k5_pick.py --copy --dest ~/$K8T --tag raw"
$SSH "rm -f ~/$K8T/members*.json" < /dev/null
$SSH "$PK --root $RA --ttr_size 224; rc=\$?; mv ~/$K8T/members.json ~/$K8T/members_A.json 2>/dev/null; exit \$rc" < /dev/null; rcA=$?
$SSH "$PK --root $RC --seed 1044 --suffix ttr288mix50 --folds 0,1,2,3,4 --ttr_size 288 --no_kaggle_receipt; rc=\$?; mv ~/$K8T/members.json ~/$K8T/members_C.json 2>/dev/null; exit \$rc" < /dev/null; rcC=$?
echo "picker rc: set A $rcA, set C $rcC (3 = no C fold final yet; 1 = a check FAILED)"
[ $rcA -eq 3 ] && { echo "K8COLLECT: set A not final?"; exit 3; }
[ $rcA -ne 0 ] && die "set A (s1043) picker rc $rcA: read the FAIL lines above"
[ $rcC -eq 1 ] && die "set C (s1044 TTR288) picker check FAILED on a finished fold: read the FAIL lines above (do not ship)"
[ $rcC -ne 0 ] && [ $rcC -ne 3 ] && die "set C picker rc $rcC"
MA="--dest ~/$K8T --c_margin $MARGIN"
[ -n "$BARS" ] && MA="$MA --c_bars $BARS"
[ "$INTERIM" = "1" ] && MA="$MA --allow_interim"
$SSH "~/miniconda3/envs/kmimic/bin/python ~/$K8T/k5/box/k8_merge.py $MA" < /dev/null; rcM=$?
[ $rcM -eq 3 ] && { echo "K8COLLECT NOT FINAL: set C is not 5/5 (see above). Nothing to build for a submission; K8_INTERIM=1 for a TEST build only."; exit 3; }
[ $rcM -ne 0 ] && die "merge rc $rcM (read the K8MERGE line above)"
$SCP $B:$K8T/members.json "$L/members_box.json" || die "fetch members.json"

echo "== 3. fetch + verify checkpoints"
$PY - "$L/members_box.json" > "$L/.k8_want.txt" <<'EOF' || die "parse members_box.json"
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
    $SCP $B:$K8T/ckpt/$f "$L/ckpt/" || die "scp $f"
    have=$(sha256sum "$L/ckpt/$f" | cut -d' ' -f1)
  fi
  [ "$have" = "$s" ] || die "$f laptop sha256 $have != box source $s"
  echo "  OK $f sha256 ${s:0:16}"
done < "$L/.k8_want.txt"
for p in "$L"/ckpt/f*_s104[34]_*_best.pt; do
  [ -e "$p" ] || continue
  b=$(basename "$p")
  grep -q "^$b " "$L/.k8_want.txt" || { echo "  stale $b -> ckpt_old/"; mv "$p" "$L/ckpt_old/$b.$(date +%H%M%S)"; }
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
    print(f'  set {r["set"]} s{r["seed"]} fold {r["fold"]} ttr{r["ttr_size"]} {r["tag"]} epoch {r["epoch"]}/{r.get("epochs")} '
          f'cv {r["cv"]:.6f} {r["file"]} sha256 {r["sha256"][:16]} source {r.get("source")} OOF {r.get("oof_rescored_cv")}')
for s in m['skipped']:
    print(f'  NOT FINISHED set {s["set"]} f{s["fold"]}_s{s["seed"]}_{s["suffix"]}: {s["reason"][:100]}')
print(f'  set C: {m["set_c"]["status"]}')
n = {k: sum(r['set'] == k for r in m['members']) for k in 'AC'}
assert n['A'] == 5, n
sim = any('/k8_test/sim/' in str(r.get('run', '')) for r in m['members'])          # FIX 10-08: say SIMULATION
tail = '  [SIMULATION root: TEST ONLY, never version / push / submit]' if sim else ''
if m.get('do_not_submit'):
    print(f'K8COLLECT OK {len(m["members"])} member(s): A 5/5, C {n["C"]}/5  (INTERIM: TEST ONLY, do_not_submit, NEVER SUBMIT){tail}')
else:
    assert n['C'] == 5, n
    print(f'K8COLLECT OK {len(m["members"])} member(s): A 5/5, C 5/5  (FINAL){tail}')
EOF
exit 0
