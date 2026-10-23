#!/bin/bash
# k7_collect.sh (laptop, Git Bash; 2026-10-06, Claude). = k6_collect.sh for the K7 member set (mix50 labels only):
#   set A = TTR224 f{0..4}_s1043_ttr_mix50   (folds 3-4 Kaggle-filled, receipt ~/fleet_runs/kaggle_f34_run_receipt.json)
#   set B = TTR224 f{0..4}_s1044_ttr_mix50   (teammate swapiy's Kaggle runs; receipts REQUIRED:
#           <root B>/kaggle_s1044_f01_run_receipt.json, ..._f23_..., ..._f4_... (K7_RUNBOOK sec 2 copies them there))
#   set C = TTR288 f0_s1044_ttr288mix50      (box; optional: included only when its picked cv >= .8876; K7_WITH_C=0 excludes)
# One k7/box/k5_pick.py run per set (K7 picker: --ttr_size, repeatable --kaggle_receipt with seed + run-name match,
# --require_receipt, recipe-aware stale removal), merged by k7/box/k7_merge.py. Unfinished runs are SKIPPED and listed.
#   bash "$SP/k7/k7_collect.sh"     # exit 0 = k7/members.json written; 1 = a check failed (STOP); 3 = set A not final
# Env: K7T (box test area, default k7_test), K7_ROOT_A / K7_ROOT_B / K7_ROOT_C (box run roots, default ~/fleet_runs;
#      the simulation uses a scratch root), K7_WITH_C (auto | 0), K7_C_FOLDS (default 0), K7_C_BAR (default 0.8876),
#      K5_MIN_EPOCHS (default 16), K7_LAPTOP_DIR (default k7: where members.json + ckpt/ land).
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K7=$SP/k7
L=${K7_LAPTOP_DIR:-$K7}
B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
SSH="ssh -o BatchMode=yes $B"
SCP="scp -q -o BatchMode=yes"
K7T=${K7T:-k7_test}
RA=${K7_ROOT_A:-'~/fleet_runs'}; RB=${K7_ROOT_B:-'~/fleet_runs'}; RC=${K7_ROOT_C:-'~/fleet_runs'}
WITH_C=${K7_WITH_C:-auto}; C_FOLDS=${K7_C_FOLDS:-0}; C_BAR=${K7_C_BAR:-0.8876}
die(){ echo "K7COLLECT FAIL: $*"; exit 1; }
mkdir -p "$L/ckpt" "$L/ckpt_old" "$L/_real/store"

echo "== 1. sync box tools -> ~/$K7T/k5"
$SSH "mkdir -p ~/$K7T/k5/box ~/$K7T/k5/code ~/$K7T/ckpt ~/$K7T/oof ~/$K7T/runs ~/$K7T/ids ~/$K7T/kernel ~/$K7T/ds" < /dev/null || die "ssh"
$SCP "$K7/box/k5_pick.py" "$K7/box/k7_merge.py" "$K7/box/k5_box_launch.sh" "$K7/box/k2_box_stage.py" "$K7/box/k5_ids.py" $B:$K7T/k5/box/ || die "scp tools"
$SCP "$K7"/code/*.py $B:$K7T/k5/code/ || die "scp code"
$SCP "$K7/k2_check.py" $B:$K7T/k5/ || die "scp k2_check"

echo "== 2. box pick: set A (s1043 TTR224), set B (s1044 TTR224, Kaggle receipts required), set C (s1044 TTR288 f$C_FOLDS)"
PK="CUDA_VISIBLE_DEVICES= K5_MIN_EPOCHS=${K5_MIN_EPOCHS:-16} nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/$K7T/k5/box/k5_pick.py --copy --dest ~/$K7T"
$SSH "rm -f ~/$K7T/members*.json" < /dev/null
$SSH "$PK --root $RA --ttr_size 224 && mv ~/$K7T/members.json ~/$K7T/members_A.json" < /dev/null; rcA=$?
REC="--kaggle_receipt $RB/kaggle_s1044_f01_run_receipt.json --kaggle_receipt $RB/kaggle_s1044_f23_run_receipt.json --kaggle_receipt $RB/kaggle_s1044_f4_run_receipt.json"
$SSH "$PK --root $RB --seed 1044 --suffix ttr_mix50 --ttr_size 224 --require_receipt $REC; rc=\$?; mv ~/$K7T/members.json ~/$K7T/members_B.json; exit \$rc" < /dev/null; rcB=$?
rcC=skipped
if [ "$WITH_C" != "0" ]; then
  $SSH "$PK --root $RC --seed 1044 --suffix ttr288mix50 --folds $C_FOLDS --ttr_size 288 --no_kaggle_receipt; rc=\$?; mv ~/$K7T/members.json ~/$K7T/members_C.json; exit \$rc" < /dev/null; rcC=$?
fi
echo "picker rc: set A $rcA, set B $rcB (3 = no run final yet), set C $rcC (3 = not final; 1 = check failed -> C excluded)"
[ $rcA -eq 3 ] && { echo "K7COLLECT: set A not final?"; exit 3; }
[ $rcA -ne 0 ] && die "set A (s1043) picker rc $rcA: read the FAIL lines above"
[ $rcB -eq 1 ] && die "set B (s1044) picker check FAILED: read the FAIL lines above (do not ship)"
[ $rcB -ne 0 ] && [ $rcB -ne 3 ] && die "set B picker rc $rcB"
$SSH "~/miniconda3/envs/kmimic/bin/python ~/$K7T/k5/box/k7_merge.py --dest ~/$K7T --bar $C_BAR --with_c $WITH_C" < /dev/null || die "merge members.json"
$SCP $B:$K7T/members.json "$L/members_box.json" || die "fetch members.json"

echo "== 3. fetch + verify checkpoints"
$PY - "$L/members_box.json" > "$L/.k7_want.txt" <<'EOF' || die "parse members_box.json"
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
    $SCP $B:$K7T/ckpt/$f "$L/ckpt/" || die "scp $f"
    have=$(sha256sum "$L/ckpt/$f" | cut -d' ' -f1)
  fi
  [ "$have" = "$s" ] || die "$f laptop sha256 $have != box source $s"
  echo "  OK $f sha256 ${s:0:16}"
done < "$L/.k7_want.txt"
for p in "$L"/ckpt/f*_s104[34]_*_best.pt; do
  [ -e "$p" ] || continue
  b=$(basename "$p")
  grep -q "^$b " "$L/.k7_want.txt" || { echo "  stale $b -> ckpt_old/"; mv "$p" "$L/ckpt_old/$b.$(date +%H%M%S)"; }
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
n = {k: sum(r['set'] == k for r in m['members']) for k in 'ABC'}
fin = n['A'] == 5 and n['B'] == 5
print(f'K7COLLECT OK {len(m["members"])} member(s): A {n["A"]}/5, B {n["B"]}/5, C {n["C"]}' +
      ('  (FINAL A+B)' if fin else '  (INTERIM: A or B not complete)'))
EOF
exit 0
