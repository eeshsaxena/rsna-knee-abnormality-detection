#!/bin/bash
# k5_collect.sh (laptop, Git Bash): collect the K5 member set (5-fold TTR mix50, f{0..4}_s1043_ttr_mix50) from the box.
# = k4_collect.sh with ~/k5_test, k5/ and the K5 picker (box/k5_pick.py: Kaggle-filled runs accepted, E from the log,
# OOF-vs-ckpt cv content proof for every run). Usage:
#   bash "$SP/k5/k5_collect.sh"
#   K5_MIN_EPOCHS=12 bash ...           # accept a Kaggle fold with E < 16 (operator's call, Kaggle RUNBOOK sec 5)
#   K5_PICK_ARGS="--extra_root ~/k5_test/kaggle_runs" bash ...   # Kaggle outputs staged outside ~/fleet_runs
# 1. syncs the K5 box tools to ~/k5_test/k5 (box/*.py, box/*.sh, code/, k2_check.py)
# 2. box: k5_pick.py --copy (CPU, CUDA hidden, nice 19): for every run with a finished closing line, ONE tag (higher
#    own best val cv, tie -> ema), checks, copy to ~/k5_test/ckpt + OOF to ~/k5_test/oof, ~/k5_test/members.json.
#    Unfinished runs are SKIPPED and listed. A failed check on a finished run stops here (exit 1).
# 3. laptop: scp each picked checkpoint to k5/ckpt/ (skipped when the same bytes are already there), sha256 of the
#    laptop copy == the box source sha256 (else exit 1); stale k5/ckpt/f*_s1043_*_best.pt not in the set -> k5/ckpt_old/
# 4. parity npz of every member's decode_probe: k5/_real/store/<sid>.npz == ~/fleet_store/<sid>.npz (sha256)
# 5. writes k5/members.json and prints the set (+ source box/kaggle, E, OOF rescored cv).
# Exit: 0 = members.json written (>= 1 member), 1 = a check failed (do not build), 3 = no run finished yet.
# Read-only on ~/fleet_runs and ~/fleet_store; writes only ~/k5_test on the box and k5/ on the laptop. No GPU.
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K5=$SP/k5
B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
SSH="ssh -o BatchMode=yes $B"
SCP="scp -q -o BatchMode=yes"
die(){ echo "K5COLLECT FAIL: $*"; exit 1; }

echo "== 1. sync box tools -> ~/k5_test/k5"
$SSH 'mkdir -p ~/k5_test/k5/box ~/k5_test/k5/code ~/k5_test/ckpt ~/k5_test/oof ~/k5_test/runs ~/k5_test/ids ~/k5_test/kernel ~/k5_test/ds' < /dev/null || die "ssh"
$SCP "$K5/box/k5_pick.py" "$K5/box/k5_box_launch.sh" "$K5/box/k2_box_stage.py" "$K5/box/k5_ids.py" "$K5/box/k5_parity.py" \
  $B:k5_test/k5/box/ || die "scp tools"
$SCP "$K5"/code/*.py $B:k5_test/k5/code/ || die "scp code"
$SCP "$K5/k2_check.py" $B:k5_test/k5/ || die "scp k2_check"

echo "== 2. box pick (+ copy into ~/k5_test)"
$SSH "CUDA_VISIBLE_DEVICES= K5_MIN_EPOCHS=${K5_MIN_EPOCHS:-16} nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k5_test/k5/box/k5_pick.py --copy ${K5_PICK_ARGS:-}" < /dev/null
rc=$?
[ $rc -eq 3 ] && { echo "K5COLLECT: no run finished yet (nothing to collect)"; exit 3; }
[ $rc -ne 0 ] && die "k5_pick.py exit $rc (a check failed on a finished run: read the FAIL line above, do not build)"
$SCP $B:k5_test/members.json "$K5/members_box.json" || die "fetch members.json"

echo "== 3. fetch + verify checkpoints"
mkdir -p "$K5/ckpt" "$K5/ckpt_old"
$PY - "$K5/members_box.json" > "$K5/.k5_want.txt" <<'EOF' || die "parse members_box.json"
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
  [ -f "$K5/ckpt/$f" ] && have=$(sha256sum "$K5/ckpt/$f" | cut -d' ' -f1)
  if [ "$have" != "$s" ]; then
    [ -n "$have" ] && { echo "  $f on the laptop has other bytes ($have): moving it to ckpt_old/"; mv "$K5/ckpt/$f" "$K5/ckpt_old/$f.$(date +%H%M%S)"; }
    echo "  fetching $f"
    $SCP $B:k5_test/ckpt/$f "$K5/ckpt/" || die "scp $f"
    have=$(sha256sum "$K5/ckpt/$f" | cut -d' ' -f1)
  fi
  [ "$have" = "$s" ] || die "$f laptop sha256 $have != box source $s"
  echo "  OK $f sha256 ${s:0:16}"
done < "$K5/.k5_want.txt"
for p in "$K5"/ckpt/f*_s1043_*_best.pt; do
  [ -e "$p" ] || continue
  b=$(basename "$p")
  grep -q "^$b " "$K5/.k5_want.txt" || { echo "  stale $b -> ckpt_old/"; mv "$p" "$K5/ckpt_old/$b.$(date +%H%M%S)"; }
done

echo "== 4. parity npz (decode_probe studies)"
mkdir -p "$K5/_real/store"
SIDS=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); print(' '.join(sorted({s for r in m['members'] for s in r['decode_probe']})))" "$K5/members_box.json" | tr -d '[:cntrl:]') || die "probe list"
BOXSHA=$($SSH "cd ~/fleet_store && sha256sum $(for s in $SIDS; do printf '%s.npz ' "$s"; done)" < /dev/null) || die "box npz sha"
for s in $SIDS; do
  want=$(echo "$BOXSHA" | grep " $s.npz$" | cut -d' ' -f1)
  have=""; [ -f "$K5/_real/store/$s.npz" ] && have=$(sha256sum "$K5/_real/store/$s.npz" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then echo "  fetching $s.npz"; $SCP $B:fleet_store/$s.npz "$K5/_real/store/" || die "scp $s.npz"
    have=$(sha256sum "$K5/_real/store/$s.npz" | cut -d' ' -f1); fi
  [ -n "$want" ] && [ "$have" = "$want" ] || die "parity npz $s: laptop $have != box $want"
  echo "  OK $s.npz ${want:0:16}"
done

echo "== 5. k5/members.json"
$PY - "$K5/members_box.json" "$K5/ckpt" "$K5/members.json" <<'EOF' || die "write members.json"
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
    print(f'  fold {r["fold"]} {r["tag"]} epoch {r["epoch"]}/{r.get("epochs")} cv {r["cv"]:.6f} (other {r["other_tag_cv"]:.6f}) '
          f'{r["file"]} sha256 {r["sha256"][:16]} source {r.get("source")} OOF rescored {r.get("oof_rescored_cv")}')
for s in m['skipped']:
    print(f'  SKIPPED fold {s["fold"]}: {s["reason"][:110]}')
n = len(m['members'])
print(f'K5COLLECT OK {n} member(s) folds {[r["fold"] for r in m["members"]]}' + ('' if n == 5 else f'  (INTERIM: {5 - n} fold(s) not finished)'))
EOF
exit 0
