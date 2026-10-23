#!/bin/bash
# k4_collect.sh (laptop, Git Bash): collect the K4 member set (5-fold TTR mix50, f{0..4}_s1043_ttr_mix50) from the box.
#   bash "$SP/k4/k4_collect.sh"
# 1. syncs the K4 box tools to ~/k4_test/k4 (k4_pick.py, k4_box_launch.sh, k2_box_stage.py, k2_check.py)
# 2. box: k4_pick.py --copy (CPU, CUDA hidden, nice 19): for every run with a finished closing line, ONE tag (higher
#    own best val cv, tie -> ema), checks, copy to ~/k4_test/ckpt + OOF to ~/k4_test/oof, ~/k4_test/members.json.
#    Unfinished runs are SKIPPED and listed. A failed check on a finished run stops here (exit 1).
# 3. laptop: scp each picked checkpoint to k4/ckpt/ (skipped when the same bytes are already there), sha256 of the
#    laptop copy == the box source sha256 (else exit 1); stale k4/ckpt/f*_s1043_*_best.pt not in the set -> k4/ckpt_old/
# 4. parity npz of every member's decode_probe: k4/_real/store/<sid>.npz == ~/fleet_store/<sid>.npz (sha256; fetched
#    when missing or different)
# 5. writes k4/members.json (fold, tag, epoch, cv, sha256, file, + laptop path/verification) and prints the set.
# Exit: 0 = members.json written (>= 1 member), 1 = a check failed (do not build), 3 = no run finished yet.
# Read-only on ~/fleet_runs and ~/fleet_store; writes only ~/k4_test on the box and k4/ on the laptop. No GPU.
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K4=$SP/k4
B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
SSH="ssh -o BatchMode=yes $B"
SCP="scp -q -o BatchMode=yes"
die(){ echo "K4COLLECT FAIL: $*"; exit 1; }

echo "== 1. sync box tools -> ~/k4_test/k4"
$SSH 'mkdir -p ~/k4_test/k4/box ~/k4_test/ckpt ~/k4_test/oof ~/k4_test/runs ~/k4_test/ids ~/k4_test/kernel ~/k4_test/ds' || die "ssh"
$SCP "$K4/box/k4_pick.py" "$K4/box/k4_box_launch.sh" "$K4/box/k2_box_stage.py" "$K4/box/k4_ids.py" $B:k4_test/k4/box/ || die "scp tools"
$SCP "$K4/k2_check.py" $B:k4_test/k4/ || die "scp k2_check"

echo "== 2. box pick (+ copy into ~/k4_test)"
$SSH 'CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k4_test/k4/box/k4_pick.py --copy'
rc=$?
[ $rc -eq 3 ] && { echo "K4COLLECT: no run finished yet (nothing to collect)"; exit 3; }
[ $rc -ne 0 ] && die "k4_pick.py exit $rc (a check failed on a finished run: read the FAIL line above, do not build)"
$SCP $B:k4_test/members.json "$K4/members_box.json" || die "fetch members.json"

echo "== 3. fetch + verify checkpoints"
mkdir -p "$K4/ckpt" "$K4/ckpt_old"
# one line per member: file sha256
$PY - "$K4/members_box.json" > "$K4/.k4_want.txt" <<'EOF' || die "parse members_box.json"
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
  [ -f "$K4/ckpt/$f" ] && have=$(sha256sum "$K4/ckpt/$f" | cut -d' ' -f1)
  if [ "$have" != "$s" ]; then
    [ -n "$have" ] && { echo "  $f on the laptop has other bytes ($have): moving it to ckpt_old/"; mv "$K4/ckpt/$f" "$K4/ckpt_old/$f.$(date +%H%M%S)"; }
    echo "  fetching $f"
    $SCP $B:k4_test/ckpt/$f "$K4/ckpt/" || die "scp $f"
    have=$(sha256sum "$K4/ckpt/$f" | cut -d' ' -f1)
  fi
  [ "$have" = "$s" ] || die "$f laptop sha256 $have != box source $s"
  echo "  OK $f sha256 ${s:0:16}"
done < "$K4/.k4_want.txt"
# stale TTR mix50 files (an older pick of a fold, or a fold that is no longer in the set)
for p in "$K4"/ckpt/f*_s1043_*_best.pt; do
  [ -e "$p" ] || continue
  b=$(basename "$p")
  grep -q "^$b " "$K4/.k4_want.txt" || { echo "  stale $b -> ckpt_old/"; mv "$p" "$K4/ckpt_old/$b.$(date +%H%M%S)"; }
done

echo "== 4. parity npz (decode_probe studies)"
mkdir -p "$K4/_real/store"
SIDS=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); print(' '.join(sorted({s for r in m['members'] for s in r['decode_probe']})))" "$K4/members_box.json" | tr -d '[:cntrl:]') || die "probe list"
BOXSHA=$($SSH "cd ~/fleet_store && sha256sum $(for s in $SIDS; do printf '%s.npz ' "$s"; done)") || die "box npz sha"
for s in $SIDS; do
  want=$(echo "$BOXSHA" | grep " $s.npz$" | cut -d' ' -f1)
  have=""; [ -f "$K4/_real/store/$s.npz" ] && have=$(sha256sum "$K4/_real/store/$s.npz" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then echo "  fetching $s.npz"; $SCP $B:fleet_store/$s.npz "$K4/_real/store/" || die "scp $s.npz"
    have=$(sha256sum "$K4/_real/store/$s.npz" | cut -d' ' -f1); fi
  [ -n "$want" ] && [ "$have" = "$want" ] || die "parity npz $s: laptop $have != box $want"
  echo "  OK $s.npz ${want:0:16}"
done

echo "== 5. k4/members.json"
$PY - "$K4/members_box.json" "$K4/ckpt" "$K4/members.json" <<'EOF' || die "write members.json"
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
    print(f'  fold {r["fold"]} {r["tag"]} epoch {r["epoch"]} cv {r["cv"]:.6f} (other {r["other_tag_cv"]:.6f}) {r["file"]} sha256 {r["sha256"][:16]}')
for s in m['skipped']:
    print(f'  SKIPPED fold {s["fold"]}: {s["reason"][:110]}')
n = len(m['members'])
print(f'K4COLLECT OK {n} member(s) folds {[r["fold"] for r in m["members"]]}' + ('' if n == 5 else f'  (INTERIM: {5 - n} fold(s) not finished)'))
EOF
exit 0
