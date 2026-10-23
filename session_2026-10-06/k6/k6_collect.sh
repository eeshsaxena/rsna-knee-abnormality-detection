#!/bin/bash
# k6_collect.sh (laptop, Git Bash; 2026-10-06, Claude). = k5_collect.sh for the K6 member set:
#   set A = TTR mix50 f{0..4}_s1043_ttr_mix50 (folds 3-4 Kaggle-filled, run receipt ~/fleet_runs/kaggle_f34_run_receipt.json)
#   set B = TTR v4    f{0..4}_s1042_ttr        (pure llm_labels_v4_blend targets, all box-trained: --no_kaggle_receipt)
# Same picker checks per run (k6/box/k5_pick.py = k5 picker + --train_labels + --no_kaggle_receipt). Box area ~/k6_test.
#   bash "$SP/k6/k6_collect.sh"            # exit 0 = k6/members.json written; 1 = a check failed; 3 = nothing final
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K6=$SP/k6
B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
SSH="ssh -o BatchMode=yes $B"
SCP="scp -q -o BatchMode=yes"
die(){ echo "K6COLLECT FAIL: $*"; exit 1; }

echo "== 1. sync box tools -> ~/k6_test/k5"
$SSH 'mkdir -p ~/k6_test/k5/box ~/k6_test/k5/code ~/k6_test/ckpt ~/k6_test/oof ~/k6_test/runs ~/k6_test/ids ~/k6_test/kernel ~/k6_test/ds' < /dev/null || die "ssh"
$SCP "$K6/box/k5_pick.py" "$K6/box/k5_box_launch.sh" "$K6/box/k2_box_stage.py" "$K6/box/k5_ids.py" $B:k6_test/k5/box/ || die "scp tools"
$SCP "$K6"/code/*.py $B:k6_test/k5/code/ || die "scp code"
$SCP "$K6/k2_check.py" $B:k6_test/k5/ || die "scp k2_check"

echo "== 2. box pick, set A (mix50 s1043) and set B (v4 s1042)"
PK="CUDA_VISIBLE_DEVICES= K5_MIN_EPOCHS=${K5_MIN_EPOCHS:-16} nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k6_test/k5/box/k5_pick.py --copy --dest ~/k6_test"
$SSH "rm -f ~/k6_test/members*.json; $PK && mv ~/k6_test/members.json ~/k6_test/members_mix50.json" < /dev/null; rcA=$?
$SSH "$PK --seed 1042 --suffix ttr --train_labels llm_labels_v4_blend.csv --no_kaggle_receipt && mv ~/k6_test/members.json ~/k6_test/members_v4.json" < /dev/null; rcB=$?
[ $rcA -eq 1 ] || [ $rcB -eq 1 ] && die "a picker check failed (set A rc $rcA, set B rc $rcB): read the FAIL lines above"
[ $rcA -ne 0 ] && die "set A (mix50) not collected (rc $rcA)"
[ $rcB -eq 3 ] && { echo "K6COLLECT: no v4 run finished yet"; exit 3; }
[ $rcB -ne 0 ] && die "set B (v4) picker rc $rcB"
$SSH '~/miniconda3/envs/kmimic/bin/python - <<EOF
import json, time
a = json.load(open("/home/training/k6_test/members_mix50.json")); b = json.load(open("/home/training/k6_test/members_v4.json"))
assert not a["failed"] and not b["failed"], (a["failed"], b["failed"])
m = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), set="K6 = f{0..4}_s1043_ttr_mix50 + f{0..4}_s1042_ttr",
         members=a["members"] + b["members"], skipped=[dict(s, seed=1043) for s in a["skipped"]] + [dict(s, seed=1042) for s in b["skipped"]],
         failed=[], copied=True)
json.dump(m, open("/home/training/k6_test/members.json", "w"), indent=1)
print("merged", len(m["members"]), "members")
EOF' < /dev/null || die "merge members.json"
$SCP $B:k6_test/members.json "$K6/members_box.json" || die "fetch members.json"

echo "== 3. fetch + verify checkpoints"
$PY - "$K6/members_box.json" > "$K6/.k6_want.txt" <<'EOF' || die "parse members_box.json"
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
  [ -f "$K6/ckpt/$f" ] && have=$(sha256sum "$K6/ckpt/$f" | cut -d' ' -f1)
  if [ "$have" != "$s" ]; then
    [ -n "$have" ] && { echo "  $f on the laptop has other bytes ($have): moving it to ckpt_old/"; mv "$K6/ckpt/$f" "$K6/ckpt_old/$f.$(date +%H%M%S)"; }
    echo "  fetching $f"
    $SCP $B:k6_test/ckpt/$f "$K6/ckpt/" || die "scp $f"
    have=$(sha256sum "$K6/ckpt/$f" | cut -d' ' -f1)
  fi
  [ "$have" = "$s" ] || die "$f laptop sha256 $have != box source $s"
  echo "  OK $f sha256 ${s:0:16}"
done < "$K6/.k6_want.txt"
for p in "$K6"/ckpt/f*_s104[23]_*_best.pt; do
  [ -e "$p" ] || continue
  b=$(basename "$p")
  grep -q "^$b " "$K6/.k6_want.txt" || { echo "  stale $b -> ckpt_old/"; mv "$p" "$K6/ckpt_old/$b.$(date +%H%M%S)"; }
done

echo "== 4. parity npz (decode_probe studies)"
mkdir -p "$K6/_real/store"
SIDS=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); print(' '.join(sorted({s for r in m['members'] for s in r['decode_probe']})))" "$K6/members_box.json" | tr -d '[:cntrl:]') || die "probe list"
BOXSHA=$($SSH "cd ~/fleet_store && sha256sum $(for s in $SIDS; do printf '%s.npz ' "$s"; done)" < /dev/null) || die "box npz sha"
for s in $SIDS; do
  want=$(echo "$BOXSHA" | grep " $s.npz$" | cut -d' ' -f1)
  have=""; [ -f "$K6/_real/store/$s.npz" ] && have=$(sha256sum "$K6/_real/store/$s.npz" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then echo "  fetching $s.npz"; $SCP $B:fleet_store/$s.npz "$K6/_real/store/" || die "scp $s.npz"
    have=$(sha256sum "$K6/_real/store/$s.npz" | cut -d' ' -f1); fi
  [ -n "$want" ] && [ "$have" = "$want" ] || die "parity npz $s: laptop $have != box $want"
  echo "  OK $s.npz ${want:0:16}"
done

echo "== 5. k6/members.json"
$PY - "$K6/members_box.json" "$K6/ckpt" "$K6/members.json" <<'EOF' || die "write members.json"
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
for r in sorted(m['members'], key=lambda r: (r['seed'], r['fold'])):
    print(f'  s{r["seed"]} fold {r["fold"]} {r["tag"]} epoch {r["epoch"]}/{r.get("epochs")} cv {r["cv"]:.6f} '
          f'{r["file"]} sha256 {r["sha256"][:16]} source {r.get("source")} train {r.get("train_labels")} OOF {r.get("oof_rescored_cv")}')
for s in m['skipped']:
    print(f'  SKIPPED s{s.get("seed")} fold {s["fold"]}: {s["reason"][:100]}')
n = len(m['members'])
print(f'K6COLLECT OK {n} member(s)' + ('' if n == 10 else f'  (INTERIM: {10 - n} run(s) not finished)'))
EOF
exit 0
