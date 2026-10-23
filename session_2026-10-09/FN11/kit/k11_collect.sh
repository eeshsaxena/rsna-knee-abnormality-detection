#!/bin/bash
# k11_collect.sh (laptop, Git Bash; 2026-10-09, Claude; from k10_collect.sh). ONE collect for the FN11 kits:
#   C + D + E = the FINAL K10 members (box ~/k10_test/members.json, read only; laptop bytes HARD-LINKED from SP/k10/ckpt
#               after a sha256 check, so nothing is re-fetched and the k10 kit is never written)
#   E2        = TTR288 f{0..4}_s1046_ttr288frac75, frac75, 21 ep, Modal L40S (placed by ~/modal_e2/place_fold.sh,
#               receipts ~/fleet_runs/modal_e2_<run>_receipt.json): k11/box/k5_pick.py (K10 checks + K11-MREC), then
#               k11/box/k11_merge.py (PREREG_FN11 sec 2: bar E2_F >= E_F - .005, failed / low folds EXCLUDED)
#   bash "$SP/k11/k11_collect.sh"
# Exit 0 = k11/members.json written (FINAL; or INTERIM with K11_INTERIM=1, marked do_not_submit); 1 = a check failed;
#      3 = E2 not complete and K11_INTERIM unset.
# Env: K11_INTERIM=1 (TEST builds), K11_E2_SHIP=4of5 (PREREG sec 4: only after the 10-10 12:00 IST cutoff or a failed f4).
# Box: CPU only (CUDA hidden, nice 19, taskset 34-39); writes only ~/k11_test; ~/fleet_runs and ~/k10_test read only.
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K11=$SP/k11; K10=$SP/k10
B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
SSH="ssh -o BatchMode=yes $B"
SCP="scp -q -o BatchMode=yes"
T=k11_test
INTERIM=${K11_INTERIM:-0}; SHIP=${K11_E2_SHIP:-}
die(){ echo "K11COLLECT FAIL: $*"; exit 1; }
mkdir -p "$K11/ckpt" "$K11/ckpt_old" "$K11/_real/store"

echo "== 0. generated tools current"
$PY -B "$K11/make_k11_tools.py" --check | tail -n 1 | grep -q "MAKE_K11_TOOLS OK" || die "make_k11_tools --check"

echo "== 1. sync box tools -> ~/$T/k11"
$SSH "mkdir -p ~/$T/k11/box ~/$T/ckpt ~/$T/oof ~/$T/gold" < /dev/null || die "ssh"
$SCP "$K11/box/k5_pick.py" "$K11/box/k11_merge.py" "$K11/box/k11_gate.py" "$K11/box/k11_gate_bg.sh" $B:$T/k11/box/ || die "scp tools"

echo "== 2. box pick E2 (s1046 TTR288 frac75, Modal receipts f0-f4) + merge"
PK="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 taskset -c 34-39 nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/$T/k11/box/k5_pick.py --copy --dest ~/$T --tag raw"
$SSH "rm -f ~/$T/members.json ~/$T/members_E2.json; K5_MIN_EPOCHS=16 $PK --root ~/fleet_runs --seed 1046 --suffix ttr288frac75 --folds 0,1,2,3,4 --ttr_size 288 --train_labels labels_frac75_logit.csv --key_tag t288f75 --no_kaggle_receipt --modal_receipt_folds 0,1,2,3,4; rc=\$?; mv ~/$T/members.json ~/$T/members_E2.json 2>/dev/null; exit \$rc" < /dev/null; rcE2=$?
echo "picker rc: set E2 $rcE2 (0 = ok; 3 = no fold final yet; 1 = a fold FAILED a check: it is EXCLUDED by the merge, PREREG sec 2)"
[ $rcE2 -ne 0 ] && [ $rcE2 -ne 1 ] && [ $rcE2 -ne 3 ] && die "E2 picker rc $rcE2"
MA="--dest ~/$T"
[ "$INTERIM" = "1" ] && MA="$MA --allow_interim"
[ -n "$SHIP" ] && MA="$MA --e2_ship $SHIP"
$SSH "taskset -c 34-39 nice -n 19 ~/miniconda3/envs/kmimic/bin/python ~/$T/k11/box/k11_merge.py $MA" < /dev/null; rcM=$?
[ $rcM -eq 3 ] && { echo "K11COLLECT NOT FINAL: E2 is not complete (see above). K11_INTERIM=1 for a TEST build only."; exit 3; }
[ $rcM -ne 0 ] && die "merge rc $rcM (read the K11MERGE lines above)"
$SCP $B:$T/members.json "$K11/members_box.json" || die "fetch members.json"

echo "== 3. checkpoints: C / D / E hard-linked from SP/k10/ckpt (sha256 == members), E2 fetched from the box"
$PY - "$K11/members_box.json" > "$K11/.k11_want.txt" <<'EOF' || die "parse members_box.json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m['members'] and not m['failed'] and m.get('k11'), m.get('failed')
for r in m['members']:
    print(r['set'], r['file'], r['sha256'])
EOF
while read -r st f s; do
  st=$(printf '%s' "$st" | tr -d '[:cntrl:]'); f=$(printf '%s' "$f" | tr -d '[:cntrl:]'); s=$(printf '%s' "$s" | tr -d '[:cntrl:]')
  [ -z "$f" ] && continue
  have=""; [ -f "$K11/ckpt/$f" ] && have=$(sha256sum "$K11/ckpt/$f" | cut -d' ' -f1)
  if [ "$have" != "$s" ]; then
    [ -n "$have" ] && { echo "  $f on the laptop has other bytes ($have): moving it to ckpt_old/"; mv "$K11/ckpt/$f" "$K11/ckpt_old/$f.$(date +%H%M%S)"; }
    case $st in
      C|D|E) k10=""; [ -f "$K10/ckpt/$f" ] && k10=$(sha256sum "$K10/ckpt/$f" | cut -d' ' -f1)
             [ "$k10" = "$s" ] || die "$f: SP/k10/ckpt sha256 $k10 != the K10 pick $s"
             ln "$K10/ckpt/$f" "$K11/ckpt/$f" || die "hard link $f"; echo "  linked $f from SP/k10/ckpt (K10 pick, same sha256)";;
      *) echo "  fetching $f"; $SCP $B:$T/ckpt/$f "$K11/ckpt/" || die "scp $f";;
    esac
    have=$(sha256sum "$K11/ckpt/$f" | cut -d' ' -f1)
  fi
  [ "$have" = "$s" ] || die "$f laptop sha256 $have != box source $s"
  echo "  OK $st $f sha256 ${s:0:16}"
done < "$K11/.k11_want.txt"
for p in "$K11"/ckpt/f*_s104[3456]_*_best.pt; do
  [ -e "$p" ] || continue
  b=$(basename "$p")
  grep -q " $b " "$K11/.k11_want.txt" || { echo "  stale $b -> ckpt_old/"; mv "$p" "$K11/ckpt_old/$b.$(date +%H%M%S)"; }
done

echo "== 4. parity npz (decode_probe studies)"
SIDS=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); print(' '.join(sorted({s for r in m['members'] for s in r['decode_probe']})))" "$K11/members_box.json" | tr -d '[:cntrl:]') || die "probe list"
BOXSHA=$($SSH "cd ~/fleet_store && sha256sum $(for s in $SIDS; do printf '%s.npz ' "$s"; done)" < /dev/null) || die "box npz sha"
for s in $SIDS; do
  want=$(echo "$BOXSHA" | grep " $s.npz$" | cut -d' ' -f1)
  have=""; [ -f "$K11/_real/store/$s.npz" ] && have=$(sha256sum "$K11/_real/store/$s.npz" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then echo "  fetching $s.npz"; $SCP $B:fleet_store/$s.npz "$K11/_real/store/" || die "scp $s.npz"
    have=$(sha256sum "$K11/_real/store/$s.npz" | cut -d' ' -f1); fi
  [ -n "$want" ] && [ "$have" = "$want" ] || die "parity npz $s: laptop $have != box $want"
  echo "  OK $s.npz ${want:0:16}"
done

echo "== 5. members.json"
$PY - "$K11/members_box.json" "$K11/ckpt" "$K11/members.json" "$K10/members.json" <<'EOF' || die "write members.json"
import hashlib, json, sys, time
m = json.load(open(sys.argv[1])); ck = sys.argv[2]
k10 = {r['file']: r['sha256'] for r in json.load(open(sys.argv[4]))['members']}
for r in m['members']:
    p = f'{ck}/{r["file"]}'
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    assert h == r['sha256'], (r['file'], h, r['sha256'])
    if r['set'] in ('C', 'D', 'E'):
        assert k10.get(r['file']) == h, f'{r["file"]}: != the laptop K10 members.json'
    r['laptop_path'] = p.replace('\\', '/'); r['laptop_sha256_ok'] = True
assert sorted(k10) == sorted(r['file'] for r in m['members'] if r['set'] in ('C', 'D', 'E')), 'C + D + E != the K10 members'
m['collected'] = time.strftime('%Y-%m-%d %H:%M:%S')
json.dump(m, open(sys.argv[3], 'w'), indent=1)
print(f'members ({len(m["members"])}):')
for r in m['members']:
    print(f'  set {r["set"]} s{r["seed"]} fold {r["fold"]} ttr{r["ttr_size"]} {r["train_labels"]} {r["tag"]} epoch {r["epoch"]}/{r.get("epochs")} '
          f'cv {r["cv"]:.6f} {r["file"]} sha256 {r["sha256"][:16]} source {r.get("source")} OOF {r.get("oof_rescored_cv")}')
for s in m['skipped']:
    print(f'  NOT FINISHED set {s["set"]} f{s["fold"]}_s{s["seed"]}_{s["suffix"]}: {str(s["reason"])[:100]}')
n = {k: sum(r['set'] == k for r in m['members']) for k in ('C', 'D', 'E', 'E2', 'D2')}
tag = (f'INTERIM: TEST ONLY, do_not_submit, NEVER SUBMIT' if m.get('do_not_submit')
       else ('FINAL, e2_ship 4of5' if m.get('e2_ship') == '4of5' else 'FINAL'))
print(f'K11COLLECT OK {len(m["members"])} member(s): ' + ', '.join(f'{k} {v}/5' for k, v in n.items() if v) + f'  ({tag})')
EOF
exit 0
