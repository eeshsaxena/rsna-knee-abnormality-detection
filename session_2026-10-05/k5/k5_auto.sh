#!/bin/bash
# k5_auto.sh (laptop, Git Bash): K5 RUNBOOK steps 1-3 in one command, for whatever folds are final right now.
#   bash "$SP/k5/k5_auto.sh"              # collect -> build (if the set changed) -> stage on the box -> A0 x3 -> verdict
#   K5_WITH_M1=1 bash "$SP/k5/k5_auto.sh" # the same with M1 (F1 mix50, pass A only) in every kernel
# Build dir: k5/build (5 folds) or k5/build_i<N> (interim), + _m1. An existing build whose K5_BUILD.json member list and
# TTA map equal the current set is reused (not rebuilt). CPU only on the box (A0 under nice 19); no Kaggle action.
# K5_MIN_EPOCHS / K5_PICK_ARGS pass through to k5_collect.sh.
# Last line: "K5AUTO READY <build dir> members <list> tta <map>" (all three A0 BOXSTAGE PASS) or "K5AUTO FAIL ...".
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K5=$SP/k5; B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
SSH="ssh -o BatchMode=yes $B"
die(){ echo "K5AUTO FAIL: $*"; exit 1; }
M1FLAG=""; [ "${K5_WITH_M1:-0}" = "1" ] && M1FLAG="--with_m1"

bash "$K5/k5_collect.sh"; rc=$?
[ $rc -eq 3 ] && { echo "K5AUTO: no fold final yet"; exit 3; }
[ $rc -ne 0 ] && die "k5_collect.sh exit $rc"
read -r N BD SAME <<< "$($PY - "$K5" "${K5_WITH_M1:-0}" <<'EOF' | tr -d '[:cntrl:]'
import json, os, sys
k5, m1 = sys.argv[1].replace('\\', '/'), sys.argv[2] == '1'
m = json.load(open(f'{k5}/members.json'))
names = [r['file'] for r in sorted(m['members'], key=lambda r: r['fold'])]
n = len(names)
bd = f'{k5}/build' + ('' if n == 5 else f'_i{n}') + ('_m1' if m1 else '')
M1 = 'f0_s42_raw_cnxt320x24_cv8810_best.pt'
want = ([M1] if m1 else []) + names
tta = {**({M1: 'A'} if m1 else {}), **{x: 'AB' for x in names}}
same = 0
p = f'{bd}/K5_BUILD.json'
if os.path.isfile(p):
    b = json.load(open(p))
    same = int(b.get('members') == want and b.get('tta') == tta and bool(b.get('with_m1')) == m1)
print(n, bd, same)
EOF
)"
[ -n "${BD:-}" ] || die "could not read members.json"
if [ "$SAME" = "1" ]; then
  echo "== build $BD is current for this member set + TTA map (not rebuilt)"
else
  echo "== build $BD"
  OW=""; [ -d "$BD" ] && OW="--overwrite"
  $PY -W ignore "$K5/build_k5.py" --out "$BD" $M1FLAG $OW > "$BD.log" 2>&1 || { tail -n 5 "$BD.log"; die "build_k5.py (log $BD.log)"; }
  grep -E "^member f|K5 BUILD OK|^expect_tta" "$BD.log" | cut -c1-220
fi
bash "$K5/k5_stage_box.sh" "$BD" | tail -n 4 || die "stage"
T0=$(date +%s)
LAUNCH=$($SSH 'bash ~/k5_test/k5/box/k5_box_launch.sh all' < /dev/null) || die "A0 launch"
echo "$LAUNCH"
LOGS=$(echo "$LAUNCH" | sed -n 's/.* log \(runs\/.*\.log\)$/\1/p' | tr '\n' ' ')
echo "== waiting for A0 ($LOGS)"
for i in $(seq 1 160); do
  n=$($SSH "cd ~/k5_test && grep -l BOXSTAGE $LOGS 2>/dev/null | wc -l" < /dev/null | tr -d '[:cntrl:]')
  [ "$n" = "$(echo $LOGS | wc -w)" ] && break
  sleep 15
done
RES=$($SSH "cd ~/k5_test && for f in $LOGS; do echo \"== \$f\"; grep -E '^FAIL|BOXSTAGE|TTA modes|K5 ' \$f | cut -c1-200; done" < /dev/null)
echo "$RES"
echo "A0 wall $(( ($(date +%s) - T0) / 60 )) min"
NP=$(echo "$RES" | grep -c "^BOXSTAGE PASS")
[ "$NP" = "$(echo $LOGS | wc -w)" ] && ! echo "$RES" | grep -q "^FAIL" || die "A0 not all PASS (see above)"
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BD/K5_BUILD.json" | tr -d '[:cntrl:]')
TTA=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['expect_tta'])" "$BD/K5_BUILD.json" | tr -d '[:cntrl:]')
echo "K5AUTO READY $BD members $MEM tta $TTA"
