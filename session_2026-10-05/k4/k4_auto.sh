#!/bin/bash
# k4_auto.sh (laptop, Git Bash): RUNBOOK steps 1-3 in one command, for whatever folds are final right now.
#   bash "$SP/k4/k4_auto.sh"            # collect -> build (if the set changed) -> stage on the box -> A0 x3 -> wait + verdict
# Build dir: k4/build (5 folds) or k4/build_i<N> (interim). An existing build whose K4_BUILD.json member list equals
# members.json is reused (not rebuilt). CPU only on the box (A0 under nice 19); no Kaggle action.
# Last line: "K4AUTO READY <build dir> members <list>" (all three A0 BOXSTAGE PASS) or "K4AUTO FAIL ...".
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
K4=$SP/k4; B=training@192.168.10.30
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
SSH="ssh -o BatchMode=yes $B"
die(){ echo "K4AUTO FAIL: $*"; exit 1; }

bash "$K4/k4_collect.sh"; rc=$?
[ $rc -eq 3 ] && { echo "K4AUTO: no fold final yet"; exit 3; }
[ $rc -ne 0 ] && die "k4_collect.sh exit $rc"
read -r N BD SAME <<< "$($PY - "$K4" <<'EOF' | tr -d '[:cntrl:]'
import json, os, sys
k4 = sys.argv[1].replace('\\', '/')
m = json.load(open(f'{k4}/members.json'))
names = [r['file'] for r in sorted(m['members'], key=lambda r: r['fold'])]
n = len(names)
bd = f'{k4}/build' + ('' if n == 5 else f'_i{n}')
same = 0
p = f'{bd}/K4_BUILD.json'
if os.path.isfile(p):
    b = json.load(open(p))
    same = int(b.get('members') == names and not b.get('with_m1'))
print(n, bd, same)
EOF
)"
[ -n "${BD:-}" ] || die "could not read members.json"
if [ "$SAME" = "1" ]; then
  echo "== build $BD is current for this member set (not rebuilt)"
else
  echo "== build $BD"
  OW=""; [ -d "$BD" ] && OW="--overwrite"
  $PY -W ignore "$K4/build_k4.py" --out "$BD" $OW > "$BD.log" 2>&1 || { tail -n 5 "$BD.log"; die "build_k4.py (log $BD.log)"; }
  grep -E "^member f|K4 BUILD OK" "$BD.log" | cut -c1-200
fi
bash "$K4/k4_stage_box.sh" "$BD" | tail -n 3 || die "stage"
T0=$(date +%s)
LAUNCH=$($SSH 'bash ~/k4_test/k4/box/k4_box_launch.sh all' < /dev/null) || die "A0 launch"
echo "$LAUNCH"
LOGS=$(echo "$LAUNCH" | sed -n 's/.* log \(runs\/.*\.log\)$/\1/p' | tr '\n' ' ')
echo "== waiting for A0 ($LOGS)"
for i in $(seq 1 120); do
  n=$($SSH "cd ~/k4_test && grep -l BOXSTAGE $LOGS 2>/dev/null | wc -l" | tr -d '[:cntrl:]')
  [ "$n" = "$(echo $LOGS | wc -w)" ] && break
  sleep 15
done
RES=$($SSH "cd ~/k4_test && for f in $LOGS; do echo \"== \$f\"; grep -E '^FAIL|BOXSTAGE' \$f | cut -c1-160; done")
echo "$RES"
echo "A0 wall $(( ($(date +%s) - T0) / 60 )) min"
NP=$(echo "$RES" | grep -c "^BOXSTAGE PASS")
[ "$NP" = "$(echo $LOGS | wc -w)" ] && ! echo "$RES" | grep -q "^FAIL" || die "A0 not all PASS (see above)"
MEM=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$BD/K4_BUILD.json" | tr -d '[:cntrl:]')
echo "K4AUTO READY $BD members $MEM"
