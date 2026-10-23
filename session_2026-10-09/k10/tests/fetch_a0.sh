#!/bin/bash
# fetch_a0.sh (laptop): copy the box A0 logs + reports of one area into k10/tests/box/<area tag>/ and print one line per job.
#   bash "$SP/k10/tests/fetch_a0.sh" k10_test/a0sim       (or k10_test/a0 for the real build)
set -uo pipefail
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
A=${1:?area}
D="$SP/k10/tests/box/$(basename "$A")"
B=training@192.168.10.30
mkdir -p "$D"
ssh -o BatchMode=yes $B "cd ~/$A/runs && tar -cf - *.log */report.json */k9n_check.json */fn_check.json 2>/dev/null" < /dev/null | tar -C "$D" -xf - 2>/dev/null
cat "$D"/q1_*.log "$D"/q2_*.log 2>/dev/null
for f in $(ls "$D"/*.log | grep -v '/q[12]_'); do
  r=$(grep -hE "^(BOXSTAGE|K9NBOX|KFNBOX) (PASS|FAIL)" "$f" | tail -n 1 | cut -c1-24)
  w=$(grep -hE "RESULT (OK|FLEET STAGE NOT CLEAN|N STAGE NOT CLEAN)" "$f" | tail -n 1 | cut -c1-150)
  echo "$(basename "$f" .log): ${r:-RUNNING/NO VERDICT} | ${w}"
done
