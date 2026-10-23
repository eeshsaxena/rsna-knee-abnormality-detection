#!/bin/bash
# sat_run.sh SLOT...   Saturday 10-10 driver (PREREG_10-10 Amendment 3b). Per slot: DO_NOT check, push, poll until the
# commit finishes, fetch the output to out_v<V>, run both gates. NEVER submits (main submits by hand after reading the gates).
# Slots: 1 FN11-N70 | 2 FN11 | 3 FN11-N80 | 4 FN10-3-N70 | 5 FN10-3-N80.  Example: bash sat_run.sh 1 2
set -u
SP=C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
LOG=C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-10/sat_run.log
BW=$SP/k10/build_w; B11=$SP/k11/build_w11
mem() { $PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['members']))" "$1/K5_BUILD.json" | tr -d '[:cntrl:]'; }
MEMW=$(mem $BW); MEM11=$(mem $B11)
MANW=$BW/upload/rsna-knee-cnxt-fleet-k10w/fleet_manifest.json; MAN11=$B11/upload/rsna-knee-cnxt-fleet-k11w/fleet_manifest.json
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cfg() {  # slot -> KIT BUILD SLUG W3 CHECK MEM MAN BWX
  case $1 in
    1) echo "k11 build_fn11_n70 rsna-sub-k11fn-n70 0.30,0.70 fn11_check.py MEM11 MAN11 $B11";;
    2) echo "k11 build_fn11     rsna-sub-k11fn     0.40,0.60 fn11_check.py MEM11 MAN11 $B11";;
    3) echo "k11 build_fn11_n80 rsna-sub-k11fn-n80 0.20,0.80 fn11_check.py MEM11 MAN11 $B11";;
    4) echo "k10 build_fn3_n70  rsna-sub-k10fn3-n70 0.30,0.70 fn10_check.py MEMW MANW $BW";;
    5) echo "k10 build_fn3_n80  rsna-sub-k10fn3-n80 0.20,0.80 fn10_check.py MEMW MANW $BW";;
    *) return 1;;
  esac; }
push() {  # slot -> pushes, records version
  read KIT BUILD SLUG W3 CHECK MV MN BWX <<< "$(cfg $1)"
  D=$SP/$KIT/$BUILD
  if ls $D/DO_NOT_* >/dev/null 2>&1; then say "slot $1 $SLUG: DO_NOT marker present, SKIP"; return 1; fi
  out=$(kaggle kernels push -p "$D/kernel/$SLUG" 2>&1); say "slot $1 push $SLUG: $(echo "$out" | tr '\n' ' ' | cut -c1-200)"
  V=$(echo "$out" | grep -oE "version [0-9]+" | grep -oE "[0-9]+" | head -1); echo "${V:-?}" > "$D/.sat_version"
}
waitfetch() {  # slot -> poll status, fetch, gates
  read KIT BUILD SLUG W3 CHECK MV MN BWX <<< "$(cfg $1)"
  D=$SP/$KIT/$BUILD; V=$(cat "$D/.sat_version" 2>/dev/null || echo "?")
  for i in $(seq 1 120); do
    st=$(kaggle kernels status eeshsaxena/$SLUG 2>&1 | tail -1)
    echo "$st" | grep -qiE "complete|error|cancel" && break; sleep 60; done
  say "slot $1 $SLUG v$V status: $st"
  echo "$st" | grep -qi complete || { say "slot $1 NOT COMPLETE -> no gate"; return 1; }
  O=$D/out_v$V; mkdir -p "$O"
  kaggle kernels output eeshsaxena/$SLUG -p "$O" --page-size 200 > "$O/_fetch.log" 2>&1; say "slot $1 fetched $(ls $O | wc -l) files to $O"
  M=${!MV}; N=${!MN}
  $PY $SP/$KIT/fnw_check.py --work $O --fn_build $D/FN_BUILD.json --kernel fn --expect_w3 $W3 --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$M" --expect_manifest "$N" --expect_tta "$BWX/K5_BUILD.json" --expect_n_ranks log949 > "$O/_fnw_check.txt" 2>&1
  r1=$(grep -E "FNWCHECK|FNCHECK" "$O/_fnw_check.txt" | tail -1)
  $PY $SP/$KIT/$CHECK --work $O --fn_build $D/FN_BUILD.json --kernel fn --expect_n clean --expect_fleet clean --kaggle_commit --expect_members "$M" --expect_manifest "$N" --expect_tta "$BWX/K5_BUILD.json" --expect_n_ranks log949 > "$O/_check.txt" 2>&1
  r2=$(grep -E "FNCHECK" "$O/_check.txt" | tail -1)
  say "slot $1 $SLUG v$V GATES: fnw [$r1] | $CHECK [$r2]"
}
for s in "$@"; do push $s; sleep 20; done
for s in "$@"; do waitfetch $s; done
