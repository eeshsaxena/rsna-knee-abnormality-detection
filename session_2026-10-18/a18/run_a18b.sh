#!/bin/bash
# run_a18b.sh (10-18, Claude): push + gate a15, c4a15, pf50 (<= 2 concurrent commits), merge-safe fetch. NEVER submits.
set -u
cd "$(dirname "$0")/../.." || exit 1
D=session_2026-10-18/a18
HWT='"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker'"'"'s": 0.35'
C4T='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker'"'"'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
declare -A TB=( [a15]="{$HWT}" [c4a15]="$C4T" [pf50]='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker'"'"'s": 0.35}' )
declare -A AA=( [a15]=0.15 [c4a15]=0.15 [pf50]=0.25 )
declare -A TAG=( [a15]=v224r6a15j [c4a15]=v224r6a15c4j [pf50]=v224r6a25pf50j )
running() { python session_2026-10-15/v224f/running.py 6 2>/dev/null | grep RUNNING_COUNT | awk '{print $2}'; }
gate_one() { P=$1; T=${TAG[$P]}; OUT=$D/out_${T}_v1
  for k in 1 2 3 4; do rm -rf $OUT.tmp; python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$T $OUT.tmp 400 > $OUT.log 2>&1
    if ! grep -rl "Server Error" $OUT.tmp >/dev/null 2>&1; then rm -rf $OUT; mv $OUT.tmp $OUT; break; fi; echo "[$(date +%H:%M)] $P fetch had 502s, retry $k"; sleep 120; done
  py -3.12 -I session_2026-10-18/probes/probe_check_t.py $OUT --a224 ${AA[$P]} --table "${TB[$P]}" > $D/gate_${T}_v1.txt 2>&1
  echo "[$(date +%H:%M)] $P gate: $(tail -1 $D/gate_${T}_v1.txt)"; }
for P in a15 c4a15 pf50; do
  T=${TAG[$P]}
  grep -q "successfully pushed" $D/push_$T.log 2>/dev/null && continue
  for i in $(seq 1 600); do N=$(running); [ "${N:-9}" -lt 2 ] && break; sleep 60; done
  kaggle kernels push -p $D/kernel_$T > $D/push_$T.log 2>&1
  grep -q "successfully pushed" $D/push_$T.log || { sleep 120; kaggle kernels push -p $D/kernel_$T > $D/push_$T.log 2>&1; }
  echo "[$(date +%H:%M)] $P push: $(tail -1 $D/push_$T.log)"
  ( gate_one $P ) &
  sleep 60
done
wait; echo "[$(date +%H:%M)] RUN_A18B_DONE"
