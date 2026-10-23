#!/bin/bash
# run_a18.sh (10-18, Claude): A18 probes: push as GPU commit slots free (max 2 concurrent), fetch (merge-safe), gate. NEVER submits.
set -u
cd "$(dirname "$0")/../.." || exit 1
D=session_2026-10-18/a18
HW='"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker'"'"'s": 0.35'
declare -A TB=( [c4]='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker'"'"'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
                [fr65]='{'"$HW"', "Fracture": 0.65}' [a15]='{'"$HW"'}' [pf50]='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker'"'"'s": 0.35}'
                [lm40]='{'"$HW"', "Lateral Meniscus": 0.4}' [bk50]='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker'"'"'s": 0.5}' )
declare -A TAG=( [c4]=v224r6a25c4j [fr65]=v224r6a25fr65j [a15]=v224r6a15j [pf50]=v224r6a25pf50j [lm40]=v224r6a25lm40j [bk50]=v224r6a25bk50j )
running() { python session_2026-10-15/v224f/running.py 6 2>/dev/null | grep RUNNING_COUNT | awk '{print $2}'; }
gate_one() { P=$1; T=${TAG[$P]}; OUT=$D/out_${T}_v1; A=0.25; [ $P = a15 ] && A=0.15
  for k in 1 2 3 4; do rm -rf $OUT.tmp; python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$T $OUT.tmp 400 > $OUT.log 2>&1
    if ! grep -rl "Server Error" $OUT.tmp >/dev/null 2>&1; then rm -rf $OUT; mv $OUT.tmp $OUT; break; fi; echo "[$(date +%H:%M)] $P fetch had 502s, retry $k"; sleep 120; done
  py -3.12 -I session_2026-10-18/probes/probe_check_t.py $OUT --a224 $A --table "${TB[$P]}" > $D/gate_${T}_v1.txt 2>&1
  echo "[$(date +%H:%M)] $P gate: $(tail -1 $D/gate_${T}_v1.txt)"; }
for P in c4 fr65 a15 pf50 lm40 bk50; do
  T=${TAG[$P]}
  grep -q "successfully pushed" $D/push_$T.log 2>/dev/null && continue
  for i in $(seq 1 600); do N=$(running); [ "${N:-9}" -lt 2 ] && break; sleep 60; done
  kaggle kernels push -p $D/kernel_$T > $D/push_$T.log 2>&1
  grep -q "successfully pushed" $D/push_$T.log || { sleep 120; kaggle kernels push -p $D/kernel_$T > $D/push_$T.log 2>&1; }
  echo "[$(date +%H:%M)] $P push: $(tail -1 $D/push_$T.log)"
  ( gate_one $P ) &
  sleep 60
done
wait
echo "[$(date +%H:%M)] RUN_A18_DONE"
