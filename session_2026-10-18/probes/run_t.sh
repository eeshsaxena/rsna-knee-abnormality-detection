#!/bin/bash
# run_t.sh (10-18, Claude): A17 table probes: push the ones not yet pushed as GPU commit slots free (max 2 concurrent), then fetch + gate all.
# NEVER submits (submits need Eesh's explicit go; see submit_t.py).
set -u
cd "$(dirname "$0")/../.." || exit 1
D=session_2026-10-18/probes
declare -A TB=( [pf50]='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker'"'"'s": 0.35}'
                [bk50]='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker'"'"'s": 0.5}'
                [lm40]='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker'"'"'s": 0.35, "Lateral Meniscus": 0.4}'
                [fr65]='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker'"'"'s": 0.35, "Fracture": 0.65}'
                [c4]='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker'"'"'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}' )
running() { python session_2026-10-15/v224f/running.py 6 2>/dev/null | grep RUNNING_COUNT | awk '{print $2}'; }
for P in lm40 fr65 c4; do
  TAG=v224r6hw${P}j
  grep -q "successfully pushed" $D/push_$TAG.log 2>/dev/null && continue
  for i in $(seq 1 400); do
    N=$(running); [ "${N:-9}" -lt 2 ] && break; sleep 60
  done
  kaggle kernels push -p $D/kernel_$TAG > $D/push_$TAG.log 2>&1
  if ! grep -q "successfully pushed" $D/push_$TAG.log; then sleep 120; kaggle kernels push -p $D/kernel_$TAG > $D/push_$TAG.log 2>&1; fi
  echo "[$(date +%H:%M)] $P push: $(tail -1 $D/push_$TAG.log)"
  sleep 30
done
for P in pf50 bk50 lm40 fr65 c4; do
  TAG=v224r6hw${P}j; OUT=$D/out_${TAG}_v1
  grep -q "successfully pushed" $D/push_$TAG.log || { echo "$P NOT PUSHED"; continue; }
  [ -f $D/gate_${TAG}_v1.txt ] && tail -1 $D/gate_${TAG}_v1.txt | grep -q "PASS 16/16" && { echo "$P already gated"; continue; }
  rm -rf $OUT
  python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$TAG $OUT 300 > $OUT.log 2>&1
  py -3.12 -I $D/probe_check_t.py $OUT --table "${TB[$P]}" > $D/gate_${TAG}_v1.txt 2>&1
  echo "[$(date +%H:%M)] $P gate: $(tail -1 $D/gate_${TAG}_v1.txt | tr -d '\r')"
done
echo RUN_T_DONE
