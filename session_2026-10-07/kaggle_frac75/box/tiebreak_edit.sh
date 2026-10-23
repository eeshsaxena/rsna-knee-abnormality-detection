#!/bin/bash
# tiebreak_edit.sh (kaggle_frac75, 2026-10-07): an EDITOR for `bash ~/queue/qctl.sh edit` (qctl holds the queue lock,
# validates the result with queue_runner.sh --check and installs it atomically). Used ONLY when the Kaggle frac75 read is
# INCONCLUSIVE-NARROW and Eesh chose the 24-epoch box tie-break (RUNBOOK sec 7c): every frac75 gate `via` then scores
# the box probe ~/fleet_runs/f0_s1043_ttr_frac75_b24 with the ORIGINAL scorer instead of the Kaggle run.
#   EDITOR=$HOME/kt_frac75/box/tiebreak_edit.sh bash ~/queue/qctl.sh edit
f=$1
old='via bash ~/queue/gates/frac75_gate.sh |'
new='via F75_RUN=/home/training/fleet_runs/f0_s1043_ttr_frac75_b24 bash ~/queue/gates/frac75_gate.sh |'
n=$(grep -v '^[[:space:]]*#' "$f" | grep -cF "$old")
sed -i "s#${old}#${new}#" "$f"
m=$(grep -v '^[[:space:]]*#' "$f" | grep -cF "$new")
echo "tiebreak_edit: $n active frac75 gate via(s) found, $m now score ~/fleet_runs/f0_s1043_ttr_frac75_b24"
[ "$n" -ge 1 ] && [ "$m" -eq "$n" ]
