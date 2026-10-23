#!/bin/bash
# r5_chain.sh (10-20 r5, Claude): R6 (gold58 approx first, then OAI in the stratified order; one compile, worker deadline
# R6_HOURS) then S6fall on the same id list (deadline S6_HOURS). Each step waits for a free GPU (no compute process for 2
# checks 60 s apart) and runs alone. Inference only.
set -u
R6_HOURS=${1:-2.75}; S6_HOURS=${2:-1.0}
cd ~/r5_oai || exit 1
PY=$HOME/miniconda3/envs/kmimic/bin/python
echo "CHAIN_START $(date '+%m-%d %H:%M:%S') r6_hours $R6_HOURS s6_hours $S6_HOURS"
$PY -I r5_run.py --wait-gpu --keep-order --data $HOME/r5_oai/data_r5 --split r5 --ids $HOME/r5_oai/ids_r5.json \
  --out $HOME/r5_oai/run_r6 --loader-workers 1 --expandable --hours $R6_HOURS > logs/run_r6.log 2>&1
echo "R6_STEP_END rc $? $(date '+%m-%d %H:%M:%S')"
rm -rf $HOME/r5_oai/run_r6/worker0/tmp
bash r5_s6.sh $HOME/r5_oai/ids_r5.json r5 $S6_HOURS > logs/run_s6.log 2>&1
echo "S6_STEP_END rc $? $(date '+%m-%d %H:%M:%S')"
echo "CHAIN_DONE $(date '+%m-%d %H:%M:%S')"
