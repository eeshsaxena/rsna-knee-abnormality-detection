#!/bin/bash
# measure_decode.sh (10-15, Claude subagent, kcstage): KC decode cost on box CPU, Kaggle-like layout (2 serve processes x --procs 2
# on 4 cores = one process per T4, --decode_only 1). Runs ORIGINAL kc_worker.py and kc_serve.py (side-table path) on the same fake
# test root and compares their rows. Box cores 28-31 (4 cores), nice 19, ionice c3, CUDA hidden. RSNA train studies only.
#   bash measure_decode.sh ROOT SIDE_CSV OUTDIR
set -u
ROOT=$1; SIDE=$2; OUT=$3
PY=~/miniconda3/envs/kmimic/bin/python
K=~/kcstage_1015
mkdir -p $OUT
E="nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1"
run_pair () {   # $1 = label, $2 = orig|serve, $3 = cores
  local t0=$(date +%s.%N)
  for k in 0 1; do
    if [ "$2" = orig ]; then
      taskset -c $3 $E $PY $K/code/kc_worker.py --root $ROOT --split test --out $OUT/$1_p$k.csv --decode_only 1 --procs 2 --shard $k/2 > $OUT/$1_p$k.log 2>&1 &
    else
      taskset -c $3 $E $PY $K/kc_serve.py --code $K/code --side $SIDE --timing $OUT/$1_p$k.timing.json -- --root $ROOT --split test --out $OUT/$1_p$k.csv --decode_only 1 --procs 2 --shard $k/2 > $OUT/$1_p$k.log 2>&1 &
    fi
  done
  wait
  local t1=$(date +%s.%N)
  echo "PAIR $1 $2 cores $3 wall_s $(echo "$t1 - $t0" | bc)"
  tail -n 1 $OUT/$1_p0.log $OUT/$1_p1.log
}
cat /proc/loadavg
run_pair warm orig 28-31
run_pair orig orig 28-31
run_pair serve serve 28-31
run_pair orig2 orig 28-31
t0=$(date +%s.%N)
taskset -c 28 $E $PY $K/kc_serve.py --code $K/code --side $SIDE --timing $OUT/serial.timing.json -- --root $ROOT --split test --out $OUT/serial.csv --decode_only 1 --procs 0 > $OUT/serial.log 2>&1
t1=$(date +%s.%N)
echo "SERIAL serve 1 core wall_s $(echo "$t1 - $t0" | bc)"; tail -n 1 $OUT/serial.log
cat /proc/loadavg
