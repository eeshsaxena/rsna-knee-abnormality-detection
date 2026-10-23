#!/bin/bash
# k14_gate_bg.sh (box; 2026-10-09, Claude; from k11_gate_bg.sh): start the PRE-REGISTERED K12 gate DETACHED. CPU only,
# NTHR 6, taskset CPUs 34-39, nice 19, ionice idle; writes only ~/k14_test/gold (or the --out / --pred_dir given).
#   bash ~/k14_test/k14/box/k14_gate_bg.sh [extra k14_gate.py args]      log ~/k14_test/gold/gate_<HHMM>.log
set -u
[ -f ~/k14_test/members.json ] || { echo "no ~/k14_test/members.json: run k12_collect.sh first"; exit 1; }
mkdir -p ~/k14_test/gold && cd ~/k14_test/gold || exit 1
H=$(date +%H%M)
CUDA_VISIBLE_DEVICES= NTHR=6 setsid nohup taskset -c 34-39 nice -n 19 ionice -c3 ~/miniconda3/envs/kmimic/bin/python -W ignore \
  ~/k14_test/k14/box/k14_gate.py --infer --members ~/k14_test/members.json "$@" > ~/k14_test/gold/gate_$H.log 2>&1 < /dev/null &
echo "started k14_gate (pid $!) log ~/k14_test/gold/gate_$H.log"
