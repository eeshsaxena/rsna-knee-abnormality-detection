#!/bin/bash
# k12_gate_bg.sh (box; 2026-10-09, Claude; from k11_gate_bg.sh): start the PRE-REGISTERED K12 gate DETACHED. CPU only,
# NTHR 6, taskset CPUs 34-39, nice 19, ionice idle; writes only ~/k12_test/gold (or the --out / --pred_dir given).
#   bash ~/k12_test/k12/box/k12_gate_bg.sh [extra k12_gate.py args]      log ~/k12_test/gold/gate_<HHMM>.log
set -u
[ -f ~/k12_test/members.json ] || { echo "no ~/k12_test/members.json: run k12_collect.sh first"; exit 1; }
mkdir -p ~/k12_test/gold && cd ~/k12_test/gold || exit 1
H=$(date +%H%M)
CUDA_VISIBLE_DEVICES= NTHR=6 setsid nohup taskset -c 34-39 nice -n 19 ionice -c3 ~/miniconda3/envs/kmimic/bin/python -W ignore \
  ~/k12_test/k12/box/k12_gate.py --infer --members ~/k12_test/members.json "$@" > ~/k12_test/gold/gate_$H.log 2>&1 < /dev/null &
echo "started k12_gate (pid $!) log ~/k12_test/gold/gate_$H.log"
