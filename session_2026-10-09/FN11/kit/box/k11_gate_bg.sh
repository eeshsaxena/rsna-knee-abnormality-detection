#!/bin/bash
# k11_gate_bg.sh (box; 2026-10-09, Claude; from k10_gold_bg.sh): start the PRE-REGISTERED FN11 gate DETACHED (setsid nohup;
# ssh returns at once). CPU only (CUDA hidden), NTHR 6, taskset CPUs 34-39, nice 19, ionice idle; writes only ~/k11_test/gold.
#   bash ~/k11_test/k11/box/k11_gate_bg.sh [extra k11_gate.py args]
# Log: ~/k11_test/gold/gate_<HHMM>.log (last line K11GATE ...; then "wrote ...").
set -u
[ -f ~/k11_test/members.json ] || { echo "no ~/k11_test/members.json: run k11_collect.sh first"; exit 1; }
mkdir -p ~/k11_test/gold && cd ~/k11_test/gold || exit 1
H=$(date +%H%M)
CUDA_VISIBLE_DEVICES= NTHR=6 setsid nohup taskset -c 34-39 nice -n 19 ionice -c3 ~/miniconda3/envs/kmimic/bin/python -W ignore \
  ~/k11_test/k11/box/k11_gate.py --infer --members ~/k11_test/members.json "$@" > ~/k11_test/gold/gate_$H.log 2>&1 < /dev/null &
echo "started k11_gate (pid $!) log ~/k11_test/gold/gate_$H.log"
