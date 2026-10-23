#!/bin/bash
# k10_gold_bg.sh (box; 2026-10-09, Claude): start the K10 gold read DETACHED (setsid nohup inside this script; ssh returns
# at once). CPU only (CUDA hidden), NTHR 6, taskset CPUs 34-39, nice 19, ionice idle; writes only ~/k10_test/gold.
#   bash ~/k10_test/k10/box/k10_gold_bg.sh infer     # predictions for every finished run without one (--members none)
#   bash ~/k10_test/k10/box/k10_gold_bg.sh final     # the decision read (--infer, --members ~/k10_test/members.json)
# Log: ~/k10_test/gold/<mode>_<HHMM>.log (last line K10GOLD ...).
set -u
mkdir -p ~/k10_test/gold && cd ~/k10_test/gold || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
case ${1:-} in
  infer) ARGS="--infer --members none";;
  final) [ -f ~/k10_test/members.json ] || { echo "no ~/k10_test/members.json: run k10_collect.sh first"; exit 1; }
         ARGS="--infer --members $HOME/k10_test/members.json";;
  *) echo "usage: k10_gold_bg.sh infer|final"; exit 2;;
esac
H=$(date +%H%M)
CUDA_VISIBLE_DEVICES= NTHR=6 setsid nohup taskset -c 34-39 nice -n 19 ionice -c3 $PYK -W ignore ~/k10_test/k10/box/k10_gold_read.py $ARGS > ~/k10_test/gold/${1}_$H.log 2>&1 < /dev/null &
echo "started k10_gold_read $1 (pid $!) log ~/k10_test/gold/${1}_$H.log"
