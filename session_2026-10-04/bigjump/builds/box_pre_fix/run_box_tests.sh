#!/bin/bash
# run_box_tests.sh: the bigjump box CPU tests (no GPU: CUDA_VISIBLE_DEVICES empty; nice 10; no process is touched).
# usage: bash ~/bigjump_build/box/run_box_tests.sh [N for the real tta_bench smoke, default 3]
set -u
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore
PY=~/miniconda3/envs/rsna/bin/python
cd ~/bigjump_build && mkdir -p out
for t in box_check box_parity "box_gp0 ${1:-3}"; do
  set -- $t
  echo "=== $1 $(date +%H:%M:%S)"
  nice -n 10 $PY -W ignore box/$1.py ${2:-} > out/$1.log 2>&1; echo "rc $?"
  grep -E "PASS|FAIL|^\(|^    " out/$1.log | head -60
done
echo "=== done $(date +%H:%M:%S)"
