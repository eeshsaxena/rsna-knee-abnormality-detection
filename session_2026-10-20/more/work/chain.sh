#!/bin/bash
# guards1020 chain: sequential GPU jobs, total <= 45 min; refuses to start any job while another compute process holds the GPU
H=$HOME/guards1020; PY=$HOME/miniconda3/envs/kmimic/bin/python; T0=$(date +%s); CAP=2700
busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
left() { echo $(( CAP - ($(date +%s) - T0) )); }
for j in def_r224 base_r224 def_r384 base_r384; do
  if busy; then echo "$(date +%T) GPU busy, stop before $j"; exit 1; fi
  L=$(left); [ $L -lt 300 ] && { echo "$(date +%T) time cap, stop before $j"; exit 1; }
  tag=${j#*_}; cd $H/rap_$j
  echo "$(date +%T) start $j"; timeout $(( L < 480 ? L : 480 )) $PY -u run_$tag.py > $H/rap_$j.log 2>&1; echo "$(date +%T) end $j rc $?"
done
if busy; then echo "$(date +%T) GPU busy, stop before R6"; exit 1; fi
L=$(left); [ $L -lt 600 ] && { echo "$(date +%T) time cap, stop before R6"; exit 1; }
echo "$(date +%T) start R6 DEF (cap $((L-60)) s)"; timeout $((L-60)) $PY -u $H/g3_r6run.py; echo "$(date +%T) end R6 rc $?"
echo CHAIN_DONE
