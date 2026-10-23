#!/bin/bash
# auto_j1.sh (2026-10-04 18:40 IST): keep the V100 busy overnight. Waits for BOTH running fold-0 jobs to exit
# (TTR pid 267158, gated pid 267280), reads gate G-T2 (gt2_read.py, final), then launches J1 ALONE (box_schedule.md
# sec 5.2 / 6): PASS or MIDDLE -> TTR fold-0 on mix50_logit seed 1043 (recipe test under production labels + a member);
# KILL -> F1 all-data mix50_logit seed 43, --epochs 10 --stop_epoch 8. Never kills anything. Guard: no other python
# compute app on the GPU and >= 12 GB free before launch.
PY=~/miniconda3/envs/rsna/bin/python
LOG=~/fleet_runs/auto_j1.log
say(){ echo "[auto-j1 $(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
say "armed; waiting for pids 267158 (TTR) and 267280 (gated)"
while kill -0 267158 2>/dev/null || kill -0 267280 2>/dev/null; do sleep 60; done
say "both fold-0 jobs exited"
cd ~/bigjump_build/code
$PY gt2_read.py --ttr ~/fleet_runs/f0_s1042_ttr --ref ~/fleet_runs/f0_s42_v4 --eval_labels ~/rsna_train/llm_labels_v4_blend.csv > ~/fleet_runs/gt2_final.txt 2>&1
V=$(grep -o "G-T2 VERDICT: [A-Z]*" ~/fleet_runs/gt2_final.txt | awk '{print $3}')
say "G-T2 final verdict: ${V:-UNKNOWN} ($(grep '^G-T2: ' ~/fleet_runs/gt2_final.txt | cut -c1-160))"
echo "[auto-j1 $(date '+%m-%d %H:%M')] G-T2 ${V:-UNKNOWN}" >> ~/fleet_runs/gate_history.log
while true; do
  NP=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c python)
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$NP" -eq 0 ] && [ "$FREE" -ge 12000 ] && break
  say "guard wait: python apps $NP free ${FREE} MiB"; sleep 120
done
COMMON="--store $HOME/fleet_store --labels $HOME/rsna_train/labels_mix50_logit.csv --folds $HOME/rsna_train/folds5.csv --exclude $HOME/rsna_train/gold74_eval_set.csv --eval_labels $HOME/rsna_train/llm_labels_v4_blend.csv --gold $HOME/rsna_train/gold74_eval_set.csv"
if [ "$V" = "KILL" ]; then
  OUT=~/fleet_runs/all_s43_mix50
  nohup $PY fleet_train.py $COMMON --fold -1 --epochs 10 --stop_epoch 8 --seed 43 --workers 10 --out $OUT > $OUT.log 2>&1 &
else
  OUT=~/fleet_runs/f0_s1043_ttr_mix50
  nohup $PY fleet_train.py --model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed 1043 --save_epochs 6-24 --workers 10 $COMMON --fold 0 --out $OUT > $OUT.log 2>&1 &
fi
say "launched J1 pid $! -> $OUT (verdict ${V:-UNKNOWN})"
