#!/bin/bash
# ~/kc/code/kc_g1_early.sh -- early G1 view ablation on the f0 epoch-3 ckpt (critique, adopted). Waits for the epoch-3
# raw ckpt to be complete (the ep3 EMA ckpt, saved after it, exists + 60 s), then runs kc_g1_ablation.py on cores 28-39
# (12 torch threads, GPU hidden; MEASURED ~7.6 s/study -> n_max 400 ~ 50 min). INFO for main only: writes
# ~/kc/gate/kc_f0_ep3_ablation.{txt,json} (EARLY_G1 lines, never a queue gate). Main decides on a clear FAIL.
# Arm (main, once KC f0 is #RUN): setsid nohup bash ~/kc/code/kc_g1_early.sh > ~/kc/logs/g1_early.log 2>&1 < /dev/null &
set -u
OUT=${KE_OUT:-$HOME/fleet_runs/f0_s2001_kc256_r7tb}; EP=${KE_EP:-3}; NMAX=${KE_NMAX:-400}; TMAX=${KE_TMAX:-21600}
RAW="$OUT/epochs/f0_s2001_raw_kcttr256-256x12_ep${EP}_best.pt"; EMA="$OUT/epochs/f0_s2001_ema_kcttr256-256x12_ep${EP}_best.pt"
t0=$(date +%s); echo "$(date '+%F %T') waiting for $RAW"
until [ -f "$EMA" ]; do
  [ $(( $(date +%s) - t0 )) -ge "$TMAX" ] && { echo "$(date '+%F %T') timeout: no epoch-$EP ckpt"; exit 1; }
  sleep 120
done
sleep 60
taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 \
  $HOME/miniconda3/envs/rsna/bin/python $HOME/kc/code/kc_g1_ablation.py --ckpt "$RAW" --store $HOME/kc_store/store \
  --labels $HOME/r7t_labels/real/labels_r7t_b.csv --folds $HOME/rsna_train/folds5.csv --exclude $HOME/rsna_train/gold74_eval_set.csv \
  --fold 0 --n_max "$NMAX" --threads 12 --out $HOME/kc/gate/kc_f0_ep${EP}_ablation
echo "$(date '+%F %T') done rc=$?"
