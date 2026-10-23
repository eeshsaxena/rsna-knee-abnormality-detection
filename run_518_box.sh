#!/bin/bash
cd 'C:\Users\eeshs\Downloads\rsna-knee'
echo "[$(date +%H:%M)] scp gold58_518.npz (440MB) to box..."
t0=$(date +%s)
scp -o BatchMode=yes _exp518_out/gold58_518.npz training@192.168.10.30:/home/training/rsna_train/gold58_518.npz
echo "[$(date +%H:%M)] scp done in $(( $(date +%s)-t0 ))s. Embedding on box CPU (40 cores)..."
ssh -o BatchMode=yes training@192.168.10.30 'cd ~/rsna_train && OMP_NUM_THREADS=40 ~/miniconda3/envs/rsna/bin/python emb_res_loo.py gold58_518.npz ~/dinov2_small train.csv 2>&1 | grep -viE "warn|Depend|Convergence|n_iter|STOP|Increase|scale|FutureWarning"'
echo "[$(date +%H:%M)] 518 BOX EXPERIMENT DONE"
