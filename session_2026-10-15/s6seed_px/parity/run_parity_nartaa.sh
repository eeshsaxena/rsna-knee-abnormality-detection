#!/bin/bash
# PX_PARITY with nartaa's real checkpoint (once ~/s6seed/nartaa/E_prior_s42_fall.pt, sha256 0c7297d5..., is on the box).
# The window-byte part of PX_PARITY is model-independent and already covers all 58 gold studies (run_parity.sh); this adds the
# logits of the real S6fall model on N studies (CPU fp32 convnext_small: ~17 TFLOP per study per side, ~30-45 min per study on
# 2 threads under load, MEASURED-scaled from the atto run). Serve side and train side run in parallel (2 processes).
#   N=2 setsid nohup bash run_parity_nartaa.sh > run_parity_nartaa.log 2>&1 < /dev/null &
set -u
cd ~/s6seed/parity
PY=~/miniconda3/envs/kmimic/bin/python
MD=$HOME/s6seed/nartaa
N=${N:-2}
want=0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12
got=$(sha256sum $MD/E_prior_s42_fall.pt | cut -d" " -f1)
[ "$got" = "$want" ] || { echo "nartaa checkpoint sha mismatch or missing ($got)"; exit 1; }
PFX="taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
tail -n +2 ~/kmimic/ids/gold58.csv | cut -d, -f1 | head -$N > shards/nartaa.lst
$PY -c "import json; print(json.dumps(open('shards/nartaa.lst').read().split()))" > shards/nartaa.json
rm -f train_side_nartaa.json
mkdir -p ref_nartaa/pxdump
$PFX $PY -u px_parity.py --src ../src --px ../px320_gold58 --models $MD --backbone convnext_small --uids shards/nartaa.lst \
  --threads 4 --train-cache train_side_nartaa.json --out train_only_nartaa.json > train_side_nartaa.log 2>&1 &
K1_PX_DUMP=$PWD/ref_nartaa/pxdump S6_SRC=$HOME/s6seed/src S6_CODE=$HOME/s6seed/code $PFX $PY -u ci_cpu_launcher.py \
  --data $HOME/kmimic/kaggle/input/rsna-knee-abnormality-detection --split train --code $HOME/s6seed/code --models $MD --arm E_prior \
  --backbone convnext_small --size 320 --crop 288 --views v4 --slices 3 --phys-mm 3.5 --eval-stride 1 \
  --anchor-weights $HOME/s6seed/weights/anatomy.pt --amp fp32 --chunk 192 --workers 0 --item-timeout 900 \
  --assignment shards/nartaa.json --output ref_nartaa --tmp /dev/shm/s6par_nartaa --dump-logits ref_nartaa/logits.json > ref_nartaa.log 2>&1
wait
$PFX $PY -u px_parity.py --src ../src --px ../px320_gold58 --models $MD --backbone convnext_small --ref ref_nartaa \
  --uids shards/nartaa.lst --threads 4 --train-cache train_side_nartaa.json --out PX_PARITY_nartaa.json > px_parity_nartaa.log 2>&1
tail -1 px_parity_nartaa.log
