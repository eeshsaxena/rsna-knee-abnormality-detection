#!/bin/bash
# dry_run.sh (gold_prep/r50, CPU only, nice 19): the armed R50 probe trainer argv, verbatim, with ONLY these changes:
#   --store ~/gold_prep/r50/store (25 symlinked studies), --out ~/gold_prep/r50/dry{A,B}, --device cpu, --min_cover 0,
#   --time_epochs 1 (stop after epoch 1; --epochs 20 and the cosine schedule stay as armed).
#   B additionally uses --save_epochs 1-20 so the per-epoch save path (epochs/ dir) is exercised in epoch 1.
# GPU hidden (CUDA_VISIBLE_DEVICES=), HF hub offline, no .pyc writes into ~/fast_build/code2.
PY=~/miniconda3/envs/rsna/bin/python; G=$HOME/gold_prep/r50
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd ~/fast_build/code2 || exit 1
for V in A B; do
  OUT=$G/dry$V; SE=6-20; [ $V = B ] && SE=1-20
  rm -rf $OUT
  echo "[dry $V $(date +%H:%M:%S)] start" >> $G/dry_run.log
  /usr/bin/time -v nice -n 19 $PY fleet_train.py --f2 1 --backbone resnet50.a1_in1k --lr 3e-4 --seed 243 --freeze_bn 1 --channels_last 1 \
    --aug heavy --train_slices 16 --drop_path 0.1 --dropout 0.3 --ema 0.9995 --epochs 20 --save_epochs $SE --workers 10 \
    --store $G/store --labels $HOME/rsna_train/labels_mix50_logit.csv --folds $HOME/rsna_train/folds5.csv --fold 0 \
    --exclude $HOME/rsna_train/gold74_eval_set.csv --eval_labels $HOME/rsna_train/llm_labels_v4_blend.csv \
    --gold $HOME/rsna_train/gold74_eval_set.csv --grad_ckpt 1 --out $OUT \
    --device cpu --min_cover 0 --time_epochs 1 > $OUT.log 2>&1
  echo "[dry $V $(date +%H:%M:%S)] rc=$? last: $(grep -v '^\s' $OUT.log | grep '^{' | tail -n 1)" >> $G/dry_run.log
done
echo "[dry done $(date +%H:%M:%S)]" >> $G/dry_run.log
