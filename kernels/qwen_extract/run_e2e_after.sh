#!/bin/bash
# Runs AFTER the extraction+frozen-retrain chain. Keeps the GPU fully busy training
# end-to-end models on the CLEAN 7B labels (the shot at breaking the frozen 0.63 ceiling).
cd ~/rsna_train
PY=~/miniconda3/envs/rsna/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LAB=pseudo_labels_qwen7b.csv
echo "[$(date +%H:%M)] waiting for clean labels + frozen head..."
while [ ! -f head_weights_7b.pt ]; do sleep 60; done
sleep 45   # let the retrain release the GPU
echo "[$(date +%H:%M)] === e2e EfficientNet-B3 on CLEAN labels ==="
$PY -u e2e_train.py --shards ~/rsna_shards_full --labels $LAB --train train.csv --lang _reports_lang.csv \
   --ckpt ~/rsna_train/e2e_b3_clean --backbone tf_efficientnet_b3 --res 224 --K 12 --epochs 20 --fold 0 \
   > e2e_b3_clean.log 2>&1
echo "[$(date +%H:%M)] b3 exit $?  best:"; grep "DONE best" e2e_b3_clean.log | tail -1
echo "[$(date +%H:%M)] === e2e EfficientNet-B0 on CLEAN labels (diversity) ==="
$PY -u e2e_train.py --shards ~/rsna_shards_full --labels $LAB --train train.csv --lang _reports_lang.csv \
   --ckpt ~/rsna_train/e2e_b0_clean --backbone tf_efficientnet_b0 --res 224 --K 12 --epochs 20 --fold 0 \
   > e2e_b0_clean.log 2>&1
echo "[$(date +%H:%M)] b0 exit $?  best:"; grep "DONE best" e2e_b0_clean.log | tail -1
echo "[$(date +%H:%M)] === ALL E2E DONE ==="
