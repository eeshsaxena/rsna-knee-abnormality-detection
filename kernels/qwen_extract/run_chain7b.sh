#!/bin/bash
# Autonomous chain on the box: 7B label extraction -> retrain deployment head -> head_weights_7b.pt
cd ~/rsna_train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
PY=~/miniconda3/envs/rsna/bin/python
echo "[$(date +%H:%M)] === extraction start ==="
$PY -u box_extract7b.py --bs 2 --all 1 > extract7b.log 2>&1
echo "[$(date +%H:%M)] === extraction exit $? ==="
if [ -f pseudo_labels_qwen7b.csv ]; then
  echo "[$(date +%H:%M)] === retrain start ==="
  $PY -u box_head_retrain.py > retrain7b.log 2>&1
  echo "[$(date +%H:%M)] === retrain exit $? ==="
  grep -E "ENSEMBLE gold-58|WROTE" retrain7b.log 2>/dev/null | tail -3
else
  echo "NO LABELS FILE — extraction failed"
fi
echo "[$(date +%H:%M)] === CHAIN COMPLETE ==="
