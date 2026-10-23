#!/bin/bash
# Box: pull 384px shards from HF, then train e2e at 384px on the CLEAN 7B labels (the ceiling-break).
cd ~/rsna_train
PY=~/miniconda3/envs/rsna/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[$(date +%H:%M)] pulling 384px shards from HF..."
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('eeshsaxena/rsna-knee-384', repo_type='dataset', local_dir='/home/training/rsna_shards_384', allow_patterns=['*.npz'])"
echo "[$(date +%H:%M)] pulled $(ls ~/rsna_shards_384/shard_*.npz 2>/dev/null | wc -l) shards"
# wait for the 224px e2e chain to FINISH (grad-ckpt uses little memory, so a memory check
# would false-trigger while e2e still runs -> serialize on the completion marker instead)
echo "[$(date +%H:%M)] waiting for 224px e2e chain to finish..."
while ! grep -q "ALL E2E DONE" ~/rsna_train/e2e_after.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] e2e chain done, GPU free:"; nvidia-smi --query-gpu=memory.free --format=csv,noheader
echo "[$(date +%H:%M)] === e2e B3 @384px on CLEAN labels ==="
$PY -u e2e_train.py --shards ~/rsna_shards_384 --labels pseudo_labels_qwen7b.csv --train train.csv --lang _reports_lang.csv \
   --ckpt ~/rsna_train/e2e_384_b3 --backbone tf_efficientnet_b3 --res 384 --K 6 --epochs 18 --fold 0 > e2e_384_b3.log 2>&1
echo "[$(date +%H:%M)] 384 B3 exit $?  best:"; grep "DONE best" e2e_384_b3.log | tail -1
echo "[$(date +%H:%M)] === 384 COMPLETE ==="
