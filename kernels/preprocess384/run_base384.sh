#!/bin/bash
# DINOv2-BASE @384px + head on clean labels — bigger backbone at the winning resolution.
cd ~/rsna_train
PY=~/miniconda3/envs/rsna/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[$(date +%H:%M)] downloading DINOv2-base..."
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov2-base', local_dir='/home/training/dinov2_base')" 2>&1 | tail -1
echo "[$(date +%H:%M)] === embed 384 shards with DINOv2-BASE ==="
$PY -u box_embed384.py --model /home/training/dinov2_base --out emb_cache_base384.npz > embed_base384.log 2>&1
echo "[$(date +%H:%M)] embed: $(tail -1 embed_base384.log)"
echo "[$(date +%H:%M)] === head on base@384 + clean labels ==="
$PY -u box_head_retrain.py --emb emb_cache_base384.npz --labels pseudo_labels_qwen7b.csv --out head_weights_base384.pt > retrain_base384.log 2>&1
echo "[$(date +%H:%M)] base@384 head: $(grep 'ENSEMBLE gold-58' retrain_base384.log | tail -1)"
$PY -c "from huggingface_hub import upload_file; upload_file(path_or_fileobj='head_weights_base384.pt',path_in_repo='head_weights_base384.pt',repo_id='eeshsaxena/rsna-knee-models',repo_type='model')" && echo "pushed base384 head"
echo "[$(date +%H:%M)] === BASE384 COMPLETE ==="
