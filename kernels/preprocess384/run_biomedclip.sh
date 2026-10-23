#!/bin/bash
# BiomedCLIP (medical image-text backbone) -> the 0.93-cluster's key. Runs after base@384.
cd ~/rsna_train
PY=~/miniconda3/envs/rsna/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[$(date +%H:%M)] waiting for base@384 to finish..."
while ! grep -q "BASE384 COMPLETE" run_base384.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] === embedding shards with BiomedCLIP ==="
$PY -u box_embed_biomedclip.py --out emb_cache_biomedclip.npz > embed_biomedclip.log 2>&1
echo "[$(date +%H:%M)] embed: $(tail -1 embed_biomedclip.log)"
echo "[$(date +%H:%M)] === head on BiomedCLIP + clean labels ==="
$PY -u box_head_retrain.py --emb emb_cache_biomedclip.npz --labels pseudo_labels_qwen7b.csv --out head_weights_biomedclip.pt > retrain_biomedclip.log 2>&1
echo "[$(date +%H:%M)] BiomedCLIP head: $(grep 'ENSEMBLE gold-58' retrain_biomedclip.log | tail -1)"
$PY -c "from huggingface_hub import upload_file; upload_file(path_or_fileobj='head_weights_biomedclip.pt',path_in_repo='head_weights_biomedclip.pt',repo_id='eeshsaxena/rsna-knee-models',repo_type='model')" && echo "pushed biomedclip head"
echo "[$(date +%H:%M)] === BIOMEDCLIP COMPLETE ==="
