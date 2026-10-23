#!/bin/bash
# Pivot to the real ceiling-break: frozen DINOv2 @384px + MIL head on clean labels.
cd ~/rsna_train
PY=~/miniconda3/envs/rsna/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[$(date +%H:%M)] waiting for fold0 e2e to finish (keep its result)..."
while ! grep -q "DONE best" e2e_384_f0.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] fold0 done. stopping e2e ensemble, pivoting to DINOv2@384"
pkill -9 -f run_384_clean; pkill -9 -f "e2e_train.py"; sleep 5
echo "[$(date +%H:%M)] === embedding 384px shards with DINOv2 ==="
$PY -u box_embed384.py > embed384.log 2>&1
echo "[$(date +%H:%M)] embed: $(tail -1 embed384.log)"
echo "[$(date +%H:%M)] === training head on DINOv2@384 features + clean labels ==="
$PY -u box_head_retrain.py --emb emb_cache_384.npz --labels pseudo_labels_qwen7b.csv --out head_weights_384.pt > retrain384.log 2>&1
echo "[$(date +%H:%M)] DINOv2@384 head: $(grep 'ENSEMBLE gold-58' retrain384.log | tail -1)"
$PY -c "from huggingface_hub import upload_file; upload_file(path_or_fileobj='head_weights_384.pt',path_in_repo='head_weights_384.pt',repo_id='eeshsaxena/rsna-knee-models',repo_type='model')" && echo "[$(date +%H:%M)] pushed head_weights_384.pt to HF"
echo "[$(date +%H:%M)] === DINO384 COMPLETE ==="
