#!/bin/bash
# Clean single autonomous 384px 3-fold B3 ensemble -> push each best model to private HF repo.
cd ~/rsna_train
PY=~/miniconda3/envs/rsna/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
REPO="eeshsaxena/rsna-knee-models"
$PY -c "from huggingface_hub import create_repo; create_repo('$REPO', repo_type='model', private=True, exist_ok=True)" 2>/dev/null
pushbest(){ f=$(ls $1/*_BEST.pt 2>/dev/null|head -1); [ -n "$f" ] && $PY -c "from huggingface_hub import upload_file; upload_file(path_or_fileobj='$f',path_in_repo='$2',repo_id='$REPO',repo_type='model')" && echo "[$(date +%H:%M)] pushed $2"; }
for fold in 0 1 2; do
  echo "[$(date +%H:%M)] === 384 B3 fold$fold ==="
  $PY -u e2e_train.py --shards ~/rsna_shards_384 --labels pseudo_labels_qwen7b.csv --train train.csv --lang _reports_lang.csv --ckpt ~/rsna_train/e2e_384_f$fold --backbone tf_efficientnet_b3 --res 384 --K 6 --epochs 18 --fold $fold > e2e_384_f$fold.log 2>&1
  echo "[$(date +%H:%M)] fold$fold done: $(grep 'DONE best' e2e_384_f$fold.log|tail -1)"
  pushbest e2e_384_f$fold b3_384_fold${fold}_BEST.pt
done
echo "[$(date +%H:%M)] === 384 ENSEMBLE COMPLETE — models on HF ==="
