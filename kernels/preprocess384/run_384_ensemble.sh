#!/bin/bash
# Fully autonomous (box survives user leaving + laptop disconnect): after 384 fold0, keep the GPU
# busy training a 3-fold 384px B3 ensemble on the clean 7B labels, pushing each best model to a
# private HF repo so the laptop can pull + ensemble + submit from anywhere.
cd ~/rsna_train
PY=~/miniconda3/envs/rsna/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
REPO="eeshsaxena/rsna-knee-models"
$PY -c "from huggingface_hub import create_repo; create_repo('$REPO', repo_type='model', private=True, exist_ok=True)" 2>/dev/null
pushbest() {  # $1=ckpt dir  $2=name on HF
  f=$(ls $1/*_BEST.pt 2>/dev/null | head -1)
  if [ -n "$f" ]; then $PY -c "from huggingface_hub import upload_file; upload_file(path_or_fileobj='$f', path_in_repo='$2', repo_id='$REPO', repo_type='model')" && echo "[$(date +%H:%M)] pushed $2 to HF"; fi
}
echo "[$(date +%H:%M)] waiting for 384 B3 fold0 (run_384_box.sh)..."
while ! grep -q "DONE best" e2e_384_b3.log 2>/dev/null; do sleep 180; done
pushbest e2e_384_b3 b3_384_fold0_BEST.pt
grep "DONE best" e2e_384_b3.log | tail -1
for fold in 1 2; do
  echo "[$(date +%H:%M)] === 384 B3 fold$fold ==="
  $PY -u e2e_train.py --shards ~/rsna_shards_384 --labels pseudo_labels_qwen7b.csv --train train.csv \
     --lang _reports_lang.csv --ckpt ~/rsna_train/e2e_384_b3_f$fold --backbone tf_efficientnet_b3 \
     --res 384 --K 6 --epochs 18 --fold $fold > e2e_384_b3_f$fold.log 2>&1
  echo "[$(date +%H:%M)] fold$fold: $(grep 'DONE best' e2e_384_b3_f$fold.log | tail -1)"
  pushbest e2e_384_b3_f$fold b3_384_fold${fold}_BEST.pt
done
echo "[$(date +%H:%M)] === 384 ENSEMBLE COMPLETE — all models on HF $REPO ==="
