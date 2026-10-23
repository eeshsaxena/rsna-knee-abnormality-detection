#!/bin/bash
# Box-side autonomous 384 trigger: wait until the HF shard count stabilizes (kernel done),
# then pull + train. No laptop dependency — survives any session restart.
cd ~/rsna_train
PY=~/miniconda3/envs/rsna/bin/python
prev=-1; stable=0
echo "[$(date +%H:%M)] watching HF for 384px shards to finish..."
while true; do
  n=$($PY -c "from huggingface_hub import HfApi; print(len([f for f in HfApi().list_repo_files('eeshsaxena/rsna-knee-384',repo_type='dataset') if f.endswith('.npz')]))" 2>/dev/null)
  [ -z "$n" ] && n=0
  echo "[$(date +%H:%M)] shards=$n"
  if [ "$n" = "$prev" ] && [ "$n" -gt 50 ]; then stable=$((stable+1)); else stable=0; fi
  if [ "$stable" -ge 2 ]; then echo "[$(date +%H:%M)] shard count stable at $n -> proceeding"; break; fi
  prev=$n; sleep 180
done
bash run_384_box.sh
