#!/bin/bash
# r5_s6.sh IDS_JSON NAME HOURS (10-20 r5, Claude): S6fall = nartaa E_prior_s42_fall.pt (sha 0c7297d5..., the c4 B_MODEL) through
# ~/s6seed/parity/ci_gpu_launcher.py (heliosli compact_infer 34f433fe main() unchanged), worker args IDENTICAL to
# ~/s6seed/parity/run_gold58_serve_gpu.sh (the gold58 GPU serve), on the store-rebuilt split ~/r5_oai/data_r5 (r5).
# Differences: data / split / ids, output + tmp under ~/r5_oai, --deadline, PYTHONDONTWRITEBYTECODE=1 (no new files in ~/s6seed).
# Waits until no compute process holds the GPU (2 checks 60 s apart), refuses nothing else.
set -u
IDS=$1; NAME=$2; HOURS=$3
M=$HOME/s6seed/nartaa/E_prior_s42_fall.pt
[ "$(sha256sum $M | cut -c1-16)" = "0c7297d58f061ab4" ] || { echo "model sha differs"; exit 1; }
PY=$HOME/miniconda3/envs/kmimic/bin/python
O=$HOME/r5_oai/s6_$NAME; rm -rf $O; mkdir -p $O/models
ln -s $M $O/models/E_prior_s42_fall.pt
ok=0; while [ $ok -lt 2 ]; do if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then ok=$((ok+1)); else ok=0; fi; [ $ok -lt 2 ] && sleep 60; done
echo "GPU_FREE $(date '+%m-%d %H:%M:%S')"; echo "S6 r5 $NAME pid $$ $(date)" > $HOME/r5_oai/RUNNING_S6.txt
DL=$($PY -c "import time; print(time.time() + $HOURS * 3600)")
cd ~/s6seed/parity || exit 1
T0=$(date +%s)
S6_SRC=$HOME/s6seed/src S6_CODE=$HOME/s6seed/code PYTHONDONTWRITEBYTECODE=1 nice -n 5 env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
  $PY -u ci_gpu_launcher.py \
  --data $HOME/r5_oai/data_r5 --split r5 --code $HOME/s6seed/code --models $O/models --arm E_prior \
  --backbone convnext_small --size 320 --crop 288 --views v4 --slices 3 --phys-mm 3.5 --eval-stride 1 \
  --anchor-weights $HOME/s6seed/weights/anatomy.pt --amp fp16 --chunk 192 --workers 2 --item-timeout 3600 \
  --assignment $IDS --output $O/out --tmp $HOME/r5_oai/tmp_s6_$NAME --dump-logits $O/out/logits.json --deadline $DL > $O/run.log 2>&1
rc=$?
rm -f $HOME/r5_oai/RUNNING_S6.txt; rm -rf $HOME/r5_oai/tmp_s6_$NAME
n=$(($(wc -l < $O/out/predictions.csv 2>/dev/null || echo 1) - 1))
echo "S6_R5_DONE $NAME rc $rc rows $n seconds $(( $(date +%s) - T0 )) $(date '+%m-%d %H:%M')" | tee $O/DONE.txt
