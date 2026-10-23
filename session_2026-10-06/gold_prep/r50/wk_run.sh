#!/bin/bash
# wk_run.sh (gold_prep/r50, CPU, nice 19): per tag (raw, ema; the worker serves ONE tag per run), stage + run the UNCHANGED
# K5 fleet_worker.py (kmimic env = Kaggle-like timm) on the R50 dry-run checkpoint, then compare served pass-A rows with
# the trainer's OOF rows for the same 3 val studies (served from DICOM via the mirror; the trainer read the store).
RUN=${1:-$HOME/gold_prep/r50/dryA}; G=$HOME/gold_prep/r50; PYK=~/miniconda3/envs/kmimic/bin/python
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
: > $G/wk_run.log
for TAG in raw ema; do
  echo "=== tag $TAG" >> $G/wk_run.log
  nice -n 19 $PYK -W ignore $G/wk_stage.py $RUN $TAG >> $G/wk_run.log 2>&1 || { echo "WK STAGE FAIL" >> $G/wk_run.log; continue; }
  (cd $G/wk && FLEET_ALLOW_CPU=1 FLEET_DIR=$G/wk/ds FLEET_OUT=$G/wk_out_$TAG.npz RSNA_COMP_ROOT=$G/wk/comp \
    nice -n 19 $PYK -W ignore $G/wk/ds/fleet_worker.py >> $G/wk_run.log 2>&1); echo "WORKER $TAG rc=$?" >> $G/wk_run.log
  mv -f $G/wk_out_$TAG.npz $G/wk/out_$TAG.npz 2>/dev/null
  nice -n 19 $PYK -W ignore $G/wk_compare.py $RUN $TAG >> $G/wk_run.log 2>&1
done
echo "WK DONE" >> $G/wk_run.log
