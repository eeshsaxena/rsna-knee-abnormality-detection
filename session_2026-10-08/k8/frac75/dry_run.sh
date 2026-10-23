#!/bin/bash
# dry_run.sh (k8/frac75, CPU only, nice 19 / ionice idle): validates the frac75 argv against the REAL trainer
# ~/fast_build/code2/fleet_train.py. The argv comes from argv_frac75.sh (the same arrays launch_frac75.sh runs), verbatim,
# with ONLY these overrides APPENDED (argparse: last occurrence wins): --store ~/novel/frac75/dry/store (31 symlinked
# studies), --out ~/novel/frac75/dry/f{F}_s1043_ttr_frac75, --device cpu, --min_cover 0, --time_epochs 1 (stop after
# epoch 1; --epochs and the cosine schedule stay as launched). F = 0 (probe argv, gc1) and F = 1 (folds argv, gc0).
# GPU hidden, HF offline, no .pyc writes into ~/fast_build/code2. Writes ~/novel/frac75/dry only.
source $HOME/novel/frac75/argv_frac75.sh
PY=~/miniconda3/envs/rsna/bin/python; D=$HOME/novel/frac75/dry
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd ~/fast_build/code2 || exit 1
echo "[dry $(date +%H:%M:%S)] trainer md5 $(md5sum fleet_train.py | cut -c1-32) | labels md5 $(md5sum $FRAC75_LABELS | cut -c1-32)" >> $D/dry_run.log
for F in 0 1; do
  frac75_argv $F
  OUT=$D/f${F}_s1043_ttr_frac75; rm -rf $OUT
  echo "[dry f$F $(date +%H:%M:%S)] argv: ${FRAC75_ARGV[*]} || overrides: --store $D/store --out $OUT --device cpu --min_cover 0 --time_epochs 1" >> $D/dry_run.log
  /usr/bin/time -v nice -n 19 ionice -c3 $PY fleet_train.py "${FRAC75_ARGV[@]}" \
    --store $D/store --out $OUT --device cpu --min_cover 0 --time_epochs 1 > $OUT.log 2>&1
  echo "[dry f$F $(date +%H:%M:%S)] rc=$? last: $(grep '^{' $OUT.log | tail -n 1)" >> $D/dry_run.log
done
echo "[dry done $(date +%H:%M:%S)]" >> $D/dry_run.log
