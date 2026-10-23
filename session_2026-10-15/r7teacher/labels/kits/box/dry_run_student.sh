#!/bin/bash
# dry_run_student.sh (r7teacher/labels, 2026-10-14, A1; box, CPU only, nice 19 / ionice idle, cores ${CORES:-34-39}): validates the
# student argv against the REAL trainer ~/fast_build/code2/fleet_train.py with the SIMULATED T_b / T_a tables. The argv is
# read from the SIMCHECK sheet line (the same generator as the real sheets), with ONLY these overrides APPENDED
# (argparse: last occurrence wins): --store ~/novel/frac75/dry/store (31 symlinked studies, read only), --out
# ~/r7t_labels/dry/<run>, --device cpu, --min_cover 0, --time_epochs 1. GPU hidden, HF offline, no .pyc writes.
# A1 FX4: after each run it prints the checkpoint's args (sel_labels must == train_labels == the SIM table name).
set -u
PY=~/miniconda3/envs/rsna/bin/python; D=$HOME/r7t_labels/dry; mkdir -p $D
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
cd ~/fast_build/code2 || exit 1
for T in b a; do
  line=$(sed -n 2p $HOME/r7t_labels/sheets/queue_r7t${T}_SIMCHECK.txt)          # fold 1 line
  argv=$(printf '%s' "${line##*|}" | sed -e 's/^[[:space:]]*//' -e "s#~/#$HOME/#g")
  run=$(printf '%s' "$argv" | sed -E 's#.*--out [^ ]*/([^ ]+).*#\1#')
  OUT=$D/$run; rm -rf $OUT
  echo "[dry T_$T $(date +%H:%M:%S)] trainer md5 $(md5sum fleet_train.py | cut -c1-32) | argv: $argv || overrides: --store ~/novel/frac75/dry/store --out $OUT --device cpu --min_cover 0 --time_epochs 1" >> $D/dry_run.log
  /usr/bin/time -v taskset -c ${CORES:-34-39} nice -n 19 ionice -c3 $PY fleet_train.py $argv \
    --store $HOME/novel/frac75/dry/store --out $OUT --device cpu --min_cover 0 --time_epochs 1 > $OUT.log 2>&1
  echo "[dry T_$T $(date +%H:%M:%S)] rc=$? first: $(grep -m1 '^labels' $OUT.log) | last: $(grep '^{' $OUT.log | tail -n 1)" >> $D/dry_run.log
  echo "[dry T_$T ckpt args] $($PY -c "import glob, torch; c = sorted(glob.glob('$OUT/f1_s*_raw_*_best.pt')); a = torch.load(c[0], map_location='cpu', weights_only=False)['args'] if c else {}; print(len(c), 'ckpt', {k: a.get(k) for k in ('fold', 'seed', 'epochs', 'ttr_size', 'train_labels', 'sel_labels', 'grad_ckpt')})" 2>&1 | tail -n 1)" >> $D/dry_run.log
done
echo "[dry done $(date +%H:%M:%S)]" >> $D/dry_run.log
