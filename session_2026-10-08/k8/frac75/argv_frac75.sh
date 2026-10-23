#!/bin/bash
# argv_frac75.sh (k8/frac75, 2026-10-07): THE argv of the frac75 runs, defined once. Sourced by launch_frac75.sh (real
# runs) and dry_run.sh (CPU validation), so the validated argv and the launched argv cannot drift apart.
# Trainer: ~/fast_build/code2/fleet_train.py (md5 3e62434a; with --grad_ckpt 1 it is the ~/bigjump_build/code trainer that
# ran J1, bit for bit: verify_perlabel.md row 14). Labels: ~/novel/frac75/labels_frac75_logit.csv (md5 ad7582b8...).
#   fold 0 = the PROBE, paired with J1: J1's argv (~/auto_j1.sh l.30, RECIPE + COMMON + --fold 0 --out) with ONLY --labels
#            and --out changed, plus an explicit --grad_ckpt 1 (the default). 24 epochs, seed 1043, save_epochs 6-24.
#   folds 1-4 (only after the probe PASSES G1-G3): 21 epochs (settled 10-07: 21 = 24 at 224), --grad_ckpt 0 (the fast
#            trainer path used for s1043 folds 2-4), seed 1043, save_epochs 6-21.
# Out dirs ~/fleet_runs/f{F}_s1043_ttr_frac75. Ckpt names inside equal set A's (f{F}_s1043_raw_ttr224-...): re-key at pack
# time (train_labels tag in the member key; build_k7.py l.79 asserts mix50 today).
FRAC75_LABELS=$HOME/novel/frac75/labels_frac75_logit.csv
FRAC75_COMMON=(--store $HOME/fleet_store --labels $FRAC75_LABELS --folds $HOME/rsna_train/folds5.csv
               --exclude $HOME/rsna_train/gold74_eval_set.csv --eval_labels $HOME/rsna_train/llm_labels_v4_blend.csv
               --gold $HOME/rsna_train/gold74_eval_set.csv)
frac75_argv(){   # $1 = fold -> sets the array FRAC75_ARGV and FRAC75_OUT
  local F=$1
  FRAC75_OUT=$HOME/fleet_runs/f${F}_s1043_ttr_frac75
  if [ "$F" = 0 ]; then
    FRAC75_ARGV=(--model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed 1043
                 --save_epochs 6-24 --workers 10 "${FRAC75_COMMON[@]}" --fold 0 --out $FRAC75_OUT --grad_ckpt 1)
  else
    FRAC75_ARGV=(--model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 21 --seed 1043
                 --save_epochs 6-21 --workers 10 "${FRAC75_COMMON[@]}" --fold $F --out $FRAC75_OUT --grad_ckpt 0)
  fi
}
