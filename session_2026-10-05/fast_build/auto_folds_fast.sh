#!/bin/bash
# auto_folds_fast.sh (2026-10-05): TTR mix50 folds 2-4 (seed 1043, the SAME recipe as J1 fold 0 and ~/auto_folds.sh)
# with the fast_build trainer ~/fast_build/code2/fleet_train.py --grad_ckpt 0 (no activation checkpointing; model math,
# recipe, out dirs, ckpt names and model_sha unchanged; ckpt args gain grad_ckpt=False). Same guards as ~/auto_folds.sh:
# before each fold waits while ~/fleet_runs/PAUSE_CHAIN exists and while any python compute app holds the GPU; skips a
# fold whose out dir already exists. Extra guard: also waits while the OLD chain (~/auto_folds.sh) is alive, so the two
# chains can never both launch a fold. Early-failure fallback: if a --grad_ckpt 0 fold exits non-zero within 45 min of
# launch (e.g. CUDA OOM at start), its out dir and log are moved aside (renamed, never deleted) and the fold is re-run
# once with --grad_ckpt 1 (= the checkpointed trainer, bit for bit). Never kills anything. Log ~/fleet_runs/auto_folds_fast.log.
# verify fixes 10-05: (1) the fallback re-run waits on the same 3 guards (PAUSE_CHAIN, GPU python, old chain) and re-checks
# that the out dir does not exist, so a deliberate kill + PAUSE_CHAIN is never overridden; (2) an nvidia-smi error counts as
# "GPU busy" (99), not "GPU free" (the old chain read it as 0 apps).
PY=~/miniconda3/envs/rsna/bin/python; LOG=~/fleet_runs/auto_folds_fast.log
say(){ echo "[auto-folds-fast $(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
old_chain(){ pgrep -f '^(/usr)?(/bin/)?(ba)?sh ([^ ]*/)?auto_folds\.sh$'; }   # exact old-chain cmdline, never this script
gpu_python(){ local o; o=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null) || { echo 99; return; }; echo "$o" | grep -c python; }   # verify fix 10-05: nvidia-smi error = busy, not free
COMMON="--store $HOME/fleet_store --labels $HOME/rsna_train/labels_mix50_logit.csv --folds $HOME/rsna_train/folds5.csv --exclude $HOME/rsna_train/gold74_eval_set.csv --eval_labels $HOME/rsna_train/llm_labels_v4_blend.csv --gold $HOME/rsna_train/gold74_eval_set.csv"
RECIPE="--model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed 1043 --save_epochs 6-24 --workers 10"
cd ~/fast_build/code2 || exit 1
say "chain start (pid $$): folds 2 3 4 with --grad_ckpt 0; old chain pids: [$(old_chain | tr '\n' ' ')]"
for F in 2 3 4; do
  w=0
  while [ -e ~/fleet_runs/PAUSE_CHAIN ] || [ "$(gpu_python)" -ne 0 ] || [ -n "$(old_chain)" ]; do
    [ $w -eq 0 ] && say "fold $F: waiting (PAUSE_CHAIN=$([ -e ~/fleet_runs/PAUSE_CHAIN ] && echo yes || echo no), gpu python=$(gpu_python), old chain=[$(old_chain | tr '\n' ' ')])"
    w=1; sleep 120
  done
  OUT=~/fleet_runs/f${F}_s1043_ttr_mix50
  if [ -e $OUT ]; then say "fold $F: $OUT exists, skipping"; continue; fi
  say "fold $F: launching (grad_ckpt 0) -> $OUT"
  T0=$(date +%s)
  $PY fleet_train.py $RECIPE $COMMON --fold $F --out $OUT --grad_ckpt 0 > $OUT.log 2>&1
  RC=$?; DT=$(( ($(date +%s) - T0) / 60 ))
  say "fold $F: exited rc=$RC after ${DT} min ($(tail -n 1 $OUT.log | cut -c1-120))"
  if [ $RC -ne 0 ] && [ $DT -lt 45 ]; then
    AS=${OUT}_fastfail_$(date +%m%d_%H%M)
    [ -e $OUT ] && mv $OUT $AS; mv $OUT.log $AS.log
    say "fold $F: early failure of the grad_ckpt 0 run, moved aside to $AS; re-running with --grad_ckpt 1"
    w=0                                                   # verify fix 10-05: the fallback honours the same 3 guards
    while [ -e ~/fleet_runs/PAUSE_CHAIN ] || [ "$(gpu_python)" -ne 0 ] || [ -n "$(old_chain)" ]; do
      [ $w -eq 0 ] && say "fold $F: fallback waiting (PAUSE_CHAIN=$([ -e ~/fleet_runs/PAUSE_CHAIN ] && echo yes || echo no), gpu python=$(gpu_python), old chain=[$(old_chain | tr '\n' ' ')])"
      w=1; sleep 60
    done
    if [ -e $OUT ]; then say "fold $F: $OUT exists, fallback skipped"; continue; fi
    $PY fleet_train.py $RECIPE $COMMON --fold $F --out $OUT --grad_ckpt 1 > $OUT.log 2>&1
    RC=$?
    say "fold $F: fallback (grad_ckpt 1) exited rc=$RC ($(tail -n 1 $OUT.log | cut -c1-120))"
  fi
done
say "chain done"
