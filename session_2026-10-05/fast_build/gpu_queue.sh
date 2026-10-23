#!/bin/bash
# gpu_queue.sh -- fast_build FB-1 checks, ONE GPU process at a time, each under flock ~/jump2/gpu.lock, after the TTA
# bench (pid 322072) has exited. Every python job gates on free GPU memory and caps itself
# (set_per_process_memory_fraction) so the running fold trainer can never OOM. Never touches any other process.
FB=$HOME/fast_build; PY=$HOME/miniconda3/envs/rsna/bin/python; L=$FB/logs2; LOG=$L/queue.log
mkdir -p $L $FB/parity $FB/tiny
say(){ echo "[fbq $(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
export PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1
while kill -0 322072 2>/dev/null; do sleep 15; done
say "TTA bench pid 322072 gone; starting"
gpu(){ local name=$1 tmo=$2; shift 2; say "start $name"; flock $HOME/jump2/gpu.lock nice -n 5 timeout $tmo "$@" > $L/$name.log 2>&1; local rc=$?; say "end $name rc=$rc"; return $rc; }
REC="--model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 2 --seed 1043 --save_epochs 1-2 --workers 4 --store $HOME/fleet_store --labels $FB/tiny/labels_tiny.csv --folds $HOME/rsna_train/folds5.csv --exclude $HOME/rsna_train/gold74_eval_set.csv --eval_labels $HOME/rsna_train/llm_labels_v4_blend.csv --fold 2"
CAP="$PY $FB/cap_run.py --cap_gib 9.8 --need_mib 12000"

# 1. unit parity (2 real studies, 4 applied steps, accum 1): trainer cudnn mode, then deterministic mode
gpu parity_bench 900 $PY $FB/parity_unit.py --mode bench --out $FB/parity
gpu parity_det 900 env CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY $FB/parity_unit.py --mode det --out $FB/parity
# 2. end-to-end tiny trainer runs (deterministic): pristine ref, code2 default, code2 --grad_ckpt 0
for R in ref_det c2def_det c2gc0_det; do
  case $R in
    ref_det)   CODE=$FB/code_ref; X="";;
    c2def_det) CODE=$FB/code2;    X="";;
    c2gc0_det) CODE=$FB/code2;    X="--grad_ckpt 0";;
  esac
  rm -rf $FB/tiny/$R
  gpu tiny_$R 900 env CUBLAS_WORKSPACE_CONFIG=:4096:8 $CAP --det -- $CODE/fleet_train.py $REC $X --out $FB/tiny/$R
  cp $L/tiny_$R.log $FB/tiny/$R.log
done
$PY $FB/compare_tiny.py ref_det:c2def_det c2def_det:c2gc0_det > $L/compare_tiny.log 2>&1; say "compare_tiny rc=$?"
# 3. worker strict-load (CPU): parity ckpts + the code2 --grad_ckpt 0 trainer's own saved ckpts
$PY $FB/worker_load_check.py $FB/parity/f2_s1043_raw_ttr224-320x24_gc0bench_best.pt $FB/parity/f2_s1043_raw_ttr224-320x24_gc1bench_best.pt \
  $(ls $FB/tiny/c2gc0_det/f2_s1043_*_best.pt $FB/tiny/c2gc0_det/epochs/f2_s1043_*_ep2_best.pt 2>/dev/null) > $L/worker_check.log 2>&1; say "worker_check rc=$?"
# 4. speed A/B on the real recipe while fold 1 shares the GPU
gpu speed_ab 3000 $PY $FB/speed_ab.py --log $L/speed_ab.jsonl
say "queue done"
