#!/bin/bash
# kc_shim_regress.sh -- BOX_PLAN_1015 sec 3.3 step 4 regression (b)/(c) + (a). Run by MAIN (it starts fleet_train.py CPU
# processes, which make the queue runner wait): ONLY while T_a f3 holds the GPU and finished by 15:20. Cores 28-39,
# torch threads 12, GPU hidden, HF offline, no .pyc writes. MEASURED walls: (b) ~4-6 min, (c) ~3 min.
#   bash ~/kc/code/shim/kc_shim_regress.sh pre    optional, BEFORE install_shim.sh: fresh pre-shim (b) reference
#   bash ~/kc/code/shim/kc_shim_regress.sh post   AFTER install_shim.sh: (a) + (b) + (c) -> ~/kc/regress/REGRESS.txt
# (b) = the PREREG_TEACHER A1 T_a dry-run form (sheet queue_r7ta_SIMCHECK.txt line 2, md5 300faa2d; overrides --store
#     ~/novel/frac75/dry/store (31 studies) --device cpu --min_cover 0 --time_epochs 1). Reference = ~/kc/regress/b_pre if
#     present, else the 10-14 14:42 pre-shim run ~/r7t_labels/dry/f1_s1048_ttr_r7tadry (trainer 3e62434a). Bar: ckpt args
#     identical. (c) = the C4 smoke argv with --kc through fleet_train.py (store ~/kc_store/test458, smoke folds).
set -u
MODE=${1:-post}
R=$HOME/kc/regress; mkdir -p $R
CODE=$HOME/fast_build/code2; T=fleet_train.py
PY=$HOME/miniconda3/envs/rsna/bin/python
RUN="taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY"
line=$(sed -n 2p $HOME/r7t_labels/sheets/queue_r7ta_SIMCHECK.txt)
argv=$(printf '%s' "${line##*|}" | sed -e 's/^[[:space:]]*//' -e "s#~/#$HOME/#g")
OVR="--store $HOME/novel/frac75/dry/store --device cpu --min_cover 0 --time_epochs 1"
md5(){ md5sum < "$1" | cut -c1-32; }
if [ "$MODE" = pre ]; then
  [ "$(md5 $CODE/$T)" = 3e62434adea4fd8971af7c96e0465773 ] || { echo "pre: the trainer is not 3e62434a (shim already installed?)"; exit 2; }
  [ -e $R/b_pre ] && { echo "pre: $R/b_pre exists"; exit 2; }
  t0=$(date +%s); ( cd $CODE && $RUN $T $argv $OVR --out $R/b_pre ) > $R/b_pre.log 2>&1; rc=$?
  echo "pre rc=$rc $(( $(date +%s) - t0 )) s | $(tail -n 1 $R/b_pre.log)"; exit $rc
fi
OUTF=$R/REGRESS.txt; [ -e $OUTF ] && mv $OUTF $OUTF.prev_$(date +%m%d_%H%M%S)
log(){ echo "$*" | tee -a $OUTF; }
log "kc_shim_regress $(date '+%F %T') | trainer md5 $(md5 $CODE/$T) | fleet_train_kc.py $(md5 $CODE/fleet_train_kc.py) kc_common.py $(md5 $CODE/kc_common.py) kc_g1.py $(md5 $CODE/kc_g1.py)"
ok=1
$PY $HOME/kc/code/shim/kc_regress_cmp.py shim $CODE/$T $HOME/kc/code/fleet_train.py.bak_3e62434a | tee -a $OUTF; [ ${PIPESTATUS[0]} -eq 0 ] || ok=0
REF=$R/b_pre; [ -d $REF ] || REF=$HOME/r7t_labels/dry/f1_s1048_ttr_r7tadry
rm -rf $R/b_post $R/b_post.log
t0=$(date +%s); ( cd $CODE && $RUN $T $argv $OVR --out $R/b_post ) > $R/b_post.log 2>&1; rc=$?
log "(b) run rc=$rc $(( $(date +%s) - t0 )) s | last: $(tail -n 1 $R/b_post.log) | reference $REF"
[ $rc -eq 0 ] || ok=0
$RUN $HOME/kc/code/shim/kc_regress_cmp.py args $REF $R/b_post | tee -a $OUTF; [ ${PIPESTATUS[0]} -eq 0 ] || ok=0
rm -rf $R/c_kc $R/c_kc.log $R/c_test.verdict $R/c_test.verdict.json
t0=$(date +%s)
( cd $CODE && $RUN $T --kc --device cpu --max_steps 2 --threads 12 --workers 4 --model ttr --ttr_size 256 \
    --backbone convnext_small.fb_in22k_ft_in1k --pretrained_file $HOME/r7port/pretrained/convnext_small_in22k_ft_in1k.safetensors \
    --mirror 0 --vbias_prior 1 --aug heavy --train_slices 8 --bs 1 --accum 2 --seed 2000 --fold 0 --min_cover 0 \
    --save_epochs 1-1 --estar_range 1-1 --store $HOME/kc_store/test458 --labels $HOME/r7t_labels/real/labels_r7t_b.csv \
    --folds $HOME/kc/smoke/folds_smoke.csv --exclude $HOME/rsna_train/gold74_eval_set.csv \
    --eval_labels $HOME/r7t_labels/real/labels_r7t_b.csv --gold $HOME/rsna_train/gold74_eval_set.csv \
    --gate_out $R/c_test.verdict --out $R/c_kc --grad_ckpt 1 ) > $R/c_kc.log 2>&1; rc=$?
log "(c) run rc=$rc $(( $(date +%s) - t0 )) s | last: $(tail -n 1 $R/c_kc.log)"
[ $rc -eq 0 ] && tail -n 1 $R/c_kc.log | grep -Eq '^\{"raw": [^,]+, "ema": [^}]+\}$' || ok=0
$RUN $HOME/kc/code/shim/kc_regress_cmp.py kc $R/c_kc $R/c_test.verdict | tee -a $OUTF; [ ${PIPESTATUS[0]} -eq 0 ] || ok=0
log "REGRESS $([ $ok = 1 ] && echo PASS || echo FAIL) trainer_md5=$(md5 $CODE/$T)"
