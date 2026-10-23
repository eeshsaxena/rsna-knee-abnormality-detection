#!/bin/bash
# CPU-only pipeline probe of the R7 trainer (compact_train v9.8, md5 8622009e) on the 5-study test cache. fm-aux off (no fastMRI store).
cd ~/r7port
P=~/miniconda3/envs/kmimic/bin/python
SRC=${SRC:-src_cpu}
OUT=${OUT:-pipe_test/run1}
COMMON="--px pipe_test/px --targets pipe_test/targets.csv --extra pipe_test/cell_weights.csv --out $OUT --arms E_lrprior=lr:lr_prior:w
 --backbone convnext_small --init pretrained/convnext_small_in22k_ft_in1k.safetensors --seeds 1 --folds 4 --epochs ${EPOCHS:-2} --batch 2
 --train-windows ${TW:-16} --crop 352 --slices 3 --slice-jitter 0.25 --ema 0.999 --include-review-holdout --syn-decomp --phys-mm 3.5
 --rare-boost --lr 1e-4 --freeze-blocks 16 --llrd 1.0 --head-lr 1e-3 --wd 0.05 --workers 2 --eval-workers 1 --ckpt-every 0
 --eval-stride ${ES:-12} --train-sampling cover --cover-stride 3 --state-targets pipe_test/state_targets_SYNTHETIC.csv
 --state-spec bundle_r7/state_spec.json --state-weight 0.5 --spatial-attn --pair-rank 0.5 --pair-frac 0.33 --save-state-oof
 --seed-base 0 --extent --extent-weight 0.2 --mirror 0.5 --aug v2 --aug-grid-fp32 --draw-seed --rare-boost-names
 --only-folds 0 --i-have-user-authorisation-to-finetune"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec nice -n 19 taskset -c 0-7 $P -u $SRC/compact_train.py $COMMON "$@"
