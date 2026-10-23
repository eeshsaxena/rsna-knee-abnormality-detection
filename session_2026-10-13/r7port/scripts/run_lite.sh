#!/bin/bash
# CPU forward-only checks on the merged 5-study cache with OUR targets: (1) R7-lite (state head off), (2) R7-lite + --grad-ckpt
cd ~/r7port
P=~/miniconda3/envs/kmimic/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
LITE="--px px_all_test --targets targets_ours/targets.csv --extra targets_ours/cell_weights.csv --arms E_lrprior=lr:lr_prior:w
 --backbone convnext_small --init pretrained/convnext_small_in22k_ft_in1k.safetensors --seeds 1 --folds 4 --epochs 16 --batch 2
 --train-windows 16 --crop 352 --slices 3 --slice-jitter 0.25 --ema 0.999 --include-review-holdout --syn-decomp --phys-mm 3.5
 --lr 1e-4 --freeze-blocks 16 --llrd 1.0 --head-lr 1e-3 --wd 0.05 --workers 2 --eval-workers 1 --ckpt-every 0 --eval-stride 1
 --train-sampling cover --cover-stride 3 --spatial-attn --seed-base 0 --mirror 0.5 --aug v2 --aug-grid-fp32 --draw-seed
 --only-folds 0 --i-have-user-authorisation-to-finetune --forward-only"
echo "== lite"; nice -n 19 taskset -c 0-7 $P -u src_cpu/compact_train.py $LITE --out pipe_test/lite_fwd 2>&1 | grep -v "^v9" | tail -4
echo "== lite + grad-ckpt"; nice -n 19 taskset -c 0-7 $P -u src_cpu/compact_train.py $LITE --grad-ckpt --out pipe_test/lite_gc_fwd 2>&1 | grep -v "^v9" | tail -3
echo "== check-targets on our full targets (pool = studies present in the cache)"; nice -n 19 taskset -c 0-7 $P -u src/compact_train.py $LITE --out pipe_test/ct --check-targets 2>&1 | grep -i "pool\|check-targets" | cut -c1-250
