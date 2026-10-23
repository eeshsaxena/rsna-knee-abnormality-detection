#!/bin/bash
# PX_PARITY part 2 (replaces the tail of run_parity.sh): the train side split into 6 parallel subsets (2 start at once, 4 when the
# serve shards end; never more than 6 processes of ours), caches merged, then the compare, then the convnext_small spot check.
set -u
cd ~/s6seed/parity
PY=~/miniconda3/envs/kmimic/bin/python
PX=~/s6seed/px320_gold58
PFX="taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
MD=$PWD/surrogate_convnext_atto; BB=convnext_atto
log() { echo "[$(date +%H:%M:%S)] $*"; }
tail -n +2 ~/kmimic/ids/gold58.csv | cut -d, -f1 | sort > shards/gold58_sorted.lst
for k in 0 1 2 3 4 5; do awk -v k=$k 'NR % 6 == k' shards/gold58_sorted.lst > shards/tr$k.lst; done
wc -l shards/tr*.lst | tail -1
train() {
  rm -f tc_$BB.$1.json
  $PFX $PY -u px_parity.py --src ../src --px $PX --models $MD --backbone $BB --uids shards/tr$1.lst --threads 2 \
    --train-cache tc_$BB.$1.json --out train_only_$BB.$1.json > train_side_$BB.$1.log 2>&1
  log "train subset $1 rc $?"
}
train 0 & train 1 &
log "waiting for the serve shards"
while pgrep -f "ci_cpu_launcher.py --data" > /dev/null; do sleep 15; done
log "serve shards done"
train 2 & train 3 & train 4 & train 5 &
wait
$PY - <<EOF
import json
m = {}
for k in range(6):
    m.update(json.load(open("tc_$BB.%d.json" % k)))
json.dump(m, open("train_side_$BB.json", "w"))
print("merged train-side cache:", len(m), "studies")
EOF
REFS="ref_atto_s0 ref_atto_s1 ref_atto_s2 ref_atto_s3"
$PFX $PY -u px_parity.py --src ../src --px $PX --models $MD --backbone $BB --ref $REFS --uids ~/kmimic/ids/gold58.csv \
  --train-cache train_side_$BB.json --out PX_PARITY_gold58_$BB.json > px_parity_$BB.log 2>&1
tail -1 px_parity_$BB.log

log "spot check: convnext_small surrogate on 1 study (serve and train side in parallel)"
head -1 shards/gold58_sorted.lst > shards/small.lst
$PY -c "import json; print(json.dumps(open('shards/small.lst').read().split()))" > shards/small.json
rm -f train_side_small.json
$PFX $PY -u px_parity.py --src ../src --px $PX --models $PWD/surrogate_convnext_small --backbone convnext_small \
  --uids shards/small.lst --threads 4 --train-cache train_side_small.json --out train_only_small.json > train_side_small.log 2>&1 &
mkdir -p ref_small/pxdump
K1_PX_DUMP=$PWD/ref_small/pxdump S6_SRC=$HOME/s6seed/src S6_CODE=$HOME/s6seed/code $PFX $PY -u ci_cpu_launcher.py \
  --data $HOME/kmimic/kaggle/input/rsna-knee-abnormality-detection --split train --code $HOME/s6seed/code \
  --models $PWD/surrogate_convnext_small --arm E_prior --backbone convnext_small --size 320 --crop 288 --views v4 --slices 3 \
  --phys-mm 3.5 --eval-stride 1 --anchor-weights $HOME/s6seed/weights/anatomy.pt --amp fp32 --chunk 192 --workers 0 \
  --item-timeout 900 --assignment shards/small.json --output ref_small --tmp /dev/shm/s6par_small --dump-logits ref_small/logits.json \
  > ref_small.log 2>&1
wait
$PFX $PY -u px_parity.py --src ../src --px $PX --models $PWD/surrogate_convnext_small --backbone convnext_small --ref ref_small \
  --uids shards/small.lst --threads 4 --train-cache train_side_small.json --out PX_PARITY_small_convnext_small.json \
  > px_parity_convnext_small.log 2>&1
tail -1 px_parity_convnext_small.log
log "PARITY_DRIVER_DONE"
