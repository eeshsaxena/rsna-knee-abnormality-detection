#!/bin/bash
# PX_PARITY driver (10-15, BOX_PLAN_1015 S4). Waits for the gold58 px build, finalizes it, then:
#  (1) serve side: compact_infer 34f433fe main() end to end on the 58 gold DICOMs (ci_cpu_launcher.py, CPU fp32), NSH shards x 1 proc,
#      model = MODEL_DIR (surrogate convnext_atto unless nartaa's E_prior_s42_fall.pt is given), window hashes + logits dumped;
#      in parallel the train side (px_parity.py --train-cache: px_build_s6 cache read as compact_train evaluates, same model)
#  (2) compare -> PX_PARITY PASS / FAIL
#  (3) spot check with the real S6fall architecture (convnext_small surrogate, pretrained trunk 2457488b) on SMALL_N studies
# CPU only, cores 28-39, nice 19, at most 6 processes of ours at any time.
set -u
cd ~/s6seed/parity
PY=~/miniconda3/envs/kmimic/bin/python
DATA=~/kmimic/kaggle/input/rsna-knee-abnormality-detection
PX=~/s6seed/px320_gold58
PFX="taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
NSH=${NSH:-5}
SMALL_N=${SMALL_N:-1}
export NSH SMALL_N
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for the px build"
while pgrep -f "px_build_s6.py --data" > /dev/null; do sleep 20; done
$PFX $PY ../px_build_s6.py --out $PX --finalize || { log "finalize failed"; exit 1; }

$PY - <<'EOF'
import csv, json, os
u = [r["StudyInstanceUID"] for r in csv.DictReader(open(os.path.expanduser("~/kmimic/ids/gold58.csv")))]
n = int(os.environ.get("NSH", "5"))
os.makedirs("shards", exist_ok=True)
for k in range(n):
    json.dump(u[k::n], open("shards/s%d.json" % k, "w"))
json.dump(u[:int(os.environ.get("SMALL_N", "1"))], open("shards/small.json", "w"))
print("shards", [len(u[k::n]) for k in range(n)])
EOF

serve() {   # tag backbone modeldir assignment
  local tag=$1 bb=$2 md=$3 asg=$4
  mkdir -p ref_$tag/pxdump
  K1_PX_DUMP=$PWD/ref_$tag/pxdump S6_SRC=$HOME/s6seed/src S6_CODE=$HOME/s6seed/code $PFX $PY -u ci_cpu_launcher.py \
    --data $DATA --split train --code $HOME/s6seed/code --models $md --arm E_prior --backbone $bb --size 320 --crop 288 --views v4 \
    --slices 3 --phys-mm 3.5 --eval-stride 1 --anchor-weights $HOME/s6seed/weights/anatomy.pt --amp fp32 --chunk 192 --workers 0 \
    --item-timeout 900 --assignment $asg --output ref_$tag --tmp /dev/shm/s6par_$tag --dump-logits ref_$tag/logits.json \
    > ref_$tag.log 2>&1
  log "serve $tag rc $?"
}

MD=${MODEL_DIR:-$PWD/surrogate_convnext_atto}
BB=${MODEL_BACKBONE:-convnext_atto}
log "serve side ($BB, $MD), $NSH shards + train side (1 proc) in parallel"
rm -f train_side_$BB.json
$PFX $PY -u px_parity.py --src ../src --px $PX --models $MD --backbone $BB --uids ~/kmimic/ids/gold58.csv --threads 3 \
  --train-cache train_side_$BB.json --out train_only_$BB.json > train_side_$BB.log 2>&1 &
for k in $(seq 0 $((NSH-1))); do serve atto_s$k $BB $MD shards/s$k.json & done
wait
log "compare"
REFS=$(for k in $(seq 0 $((NSH-1))); do echo -n "ref_atto_s$k "; done)
$PFX $PY -u px_parity.py --src ../src --px $PX --models $MD --backbone $BB --ref $REFS --uids ~/kmimic/ids/gold58.csv \
  --train-cache train_side_$BB.json --out PX_PARITY_gold58_$BB.json > px_parity_$BB.log 2>&1
tail -1 px_parity_$BB.log

log "spot check: convnext_small surrogate on $SMALL_N studies"
$PY -c "import json; print('\n'.join(json.load(open('shards/small.json'))))" > shards/small.lst
rm -f train_side_small.json
$PFX $PY -u px_parity.py --src ../src --px $PX --models $PWD/surrogate_convnext_small --backbone convnext_small \
  --uids shards/small.lst --threads 4 --train-cache train_side_small.json --out train_only_small.json > train_side_small.log 2>&1 &
serve small convnext_small $PWD/surrogate_convnext_small shards/small.json
wait
$PFX $PY -u px_parity.py --src ../src --px $PX --models $PWD/surrogate_convnext_small --backbone convnext_small --ref ref_small \
  --uids shards/small.lst --threads 4 --train-cache train_side_small.json --out PX_PARITY_small_convnext_small.json \
  > px_parity_convnext_small.log 2>&1
tail -1 px_parity_convnext_small.log
log "PARITY_DRIVER_DONE"
