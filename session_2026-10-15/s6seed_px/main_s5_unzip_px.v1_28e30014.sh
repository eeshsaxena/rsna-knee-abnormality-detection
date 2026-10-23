#!/bin/bash
# S5 (MAIN ONLY, after S6 GO and PX_PARITY PASS). Unzip the competition archive, then build the S6fall px cache for all 4,407
# train studies. Cores 12-27, nice 19, ionice idle, CUDA hidden, RAM guard 60 GB (px_build_s6 --mem-cap-gb; worker peak RSS MEASURED
# 0.9-2.2 GB per worker, typical 1.3). Never touches the GPU, ~/queue, fleet_train.py. Resumable: re-run the same step after any stop.
# Usage (arm with setsid nohup, see the bottom):
#   PARTS="$HOME/rsna_comp/part.0 ... $HOME/rsna_comp/part.7"   (byte-range parts IN BYTE ORDER, or the single archive.zip)
#   DATA=$HOME/rsna_comp/data   (extraction root; must end up holding train.csv, train_series.csv, train_series/<uid>/<series>/*.dcm)
set -u
: "${PARTS:?set PARTS to the ordered part files (or the one archive)}"
DATA=${DATA:-$HOME/rsna_comp/data}
PX=${PX:-$HOME/s6seed/px320}
PY=$HOME/miniconda3/envs/kmimic/bin/python
PROCS=${PROCS:-16}; THREADS=${THREADS:-1}   # 16 x 1 (bytes == the 2-thread PX_PARITY cache, MEASURED); 8 x 2 = the exact parity config
PFX="taskset -c 12-27 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS"
L=$HOME/s6seed/logs
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "S5 step 1: central directory check"
$PFX $PY -u $HOME/s6seed/unzip_parts.py --parts $PARTS --out "$DATA" --list-only || { log "list-only FAILED (part order / incomplete download?)"; exit 1; }

log "S5 step 2: unzip, 6 sequential streams"
$PFX $PY -u $HOME/s6seed/unzip_parts.py --parts $PARTS --out "$DATA" --procs 6 > $L/unzip_full.log 2>&1
tail -1 $L/unzip_full.log
grep -q "^UNZIP OK" $L/unzip_full.log || { log "unzip FAILED: see $L/unzip_full.log (re-run resumes)"; exit 1; }
for f in train.csv train_series.csv; do [ -s "$DATA/$f" ] || { log "missing $DATA/$f (archive root nested? set DATA to the folder that holds train.csv)"; exit 1; }; done
n=$(ls "$DATA/train_series" | wc -l); log "train_series study dirs: $n (expect 4407)"
[ "$n" -eq 4407 ] || { log "study count != 4407"; exit 1; }

log "S5 step 3: px build, $PROCS procs x $THREADS threads"
$PFX $PY -u $HOME/s6seed/px_build_s6.py --data "$DATA" --split train --src $HOME/s6seed/src --code $HOME/s6seed/code \
  --anchor $HOME/s6seed/weights/anatomy.pt --out $PX --procs $PROCS --threads $THREADS --mem-cap-gb 60 --log-every 25 --progress-s 600 \
  > $L/px_build_full.log 2>&1
tail -2 $L/px_build_full.log
$PFX $PY -u $HOME/s6seed/px_build_s6.py --out $PX --finalize | tee -a $L/px_build_full.log
log "S5 done: $PX (index.json + summary.json); errors are listed in index.json 'errors' (the trainer leaves them out of the pool)"
# arm:  PARTS="..." DATA=... setsid nohup bash ~/s6seed/main_s5_unzip_px.sh > ~/s6seed/logs/main_s5.log 2>&1 < /dev/null &
