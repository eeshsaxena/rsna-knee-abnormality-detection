#!/bin/bash
# S5 v2 (MAIN ONLY, after S6 GO and PX_PARITY PASS; v1 = main_s5_unzip_px.v1_28e30014.sh). Unzip the competition archive, then
# build the S6fall px cache for all 4,407 train studies. Cores 12-27, nice 19, ionice idle, CUDA hidden, RAM guard 60 GB
# (px_build_s6 --mem-cap-gb; worker peak RSS MEASURED 0.9-2.2 GB, typical 1.3). Never touches the GPU, ~/queue, the queue trainer.
# Resumable: re-run the same command after any stop.
# v2 (10-15 review S6R-2): the build's exit code is checked (3 = MEM_CAP stop, 4 = a worker died, other = crash): NO finalize on a
# failed build; per-study errors get ONE retry pass (px_build retries earlier errors on a re-run); after --finalize a completeness
# check (train.csv studies vs done / errors / missing, check-targets pool and gold on CPU) writes ~/s6seed/gates/PX_BUILD.verdict
# (PASS only with gold 58, pool >= 4,328 = 4,349 x (1 - .005), 0 missing) and gates/PX_ERRORS.txt. Main copies the pool number of
# a PASS verdict into Amendment 7 / gates/A7.verdict ('pool <N>'); launch_when_free.sh refuses any other pool.
# Usage (arm with setsid nohup, see the bottom):
#   PARTS="$HOME/rsna_comp/part.0 ... $HOME/rsna_comp/part.7"   (byte-range parts IN BYTE ORDER, or the single archive.zip)
#   DATA=$HOME/rsna_comp/data   (extraction root; must end up holding train.csv, train_series.csv, train_series/<uid>/<series>/*.dcm)
set -u
: "${PARTS:?set PARTS to the ordered part files (or the one archive)}"
KIT=${S5_KIT:-$HOME/s6seed}
DATA=${DATA:-$HOME/rsna_comp/data}
PX=${PX:-$KIT/px320}
PY=$HOME/miniconda3/envs/kmimic/bin/python
PYT=$HOME/miniconda3/envs/rsna/bin/python
PROCS=${PROCS:-16}; THREADS=${THREADS:-1}   # 16 x 1 (bytes == the 2-thread PX_PARITY cache, MEASURED); 8 x 2 = the exact parity config
CORES=${S5_CORES:-12-27}
POOL_FLOOR=${S5_POOL_FLOOR:-4328}
PFX="taskset -c $CORES nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS"
L=$KIT/logs; G=$KIT/gates; mkdir -p "$L" "$G"
V=$G/PX_BUILD.verdict
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "FAIL $*"; echo "FAIL $(date '+%F %T') $*" > "$V"; exit "${2:-1}"; }
die2() { local rc=$1; shift; log "FAIL $*"; echo "FAIL $(date '+%F %T') $*" > "$V"; exit "$rc"; }

log "S5 step 1: central directory check"
$PFX $PY -u $KIT/unzip_parts.py --parts $PARTS --out "$DATA" --list-only || die "list-only FAILED (part order / incomplete download?)"

log "S5 step 2: unzip, 6 sequential streams"
$PFX $PY -u $KIT/unzip_parts.py --parts $PARTS --out "$DATA" --procs 6 > $L/unzip_full.log 2>&1
tail -1 $L/unzip_full.log
grep -q "^UNZIP OK" $L/unzip_full.log || die "unzip FAILED: see $L/unzip_full.log (re-run resumes)"
for f in train.csv train_series.csv; do [ -s "$DATA/$f" ] || die "missing $DATA/$f (archive root nested? set DATA to the folder that holds train.csv)"; done
n=$(ls "$DATA/train_series" | wc -l); log "train_series study dirs: $n (expect 4407)"
[ "$n" -eq 4407 ] || die "study count $n != 4407"

build() {   # one build pass; $1 = pass name
  log "S5 step 3 ($1): px build, $PROCS procs x $THREADS threads"
  $PFX $PY -u $KIT/px_build_s6.py --data "$DATA" --split train --src $KIT/src --code $KIT/code \
    --anchor $KIT/weights/anatomy.pt --out $PX --procs $PROCS --threads $THREADS --mem-cap-gb 60 --log-every 25 --progress-s 600 \
    >> $L/px_build_full.log 2>&1
  local rc=$?
  tail -2 $L/px_build_full.log
  [ "$rc" = "0" ] || die2 "$rc" "px build ($1) exit $rc (3 = MEM_CAP stop, 4 = a worker died): NOT finalized; fix the cause and re-run this script (the build resumes)"
}
nerr() { $PY -c 'import json,glob,sys; d=set(); e=set()
for p in glob.glob(sys.argv[1] + "/part*.json"):
    s = json.load(open(p)); d |= set(s["done"]); e |= set(s["errors"])
print(len(e - d))' "$PX"; }
build pass1
E1=$(nerr); log "per-study errors after pass 1: $E1"
if [ "${E1:-0}" -gt 0 ]; then build "retry of $E1 errored studies"; fi

log "S5 step 4: finalize"
$PFX $PY -u $KIT/px_build_s6.py --out $PX --finalize | tee -a $L/px_build_full.log
[ "${PIPESTATUS[0]}" = "0" ] || die "finalize failed"

log "S5 step 5: completeness + check-targets (CPU)"
CT=$L/px_build_check_targets.log
( cd $KIT && $PFX $PYT -u $KIT/ct_s6v100.py --px $PX --targets $KIT/targets_s6seed.csv --out $KIT/runs/px_build_check --arms E_prior \
    --backbone convnext_small --seed-base 43 --folds 5 --only-folds 0 --check-targets ) > $CT 2>&1 || die "check-targets rc != 0 (see $CT)"
POOL=$(grep -o '^pool [0-9]*' $CT | head -1 | cut -d' ' -f2); GOLD=$(grep -o '^pool [0-9]* gold [0-9]*' $CT | head -1 | cut -d' ' -f4)
read -r LISTED DONE ERRS MISSING < <($PY -c 'import csv,json,sys
idx = json.load(open(sys.argv[1] + "/index.json")); listed = {r["StudyInstanceUID"] for r in csv.DictReader(open(sys.argv[2]))}
done = {u for u, v in idx["done"].items() if v[1] > 0}; err = set(idx.get("errors") or {})
open(sys.argv[3], "w").write("".join("%s %s\n" % (u, str(idx["errors"][u])[:200]) for u in sorted(err)))
print(len(listed), len(done & listed), len(err & listed), len(listed - done - err))' "$PX" "$DATA/train.csv" "$G/PX_ERRORS.txt")
MSG="listed $LISTED done $DONE errors $ERRS missing $MISSING | check-targets pool $POOL gold $GOLD | steps 16 x floor(pool/2) = $((16 * (${POOL:-0} / 2))) | px $PX"
log "$MSG"
[ "$GOLD" = "58" ] || die "gold $GOLD != 58: $MSG"
[ "${MISSING:-1}" = "0" ] || die "$MISSING studies neither done nor errored: $MSG"
[ "${POOL:-0}" -ge "$POOL_FLOOR" ] || die "pool $POOL < floor $POOL_FLOOR: $MSG"
echo "PASS $(date '+%F %T') pool $POOL gold $GOLD errors $ERRS missing 0 ($MSG)" > "$V"
log "S5 done: $(cat $V)"
log "errors (left out of the pool by the trainer): $G/PX_ERRORS.txt. Put 'pool $POOL steps $((16 * (POOL / 2)))' into Amendment 7 and gates/A7.verdict."
# arm:  PARTS="..." DATA=... setsid nohup bash ~/s6seed/main_s5_unzip_px.sh > ~/s6seed/logs/main_s5.log 2>&1 < /dev/null &
