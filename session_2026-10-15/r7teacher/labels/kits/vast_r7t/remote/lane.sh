#!/bin/bash
# lane.sh G "F1 [F2 ..]": instance side. Runs the listed folds ONE AFTER ANOTHER on GPU G (CUDA_VISIBLE_DEVICES=G),
# each as its own fleet_train.py process with the exact box argv (argv_template.txt, ~/ -> /workspace/rsna/).
# Idempotent: a fold whose log already ends with the closing {"raw": .., "ema": ..} line is skipped, a live trainer
# for the fold is waited for, a partial run is moved aside (never deleted) and restarted. A fold is only STARTED if it
# can finish before DEADLINE_EPOCH (plan.env, = watcher time cap minus fetch time), using the measured duration of the
# previous fold on this lane (else the MODEL mid estimate); otherwise it is marked skipped and goes back to
# the box. One retry for a failure in the first 15 min (e.g. a transient loader crash).
set -u
R=${RSNA_ROOT:-/workspace/rsna}; . $R/kit/kit.env; . $R/kit/plan.env       # FOLDS DEADLINE_EPOCH MODEL_FOLD_S PYI
G=$1; MYFOLDS=$2; S=$R/status; mkdir -p $S $R/logs $R/fleet_runs
say(){ echo "[lane$G $(date '+%m-%d %H:%M:%S')] $*" >> $R/logs/lane_$G.log; }
closing(){ tail -n 1 "$1" 2>/dev/null | tr -d '\r' | grep -Eq '^\{"raw": [^,]+, "ema": [^}]+\}$'; }
alive(){ pgrep -f -- "fleet_train[.]py .*--out $R/fleet_runs/$1( |\$)" > /dev/null 2>&1; }
echo $$ > $S/lane_$G.pid
say "start pid $$ | GPU $G | folds [$MYFOLDS] | deadline $(date -d @$DEADLINE_EPOCH '+%F %T')"
prev=""
for F in $MYFOLDS; do
  N=f${F}_s${SEED}_${RUNTAG}; OUT=$R/fleet_runs/$N; LOGF=$OUT.log
  if closing $LOGF; then say "f$F already complete"; touch $S/f$F.done; continue; fi
  if alive $N; then say "f$F trainer already alive: waiting for it"; t0=$(cat $S/f$F.running 2>/dev/null); while alive $N; do sleep 60; done
    rm -f $S/f$F.running
    if closing $LOGF; then touch $S/f$F.done; [ -n "$t0" ] && prev=$(( $(date +%s) - t0 )); say "f$F finished (measured $(( ${prev:-0} / 60 )) min)"; continue; fi; fi
  try=0
  while :; do
    fac=103; [ "$F" = 4 ] && fac=110                     # coordinator 10-09: f4 only if it clearly fits
    if [ -n "$prev" ]; then est=$(( prev * fac / 100 )); else est=$(( MODEL_FOLD_S * 105 / 100 )); fi   # measured fold on this GPU
    if [ $(( $(date +%s) + est )) -gt "$DEADLINE_EPOCH" ]; then
      say "SKIP f$F: estimated $(( est / 60 )) min would end after the deadline; the fold goes back to the box"
      echo "estimated $(( est / 60 )) min > time left $(( (DEADLINE_EPOCH - $(date +%s)) / 60 )) min" > $S/f$F.skipped; break
    fi
    if [ -e $OUT ] || [ -e $LOGF ]; then A=${OUT}_partial_$(date +%m%d_%H%M%S); mv -n $OUT $A 2>/dev/null; mv -n $LOGF $A.log 2>/dev/null; say "f$F partial run moved aside to $A"; fi
    ARGV=$(sed -e "s#@R@#$R#g" -e "s#@F@#$F#g" $R/kit/argv_template.txt)
    rm -f $S/f$F.failed $S/f$F.skipped; date +%s > $S/f$F.running
    say "LAUNCH f$F on GPU $G: $PYI fleet_train.py $ARGV"
    t0=$(date +%s)
    ( cd $R/fast_build/code2 && exec env CUDA_VISIBLE_DEVICES=$G HF_HOME=$R/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
        $PYI fleet_train.py $ARGV ) > $LOGF 2>&1 < /dev/null
    rc=$?; dt=$(( $(date +%s) - t0 )); rm -f $S/f$F.running
    if [ $rc -eq 0 ] && closing $LOGF; then
      say "DONE f$F rc 0 in $(( dt / 60 )) min: $(tail -n 1 $LOGF)"; touch $S/f$F.done; prev=$dt; break
    fi
    say "FAILED f$F rc $rc after $(( dt / 60 )) min: $(tail -n 2 $LOGF | tr '\n' ' ' | cut -c1-300)"
    if [ $try -eq 0 ] && [ $dt -lt 900 ]; then try=1; say "f$F: early failure, one retry"; continue; fi
    echo "rc $rc after $(( dt / 60 )) min" > $S/f$F.failed; break
  done
done
say "lane exit"; rm -f $S/lane_$G.pid
