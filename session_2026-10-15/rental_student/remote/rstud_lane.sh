#!/bin/bash
# rstud_lane.sh LANE_ID GPU FOLDS (instance side, setsid via start_lanes.sh; also the box CPU mock with RSNA_ROOT + MOCK_*).
# Runs FOLDS (comma list) one after another on GPU (CUDA_VISIBLE_DEVICES=GPU), each its own fleet_train.py with the exact
# box argv (kit/argv_template.txt, @R@ -> $R, @F@ -> fold). GPU "any" = wait until some GPU has no running trainer for 2
# polls (60 s apart, never before LANE_ANY_DELAY_S), then take it (the 5th fold after round 1).
# Per fold (pattern = vastS1044 / vastR7T lane.sh): skipped if complete; admission = it must end before plan.env
# DEADLINE_EPOCH (estimate = this lane's previous measured fold x1.05, else plan.env FOLD_EST_S x1.05), else
# status/f<F>.skipped (the fold goes back to the box); a partial run is moved aside (never deleted); one retry for a failure
# in the first 15 min. On success: <N>.receipt.json, <N>.sha256 (sha256 of every file of the fold dir + <N>.log + the
# receipt; paths relative to fleet_runs), then status/f<F>.done. status/f<F>.running = "<start epoch> <gpu>".
set -u
R=${RSNA_ROOT:-/workspace/rsna}; K=$R/kit; S=$R/status; FR=$R/fleet_runs
. $K/kit.env; . $K/plan.env      # kit: RUNTAG SEED ...; plan: FOLDS DEADLINE_EPOCH FOLD_EST_S PYI NG
LID=$1; G=$2; MYF=$(echo "$3" | tr ',' ' ')
mkdir -p $S $R/logs $FR
echo $$ > $S/lane_$LID.pid; echo "$2:$3" > $S/lane_$LID.spec
say(){ echo "[lane$LID $(date '+%m-%d %H:%M:%S')] $*" >> $R/logs/lane_$LID.log; }
closing(){ tail -n 1 "$1" 2>/dev/null | tr -d '\r' | grep -Eq '^\{"raw": [^,]+, "ema": [^}]+\}$'; }
alive(){ pgrep -f -- "fleet_train[.]py .*--out $FR/$1( |\$)" > /dev/null 2>&1; }
busy(){ local f; for f in $S/f*.running; do [ -e "$f" ] || continue; [ "$(awk '{print $2}' "$f")" = "$1" ] && return 0; done; return 1; }
gpu_name(){ [ "${MOCK_CPU:-0}" = 1 ] && { echo "mock-cpu"; return; }; nvidia-smi --query-gpu=name --format=csv,noheader -i "$1" 2>/dev/null | head -n 1; }
say "start pid $$ | GPU $G | folds [$MYF] | deadline $(date -d @$DEADLINE_EPOCH '+%F %T') | est/fold $(( FOLD_EST_S / 60 )) min"
if [ "$G" = any ]; then
  sleep "${LANE_ANY_DELAY_S:-120}"; pf=""
  while :; do
    fr=""; for g in $(seq 0 $(( NG - 1 ))); do busy $g || { fr=$g; break; }; done
    if [ -n "$fr" ] && [ "$fr" = "$pf" ]; then G=$fr; break; fi
    pf=$fr; sleep "${LANE_ANY_POLL_S:-60}"
  done
  echo "$G:$3" > $S/lane_$LID.spec; say "any-lane took GPU $G"
fi
prev=""
for F in $MYF; do
  N=f${F}_s${SEED}_${RUNTAG}; OUT=$FR/$N; LOGF=$OUT.log
  if closing $LOGF && [ -s $FR/$N.sha256 ]; then say "f$F already complete"; touch $S/f$F.done; continue; fi
  if alive $N; then say "f$F trainer already alive: waiting"; while alive $N; do sleep 60; done; fi
  try=0
  while :; do
    . $K/plan.env                                            # the driver may refresh FOLD_EST_S / DEADLINE_EPOCH after the smoke
    if [ -n "$prev" ]; then est=$(( prev * 105 / 100 )); else est=$(( FOLD_EST_S * 105 / 100 )); fi
    if [ $(( $(date +%s) + est )) -gt "$DEADLINE_EPOCH" ]; then
      say "SKIP f$F: estimated $(( est / 60 )) min would end after the lane deadline; the fold goes back to the box"
      echo "estimated $(( est / 60 )) min > time left $(( (DEADLINE_EPOCH - $(date +%s)) / 60 )) min" > $S/f$F.skipped; break
    fi
    if [ -e $OUT ] || [ -e $LOGF ]; then A=${OUT}_partial_$(date +%m%d_%H%M%S); mv -n $OUT $A 2>/dev/null; mv -n $LOGF $A.log 2>/dev/null; say "f$F partial run moved aside to $A"; fi
    ARGV=$(sed -e "s#@R@#$R#g" -e "s#@F@#$F#g" $K/argv_template.txt | tr -d '\r\n')
    if [ "${MOCK_CPU:-0}" = 1 ]; then CS=$(echo ${MOCK_CPUSETS:-0-1} | cut -d' ' -f$(( G + 1 ))); PRE="taskset -c $CS nice -n 19 ionice -c3"; CVD=""
    else PRE=""; CVD=$G; fi
    rm -f $S/f$F.failed $S/f$F.skipped $S/f$F.done; echo "$(date +%s) $G" > $S/f$F.running
    say "LAUNCH f$F on GPU $G: $PYI fleet_train.py $ARGV ${EXTRA_ARGS:-}"
    t0=$(date +%s)
    ( cd $R/fast_build/code2 && exec env CUDA_VISIBLE_DEVICES=$CVD HF_HOME=$R/hf HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
        OMP_NUM_THREADS=${OMP_T:-4} $PRE $PYI fleet_train.py $ARGV ${EXTRA_ARGS:-} ) > $LOGF 2>&1 < /dev/null &
    tp=$!; echo $tp > $S/f$F.tpid; wait $tp; rc=$?; rm -f $S/f$F.tpid; t1=$(date +%s); dt=$(( t1 - t0 ))
    if [ $rc -eq 0 ] && closing $LOGF; then
      RCPT=$FR/$N.receipt.json RN=$N RF=$F RG=$G RGN="$(gpu_name $G)" RLID=$LID RT0=$t0 RT1=$t1 RRC=$rc RARGV="$ARGV ${EXTRA_ARGS:-}" \
      RCLOSE="$(tail -n 1 $LOGF | tr -d '\r')" RTAB=$R/r7t_labels/real/$TABLE RCODE=$R/fast_build/code2 RSEED=$SEED \
      ${PYR:-$PYI} - <<'PYR' || say "receipt write failed for f$F"
import json, os, hashlib, glob
e = os.environ
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump(dict(run=e['RN'], fold=int(e['RF']), seed=int(e['RSEED']), gpu_index=e['RG'], gpu_name=e['RGN'], lane=e['RLID'],
               started=int(e['RT0']), ended=int(e['RT1']), wall_s=int(e['RT1']) - int(e['RT0']), rc=int(e['RRC']),
               argv=e['RARGV'].strip(), closing=e['RCLOSE'], table_sha256=sha(e['RTAB']),
               code_sha256={os.path.basename(p): sha(p) for p in sorted(glob.glob(e['RCODE'] + '/*.py'))},
               container_id=e.get('CONTAINER_ID', '')), open(e['RCPT'], 'w'), indent=1)
PYR
      ( cd $FR && { find $N -type f; echo $N.log; echo $N.receipt.json; } | LC_ALL=C sort | xargs -d '\n' sha256sum > $N.sha256.tmp && mv -f $N.sha256.tmp $N.sha256 ) \
        || say "sha256 list failed for f$F"
      rm -f $S/f$F.running; prev=$dt
      if [ -s $FR/$N.sha256 ] && [ -s $FR/$N.receipt.json ]; then touch $S/f$F.done
      else echo "trained (rc 0) but the receipt / sha256 list could not be written: outputs kept in $OUT, not fetched automatically" > $S/f$F.failed; fi
      say "DONE f$F rc 0 in $(( dt / 60 )) min: $(tail -n 1 $LOGF | cut -c1-120) | files $(wc -l < $FR/$N.sha256)"; break
    fi
    rm -f $S/f$F.running
    say "FAILED f$F rc $rc after $(( dt / 60 )) min: $(tail -n 2 $LOGF | tr '\n' ' ' | cut -c1-300)"
    if [ $try -eq 0 ] && [ $dt -lt 900 ] && ! grep -q 'out of memory' $LOGF; then try=1; say "f$F: early failure, one retry"; continue; fi
    echo "rc $rc after $(( dt / 60 )) min: $(grep -m1 -E 'Error|error' $LOGF | cut -c1-200)" > $S/f$F.failed; break
  done
done
say "lane exit"; rm -f $S/lane_$LID.pid
