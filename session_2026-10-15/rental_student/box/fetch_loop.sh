#!/bin/bash
# fetch_loop.sh (BOX, setsid; started by driver.sh after RUN_STARTED; survives laptop / Claude session loss: lesson 10-10).
# Pattern = session_2026-10-13/raptor/vast/box_fetch.sh (pulled 4 seeds sha-verified, FINAL_FETCH_OK, guard destroyed).
# Reads ~/rstud/run.env (I_HOST I_PORT GUARD_DEADLINE FOLDS ...). PULL ONLY: its only writes on the instance are status
# markers (box_seen, f<F>.fetched_ok, FINAL_FETCH_OK). Every POLL_S (300 s):
#   done fold   -> tar pull of <N>/ + <N>.log + <N>.receipt.json + <N>.sha256 into ~/rstud/fetch/tmp_<N>, verify_fold.py
#                  (sha256 of every file, no extras, closing line, ckpt args vs the box recipe, receipt), then PLACE into the
#                  EMPTY placeholder ~/fleet_runs/<N> (+ <N>.log) and write ~/fleet_runs/rstud_<N>_receipt.json;
#                  3 failed verifies -> kept aside as ~/rstud/fetch/bad_<N>_<ts>, placeholder released (fold back to the box)
#   failed / skipped fold -> placeholder released (the box T_b sheet line can train it), resolved
#   from GUARD_DEADLINE - 40 min: running folds pulled as ~/fleet_runs/<N>_rentpartial_<ts> (never a student), unstarted
#                  folds resolved; placeholders released
# When every planned fold is resolved: FINAL_FETCH_OK on the instance (its guard self-destroys 15 min later) and
# ~/rstud/FINAL_FETCH_OK. Idempotent (resolved markers in ~/rstud/fetch). Tests: RSTUD_SSH (mock transport), RSTUD_FR,
# RSTUD_POLL_S, RSTUD_MAX_POLLS, VERIFY_SKIP.
set -u
D=${RSTUD_HOME:-$HOME/rstud}; SRC=${RSTUD_SRC:-$HOME/rstud/src}; . $SRC/remote/kit.env; . ${RSTUD_ENV:-$D/run.env}
FR=${RSTUD_FR:-$HOME/fleet_runs}; FD=$D/fetch; mkdir -p $FD; L=$FD/fetch.log; POLL=${RSTUD_POLL_S:-300}
PY=$HOME/miniconda3/envs/rsna/bin/python; R=${I_ROOT:-/workspace/rsna}
SSH=${RSTUD_SSH:-ssh -i $HOME/.ssh/vast_rsna_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_rstud -o LogLevel=ERROR -p $I_PORT root@$I_HOST}
OPS="bash $SRC/box/ops.sh"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L; }
if [ -s $FD/fetch.pid ] && kill -0 "$(cat $FD/fetch.pid)" 2>/dev/null && [ "$(cat $FD/fetch.pid)" != $$ ]; then echo "fetch loop already running (pid $(cat $FD/fetch.pid))"; exit 0; fi
echo $$ > $FD/fetch.pid
say "start pid $$ | instance ${INSTANCE_ID:-?} $I_HOST:$I_PORT | folds [$FOLDS] | guard deadline $(date -d @$GUARD_DEADLINE '+%m-%d %H:%M') | FR $FR"
# review 10-14: verify against the bundle that was PUSHED (driver snapshot; a later re-pack cannot change it) and, for a REAL
# bundle, only if that table is still the T_b table published in ~/r7t_labels/final/LABELS_READY (else nothing is placed)
BJ=$D/state_${INSTANCE_ID:-none}/BUNDLE.json; [ -s $BJ ] || BJ=${BUNDLE:-$D/bundle}/BUNDLE.json
TSHA=$(sed -n 's/.*"table_sha256": "\([0-9a-f]*\)".*/\1/p' $BJ); [ -n "$TSHA" ] || TSHA=NO_BUNDLE_TABLE_SHA
if [ "$(sed -n 's/.*"mode": "\([A-Z]*\)".*/\1/p' $BJ)" = REAL ] \
   && ! grep -F 'labels_r7t_b.csv' $HOME/r7t_labels/final/LABELS_READY 2>/dev/null | grep -qF "sha256 $TSHA"; then
  say "STOP-PLACE: the pushed table sha256 ${TSHA:0:16} is not the T_b sha256 in ~/r7t_labels/final/LABELS_READY: every fold fails verify (kept aside)"
  TSHA=NOT_IN_LABELS_READY; fi
say "verify table sha256 ${TSHA:0:16} (from $BJ)"
resolve(){ echo "$2" > $FD/resolved_f$1; say "f$1 RESOLVED: $2"; }
place(){   # N TMP F -> 0 placed
  local N=$1 T=$2 F=$3 DD=$FR/$1
  if pgrep -f -- "fleet_train[.]py .*--out [^ ]*/$N( |\$)" > /dev/null; then say "f$F: a box trainer runs $N: not placing"; return 1; fi
  if [ -e $DD.log ]; then say "f$F: $DD.log already exists: not placing"; return 1; fi
  if [ -d $DD ] && [ -n "$(ls -A $DD)" ]; then say "f$F: $DD is not empty: not placing"; return 1; fi
  [ -d $DD ] || { $OPS ph-create $F >> $L 2>&1 || return 1; }
  mv $T/$N/* $DD/ && mv $T/$N.log $DD.log || { say "f$F: move into the placeholder failed"; return 1; }
  return 0
}
miss=0; polls=0; vfail=()
while :; do
  now=$(date +%s); polls=$((polls + 1)); all=1
  st=$($SSH "RSNA_ROOT=$R bash $R/kit/status.sh touch_seen; RSNA_ROOT=$R bash $R/kit/status.sh" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ] || [ -z "$st" ]; then
    miss=$((miss + 1)); say "instance unreachable ($miss)"
    if [ $now -gt $((GUARD_DEADLINE + 3600)) ]; then
      say "past the guard deadline + 1 h: giving up; unresolved folds return to the box"
      for F in $FOLDS; do [ -e $FD/resolved_f$F ] || { $OPS ph-release $F >> $L 2>&1; resolve $F unreachable; }; done
      date > $D/FETCH_GAVE_UP; rm -f $FD/fetch.pid; exit 1
    fi
    sleep $POLL; continue
  fi
  miss=0
  near=0; [ $now -ge $((GUARD_DEADLINE - 2400)) ] && near=1
  pp=()                                  # pull every newly finished fold IN PARALLEL (5 x ~3.9 GB incl. epoch ckpts)
  for F in $FOLDS; do
    [ -e $FD/resolved_f$F ] && continue; N=f${F}_s${SEED}_${RUNTAG}; T=$FD/tmp_$N
    s=$(printf '%s\n' "$st" | sed -n "s/^FOLD f$F state=\([a-z]*\).*/\1/p")
    [ "$s" = done ] && [ ! -e $T/.pulled ] || continue
    rm -rf $T; mkdir -p $T
    ( $SSH "cd $R/fleet_runs && tar cf - $N $N.log $N.receipt.json $N.sha256" | tar xf - -C $T 2>> $L && touch $T/.pulled ) & pp+=($!)
  done
  t0=$(date +%s); for p in "${pp[@]}"; do wait $p; done; [ ${#pp[@]} -gt 0 ] && say "pulled ${#pp[@]} fold(s) in parallel in $(( $(date +%s) - t0 ))s"
  for F in $FOLDS; do
    [ -e $FD/resolved_f$F ] && continue
    N=f${F}_s${SEED}_${RUNTAG}
    s=$(printf '%s\n' "$st" | sed -n "s/^FOLD f$F state=\([a-z]*\).*/\1/p")
    case $s in
      done)
        T=$FD/tmp_$N
        if [ -e $T/.pulled ] && rm -f $T/.pulled; then
          ARGV=$(sed -e "s#@R@#$R#g" -e "s#@F@#$F#g" $SRC/remote/argv_template.txt | tr -d '\r\n')
          V=$(nice -n 19 $PY $SRC/box/verify_fold.py $T $N $F --mode canonical --table $TABLE --seed $SEED --ttr $TTR --tsl $TSL \
                --epochs ${VERIFY_EPOCHS:-$EPOCHS} --model_sha $MODEL_SHA --argv "$ARGV" --table_sha256 "$TSHA" --skip "${VERIFY_SKIP:-}" 2>> $L); vr=$?
          echo "$V" >> $L
          if [ $vr -eq 0 ] && place $N $T $F; then
            mkdir -p $FD/$N && cp -p $T/$N.receipt.json $T/$N.sha256 $FD/$N/ && echo "$V" > $FD/$N/verify.json
            VJ="$V" $PY - "$FR/rstud_${N}_receipt.json" "$T/$N.receipt.json" <<PYR
import json, sys, os
r = json.load(open(sys.argv[2])); v = json.loads(os.environ['VJ'])
json.dump(dict(run="$N", source="Vast instance ${INSTANCE_ID:-?} (${GPU_NAME:-?}, offer ${OFFER:-?}), RSNA-only bundle", placed_into="$FR/$N",
               instance_receipt=r, box_verify=v, verified_at="$(date '+%F %T %Z')"), open(sys.argv[1], 'w'), indent=1)
PYR
            rm -rf $T; $SSH "touch $R/status/f$F.fetched_ok" 2>/dev/null
            resolve $F ok; say "f$F FETCHED + VERIFIED + PLACED -> $FR/$N ($(du -sh $FR/$N | cut -f1); $(printf '%s' "$V" | sed -n 's/.*"wall_h": \([0-9.]*\).*/\1/p') h on the rental)"
          else
            vfail[$F]=$(( ${vfail[$F]:-0} + 1 )); say "f$F verify/place failed (attempt ${vfail[$F]})"; all=0
            if [ ${vfail[$F]} -ge 3 ]; then mv $T $FD/bad_${N}_$(date +%m%d_%H%M%S); $OPS ph-release $F >> $L 2>&1; resolve $F verify_failed; else rm -rf $T; fi
          fi
        else say "f$F pull failed (retry next poll)"; rm -rf $T; all=0; fi;;
      failed|skipped)
        $OPS ph-release $F >> $L 2>&1; resolve $F "$s on the instance: $($SSH "cat $R/status/f$F.$s" 2>/dev/null | head -c 160)";;
      running)
        if [ $near = 1 ]; then
          P=${N}_rentpartial_$(date +%m%d_%H%M); T=$FD/tmp_$P; rm -rf $T; mkdir -p $T
          if $SSH "cd $R/fleet_runs && { find $N -type f; echo $N.log; } | LC_ALL=C sort | xargs -d '\n' sha256sum > $N.sha256 && tar cf - $N $N.log $N.sha256" | tar xf - -C $T \
             && nice -n 19 $PY $SRC/box/verify_fold.py $T $N $F --mode partial >> $L 2>&1; then
            mkdir -p $FR/$P && mv $T/$N/* $FR/$P/ && mv $T/$N.log $FR/$P.log && rm -rf $T
            $OPS ph-release $F >> $L 2>&1; resolve $F "partial (near the guard deadline) -> $FR/$P (not a student)"
          else say "f$F partial pull failed"; all=0; fi
        else all=0; fi;;
      pending)
        if [ $near = 1 ]; then $OPS ph-release $F >> $L 2>&1; resolve $F "never started before the deadline"; else all=0; fi;;
      *) all=0;;
    esac
    [ -e $FD/resolved_f$F ] || all=0
  done
  if [ $all = 1 ]; then
    $SSH "date > $R/status/FINAL_FETCH_OK" && say "FINAL_FETCH_OK written on the instance (its guard destroys it 15 min later)"
    date > $D/FINAL_FETCH_OK
    say "ALL RESOLVED: $(for F in $FOLDS; do echo -n "f$F=$(head -c 40 $FD/resolved_f$F) | "; done)"; rm -f $FD/fetch.pid; exit 0
  fi
  [ -n "${RSTUD_MAX_POLLS:-}" ] && [ $polls -ge $RSTUD_MAX_POLLS ] && { say "test: max polls reached"; rm -f $FD/fetch.pid; exit 2; }
  sleep $POLL
done
