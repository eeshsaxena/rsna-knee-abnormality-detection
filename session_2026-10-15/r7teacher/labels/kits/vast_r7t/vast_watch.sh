#!/bin/bash
# vast_watch.sh: the money guard. Start it RIGHT AFTER vast_launch.sh rents the instance, in the background
# (Claude: Bash run_in_background; it dies with the session, so re-run it after any restart: it resumes from state.env).
# Every 10 min (POLL_S): instance status (keeps the 2FA session active), credit, spend vs cap, per-fold progress,
# GPU utilisation. It FINALISES (vast_fetch.sh --auto, then vast_destroy.sh --yes, which also gives unfinished folds
# back to the box queue) on the first of:
#   all planned folds done | spend >= budget - reserve | credit <= floor + reserve | time cap (DEADLINE_EPOCH)
#   | instance exited/offline/unknown on 3 polls | training lanes dead twice (one automatic restart first)
#   | (vastR7T) the credit nears the finalise line of the concurrent L5 run while L5 is alive: THIS run is
#     finalised so L5 is never pushed into its own stop. Spend = max(DPH x time + data-in, credit drop - L5's billing).
#     L5 (../vastL5) is only READ: its state.env and whether its instance id is still on the account.
#   bash vast_watch.sh [--once] [--dry-run]     (--once: one poll, finalise if a trigger fires; --dry-run: no actions)
# Started BEFORE the launch (no instance in state.env yet) it is a 2FA KEEPALIVE: a credit read every 10 min keeps the
# session from expiring for inactivity; it switches to watching as soon as vast_launch.sh records the instance.
SCRIPT_TAG=watch; source "$(dirname "$0")/config.sh"
ONCE=0; DRY=0
for a in "$@"; do case $a in --once) ONCE=1;; --dry-run) DRY=1; ONCE=1;; *) die "unknown argument $a";; esac; done
f2(){ "$PYL" -c "print('%.2f' % ($1))"; }
finalize(){
  local why=$1 i rc
  log "FINALISE: $why"
  if [ $DRY = 1 ]; then log "DRY: would run vast_fetch.sh --auto, write FINAL_FETCH_OK, then vast_destroy.sh --yes"; return 0; fi
  if [ -n "$(st_get TRAIN_STARTED)" ]; then
    for i in 1 2 3; do
      bash "$KIT/vast_fetch.sh" --auto && break
      [ "$HARD" = 1 ] && { log "hard spend limit: no fetch retry"; break; }
      log "fetch attempt $i incomplete; retry in 5 min"; sleep 300
    done
  fi
  ix "date > $R/status/FINAL_FETCH_OK" 2>/dev/null || true
  bash "$KIT/vast_destroy.sh" --yes --reason "$why"; rc=$?
  while [ $rc = 3 ]; do     # 2FA lost: keep trying; the on-instance guard is the backstop
    [ "$(date +%s)" -gt $(( $(st_get GUARD_DEADLINE) + 1800 )) ] && { log "past the guard deadline + 30 min: stop retrying (check the console)"; break; }
    sleep 300; bash "$KIT/vast_destroy.sh" --yes --reason "$why (retry)"; rc=$?
  done
  log "watcher exit (destroy rc $rc)"; exit 0
}
BADN=0; DEADLANES=0; SSHFAIL=0; LASTALERT=0
log "== vast_watch start | instance $(st_get INSTANCE_ID) | budget \$$(st_get BUDGET) | cap $(date -d @"$(st_get DEADLINE_EPOCH)" '+%m-%d %H:%M' 2>/dev/null) | poll ${POLL_S}s"
while :; do
  IID=$(st_get INSTANCE_ID)
  if [ -z "$IID" ]; then   # KEEPALIVE mode (started before the launch): keep the 2FA session active, alert if it dies
    if C=$(credit_now 2>/dev/null); then log "keepalive: no instance yet | 2FA session alive | credit \$$C"
    else [ $DRY = 1 ] || touch "$LOGDIR/ALERT_2FA"; log "keepalive: ALERT no 2FA session (RUNBOOK step 0)"; fi
    [ $ONCE = 1 ] && exit 0; sleep "$POLL_S"; continue
  fi
  if [ "$(st_get DESTROYED)" = 1 ]; then
    if [ -e "$LOGDIR/poll.active" ]; then   # poll_launch.sh is still looking for the next offer: idle as a 2FA keepalive
      C=$(credit_now 2>/dev/null) && log "keepalive: instance $IID destroyed, poll_launch active | credit \$$C" || { touch "$LOGDIR/ALERT_2FA"; log "keepalive: ALERT no 2FA session"; }
      [ $ONCE = 1 ] && exit 0; sleep "$POLL_S"; continue
    fi
    log "instance $IID already destroyed: watcher exit"; exit 0
  fi
  NOW=$(date +%s); L0=$(st_get LAUNCH_EPOCH); DPH=$(st_get DPH); B=$(st_get BUDGET); C0=$(st_get CREDIT0); DL=$(st_get DEADLINE_EPOCH)
  J=$(inst_json "$IID"); arc=$?
  if [ $arc = 3 ]; then S=2FA-LOST
    if [ $((NOW - LASTALERT)) -ge 3600 ]; then touch "$LOGDIR/ALERT_2FA"; LASTALERT=$NOW
      log "ALERT: the Vast 2FA session expired: the watcher can no longer destroy. Eesh must run RUNBOOK step 0 NOW (tfa send-email + tfa login)."; fi
  elif [ $arc != 0 ]; then S=api-error
  else S=$(printf '%s' "$J" | jq_py 'print(d.get("actual_status"))' 2>/dev/null); rm -f "$LOGDIR/ALERT_2FA"; fi
  C=""; [ $arc = 0 ] && C=$(credit_now 2>/dev/null)
  L5ID=$(l5_get INSTANCE_ID)
  if [ $arc = 0 ] && [ -n "$L5ID" ] && l5_alive; then      # read only: is L5's instance still on the account?
    if IL=$(vast_raw show instances 2>/dev/null) && ! printf '%s' "$IL" | grep -Eq "\"id\": $L5ID[,}]"; then
      st_set L5_GONE_AT "$NOW"; log "L5 instance $L5ID is no longer on the account: its billing stopped by now"; fi
  fi
  XF=$(st_get XFER); XF=${XF:-0}; L5S=$(l5_spend_since "$L0")
  SE=$(f2 "$DPH * ($NOW - $L0) / 3600 + $XF")
  SP=$SE; [ -n "$C" ] && [ -n "$C0" ] && SP=$(f2 "max($SE, $C0 - $C - $L5S)")
  L5TXT="L5 $(l5_alive && echo alive || echo gone) (its share \$$L5S)"
  OUT=""; ALLD=0; LA=-1
  if [ "$(st_get SETUP_OK)" = 1 ]; then
    if OUT=$(ix "bash $R/kit/status.sh" 2>/dev/null); then SSHFAIL=0
      ALLD=$(printf '%s\n' "$OUT" | sed -n 's/^ALL_DONE=//p'); LA=$(printf '%s\n' "$OUT" | sed -n 's/^LANES_ALIVE=//p')
    else SSHFAIL=$((SSHFAIL + 1)); fi
  fi
  FOLDLINE=$(printf '%s\n' "$OUT" | sed -n -E 's/^FOLD (f[0-9]) state=([^ ]*) epochs=([0-9]+)\/21 last_raw_val=([^ ]*) train_min\/ep=([^ ]*).*/\1:\2 ep\3 val\4 \5m/p' | tr '\n' ' ')
  GPUL=$(printf '%s\n' "$OUT" | sed -n -E 's/^GPU ([0-9]+), [^,]*, ([0-9]+) %.*/\2%/p' | tr '\n' ',' | sed 's/,$//')
  if [ $DRY = 0 ] && [ -n "$OUT" ] && [ -n "$(st_get TRAIN_STARTED)" ]; then   # lazy placeholders for folds a lane has started
    for F in $(st_get FOLDS); do
      [ "$(st_get PH_f$F)" = 1 ] && continue
      printf '%s\n' "$OUT" | grep -Eq "^FOLD f$F state=(running|done)" || continue
      PO=$(bops ph-create "$F" 2>&1); log "f$F started on Vast: box placeholder: $PO"
      printf '%s' "$PO" | grep -q '^OK' && st_set PH_f$F 1
    done
  fi
  if [ $DRY = 0 ] && [ -n "$OUT" ] && [ -n "$(st_get TRAIN_STARTED)" ]; then   # fetch + verify each finished fold at once
    for F in $(st_get FOLDS); do
      [ "$(st_get FETCHED_f$F)" = ok ] && continue
      printf '%s\n' "$OUT" | grep -Eq "^FOLD f$F state=done" || continue
      log "f$F finished: fetching it now"; bash "$KIT/vast_fetch.sh" "$F" > /dev/null 2>&1 && log "f$F fetched + verified" || log "WARN: incremental fetch of f$F failed (the final fetch retries)"
    done
  fi
  log "$S | spend ~\$$SP of \$$B (credit ${C:-?}; $L5TXT) | $(( (NOW - L0) / 60 )) min of $(( (DL - L0) / 60 )) | ${FOLDLINE:-setup not finished} | GPU util ${GPUL:--}$([ $SSHFAIL -gt 0 ] && echo " | ssh fail x$SSHFAIL")"
  HARD=0; num_ge "$SP" "$("$PYL" -c "print($B - 0.03)")" && HARD=1
  WHY=""
  [ -n "$(st_get TRAIN_STARTED)" ] && [ "$ALLD" = 1 ] && WHY="all planned folds finished"
  [ -z "$WHY" ] && num_ge "$SP" "$("$PYL" -c "print($B - $RESERVE_USD)")" && WHY="spend \$$SP reached budget \$$B minus reserve"
  [ -z "$WHY" ] && [ -n "$C" ] && ! num_ge "$C" "$("$PYL" -c "print($FLOOR_USD + $RESERVE_USD)")" && WHY="credit \$$C at the floor"
  [ -z "$WHY" ] && [ "$NOW" -ge "$DL" ] && WHY="time cap reached"
  if [ -z "$WHY" ] && [ -n "$C" ] && l5_alive; then
    LT=$(l5_trig)
    num_ge "$C" "$("$PYL" -c "print($LT + $L5_MARGIN)")" || WHY="credit \$$C within \$$L5_MARGIN of L5's finalise line \$$LT while L5 still runs: finalising THIS run (never L5)"
  fi
  # 10-10: the Vast API reported a healthy, training instance as 'offline' for one poll; count a bad status only when
  # our own ssh status probe also failed (or setup has not finished), so an API blip can never destroy a live run
  case $S in exited|offline|unknown) if [ "$(st_get SETUP_OK)" != 1 ] || [ $SSHFAIL -gt 0 ]; then BADN=$((BADN + 1)); else log "note: API says $S but the instance answers over ssh: ignored"; BADN=0; fi;; *) BADN=0;; esac
  [ -z "$WHY" ] && [ $BADN -ge 3 ] && WHY="instance $S on 3 polls"
  if [ -z "$WHY" ] && [ -n "$(st_get TRAIN_STARTED)" ] && [ "$LA" = 0 ] && [ "$ALLD" = 0 ]; then
    DEADLANES=$((DEADLANES + 1))
    if [ $DEADLANES -ge 2 ] && [ "$(st_get LANE_RESTARTS)" = 1 ]; then WHY="training lanes died again after a restart"
    elif [ $DEADLANES -ge 2 ] && [ $DRY = 0 ]; then st_set LANE_RESTARTS 1; DEADLANES=0; log "lanes dead with folds unfinished: restarting once (vast_train.sh)"; bash "$KIT/vast_train.sh" > /dev/null 2>&1; fi
  else DEADLANES=0; fi
  [ -n "$WHY" ] && finalize "$WHY"
  [ $ONCE = 1 ] && { log "single poll done (no trigger)"; exit 0; }
  W=$POLL_S; [ $((DL - $(date +%s))) -lt 2700 ] && W=180                     # last 45 min before the cap: poll every 3 min
  if [ -n "$C" ] && l5_alive && ! num_ge "$C" "$("$PYL" -c "print($(l5_trig) + $L5_MARGIN + 0.40)")"; then W=180; fi
  [ $((DL - $(date +%s) + 5)) -lt $W ] && W=$((DL - $(date +%s) + 5)); [ $W -lt 30 ] && W=30
  sleep $W
done
