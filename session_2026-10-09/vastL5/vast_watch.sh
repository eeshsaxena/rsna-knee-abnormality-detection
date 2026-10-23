#!/bin/bash
# vast_watch.sh: the money guard. Start it RIGHT AFTER vast_launch.sh rents the instance, in the background
# (Claude: Bash run_in_background; it dies with the session, so re-run it after any restart: it resumes from state.env).
# Every 10 min (POLL_S): instance status (keeps the 2FA session active), credit, spend vs cap, per-fold progress,
# GPU utilisation. It FINALISES (vast_fetch.sh --auto, then vast_destroy.sh --yes, which also gives unfinished folds
# back to the box queue) on the first of:
#   all planned folds done | spend >= budget - reserve | credit <= floor + reserve | time cap (DEADLINE_EPOCH)
#   | instance exited/offline/unknown on 3 polls | training lanes dead twice (one automatic restart first)
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
  [ "$(st_get DESTROYED)" = 1 ] && { log "instance $IID already destroyed: watcher exit"; exit 0; }
  NOW=$(date +%s); L0=$(st_get LAUNCH_EPOCH); DPH=$(st_get DPH); B=$(st_get BUDGET); C0=$(st_get CREDIT0); DL=$(st_get DEADLINE_EPOCH)
  J=$(inst_json "$IID"); arc=$?
  if [ $arc = 3 ]; then S=2FA-LOST
    if [ $((NOW - LASTALERT)) -ge 3600 ]; then touch "$LOGDIR/ALERT_2FA"; LASTALERT=$NOW
      log "ALERT: the Vast 2FA session expired: the watcher can no longer destroy. Eesh must run RUNBOOK step 0 NOW (tfa send-email + tfa login)."; fi
  elif [ $arc != 0 ]; then S=api-error
  else S=$(printf '%s' "$J" | jq_py 'print(d.get("actual_status"))' 2>/dev/null); rm -f "$LOGDIR/ALERT_2FA"; fi
  C=""; [ $arc = 0 ] && C=$(credit_now 2>/dev/null)
  SE=$(f2 "$DPH * ($NOW - $L0) / 3600")
  SP=$SE; [ -n "$C" ] && [ -n "$C0" ] && SP=$(f2 "max($SE, $C0 - $C)")
  OUT=""; ALLD=0; LA=-1
  if [ "$(st_get SETUP_OK)" = 1 ]; then
    if OUT=$(ix "bash $R/kit/status.sh" 2>/dev/null); then SSHFAIL=0
      ALLD=$(printf '%s\n' "$OUT" | sed -n 's/^ALL_DONE=//p'); LA=$(printf '%s\n' "$OUT" | sed -n 's/^LANES_ALIVE=//p')
    else SSHFAIL=$((SSHFAIL + 1)); fi
  fi
  FOLDLINE=$(printf '%s\n' "$OUT" | sed -n -E 's/^FOLD (f[0-9]) state=([^ ]*) epochs=([0-9]+)\/21 last_raw_val=([^ ]*) train_min\/ep=([^ ]*).*/\1:\2 ep\3 val\4 \5m/p' | tr '\n' ' ')
  GPUL=$(printf '%s\n' "$OUT" | sed -n -E 's/^GPU ([0-9]+), [^,]*, ([0-9]+) %.*/\2%/p' | tr '\n' ',' | sed 's/,$//')
  log "$S | spend ~\$$SP of \$$B (credit ${C:-?}) | $(( (NOW - L0) / 60 )) min of $(( (DL - L0) / 60 )) | ${FOLDLINE:-setup not finished} | GPU util ${GPUL:--}$([ $SSHFAIL -gt 0 ] && echo " | ssh fail x$SSHFAIL")"
  HARD=0; num_ge "$SP" "$("$PYL" -c "print($B - 0.15)")" && HARD=1
  WHY=""
  [ -n "$(st_get TRAIN_STARTED)" ] && [ "$ALLD" = 1 ] && WHY="all planned folds finished"
  [ -z "$WHY" ] && num_ge "$SP" "$("$PYL" -c "print($B - $RESERVE_USD)")" && WHY="spend \$$SP reached budget \$$B minus reserve"
  [ -z "$WHY" ] && [ -n "$C" ] && ! num_ge "$C" "$("$PYL" -c "print($FLOOR_USD + $RESERVE_USD)")" && WHY="credit \$$C at the floor"
  [ -z "$WHY" ] && [ "$NOW" -ge "$DL" ] && WHY="time cap reached"
  case $S in exited|offline|unknown) BADN=$((BADN + 1));; *) BADN=0;; esac
  [ -z "$WHY" ] && [ $BADN -ge 3 ] && WHY="instance $S on 3 polls"
  if [ -z "$WHY" ] && [ -n "$(st_get TRAIN_STARTED)" ] && [ "$LA" = 0 ] && [ "$ALLD" = 0 ]; then
    DEADLANES=$((DEADLANES + 1))
    if [ $DEADLANES -ge 2 ] && [ "$(st_get LANE_RESTARTS)" = 1 ]; then WHY="training lanes died again after a restart"
    elif [ $DEADLANES -ge 2 ] && [ $DRY = 0 ]; then st_set LANE_RESTARTS 1; DEADLANES=0; log "lanes dead with folds unfinished: restarting once (vast_train.sh)"; bash "$KIT/vast_train.sh" > /dev/null 2>&1; fi
  else DEADLANES=0; fi
  [ -n "$WHY" ] && finalize "$WHY"
  [ $ONCE = 1 ] && { log "single poll done (no trigger)"; exit 0; }
  W=$POLL_S; [ $((DL - $(date +%s) + 5)) -lt $W ] && W=$((DL - $(date +%s) + 5)); [ $W -lt 30 ] && W=30
  sleep $W
done
