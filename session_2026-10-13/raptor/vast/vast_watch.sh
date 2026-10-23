#!/bin/bash
# vast_watch.sh (laptop BACKUP only; the box fetch loop + on-instance guard are primary). Every 10 min: if the box wrote
# ~/raptor_store/rented/FINAL_FETCH_OK more than 20 min ago, or the guard deadline + 15 min passed, and the instance still
# exists -> destroy it with the CLI (needs Eesh's 2FA session). Logs spend (credit0 - credit now) each round.
SCRIPT_TAG=watch; source "$(dirname "$0")/config.sh"
IID=$(st_get INSTANCE_ID); GDL=$(st_get GUARD_DEADLINE); C0=$(st_get CREDIT0)
log "watch start | instance $IID | guard deadline $(date -d @$GDL '+%m-%d %H:%M')"
while :; do
  [ "$(st_get DESTROYED)" = 1 ] && { log "instance marked destroyed: watcher exits"; exit 0; }
  c=$(credit_now 2>/dev/null) && log "credit \$$c | spent \$$("$PYL" -c "print(round($C0 - $c, 2))")" || log "credit unknown (2FA session?)"
  f=$(bx 'stat -c %Y ~/raptor_store/rented/FINAL_FETCH_OK 2>/dev/null' 2>/dev/null)
  now=$(date +%s)
  if { [ -n "$f" ] && [ $((now - f)) -ge 1200 ]; } || [ $now -ge $((GDL + 900)) ]; then
    S=$(vast_raw show instance "$IID" 2>/dev/null | jq_py 'print(d.get("actual_status"))' 2>/dev/null)
    if [ -z "$S" ] || [ "$S" = None ]; then log "instance $IID gone (guard destroyed it)"; st_set DESTROYED 1; exit 0; fi
    log "instance still $S after final fetch / deadline: destroying (backup path)"; bash "$KIT/vast_destroy.sh" --yes; exit 0
  fi
  sleep 600
done
