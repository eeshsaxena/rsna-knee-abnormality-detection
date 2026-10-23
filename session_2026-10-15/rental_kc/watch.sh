#!/bin/bash
# watch.sh (rental_kc laptop BACKUP only; the box driver / fetch loop and the on-instance guard are primary). Every 10 min: kit spend
# (credit drop since ledger CREDIT_START) and the instance status; destroys the instance with the CLI (needs the 2FA session)
# when the box wrote ~/rkc/FINAL_FETCH_OK 20+ min ago, ~/rkc/ABORTED 5+ min ago, ~/rkc/FETCH_GAVE_UP, the guard
# deadline + 15 min passed, or the kit spend reached CAP - $0.30. Exits when the instance is gone.
SCRIPT_TAG=watch; source "$(dirname "$0")/config.sh"
IID=$(st_get INSTANCE_ID); GDL=$(st_get GUARD_DEADLINE)
[ -n "$IID" ] || die "no instance in state.env"
log "watch start | instance $IID | guard deadline $(date -d @$GDL '+%m-%d %H:%M') | cap \$$CAP_USD"
while :; do
  [ "$(st_get DESTROYED)" = 1 ] && { log "instance marked destroyed: watcher exits"; exit 0; }
  sp=$(spent_total 2>/dev/null) && log "kit spend \$$sp of \$$CAP_USD" || sp=""
  S=$(vast_raw show instance "$IID" 2>/dev/null | jq_py 'print(d.get("actual_status"))' 2>/dev/null)
  if [ -n "$sp" ] && { [ -z "$S" ] || [ "$S" = None ]; } && [ -z "$(instances_json 2>/dev/null | jq_py "print(''.join(str(i['id']) for i in (d if isinstance(d, list) else d.get('instances', [])) if str(i['id']) == '$IID'))" 2>/dev/null)" ]; then
    log "instance $IID gone (guard or box finished it)"; st_set DESTROYED 1; st_set DESTROYED_AT "$(date '+%F %T')"; exit 0; fi
  m=$(bx 'for f in FINAL_FETCH_OK ABORTED FETCH_GAVE_UP; do [ -e ~/rkc/$f ] && echo "$f $(stat -c %Y ~/rkc/$f)"; done' 2>/dev/null)
  now=$(date +%s); why=""
  while read -r k t; do [ -n "$k" ] || continue
    case $k in FINAL_FETCH_OK) [ $((now - t)) -ge 1200 ] && why="box FINAL_FETCH_OK 20+ min ago";; ABORTED) [ $((now - t)) -ge 300 ] && why="box ABORTED";; FETCH_GAVE_UP) why="box fetch gave up";; esac
  done <<< "$m"
  [ $now -ge $((GDL + 900)) ] && why="guard deadline + 15 min passed"
  [ -n "$sp" ] && num_ge "$sp" "$("$PYL" -c "print($CAP_USD - 0.30)")" && why="kit spend \$$sp reached the cap line"
  if [ -n "$why" ]; then log "BACKUP DESTROY ($why): instance status ${S:-?}"; bash "$KIT/destroy.sh" --yes; exit 0; fi
  sleep 600
done
