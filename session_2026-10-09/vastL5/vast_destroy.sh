#!/bin/bash
# vast_destroy.sh: wipe the competition data on the instance, DESTROY the instance (stops all billing), verify it is
# gone, then give every fold that was not fetched complete back to the box queue (box_placeholders.sh release).
#   bash vast_destroy.sh --yes [--reason TEXT] [--no-wipe] [--keep-placeholders]
#   bash vast_destroy.sh            (without --yes: prints what it would do, does nothing)
# Called by vast_watch.sh at the end; also the manual kill switch. Idempotent: an instance already gone is fine.
SCRIPT_TAG=destroy; source "$(dirname "$0")/config.sh"
YES=0; WIPE=1; KEEPPH=0; REASON=manual
while [ $# -gt 0 ]; do case $1 in --yes) YES=1;; --no-wipe) WIPE=0;; --keep-placeholders) KEEPPH=1;; --reason) REASON=$2; shift;;
  --dry-run) YES=0;; *) die "unknown argument $1";; esac; shift; done
IID=$(st_get INSTANCE_ID); FOLDS=$(st_get FOLDS)
[ -n "$IID" ] || { log "no instance in state.env: nothing to destroy"; exit 0; }
REL=""; for F in $FOLDS; do [ "$(st_get FETCHED_f$F)" = ok ] || [ -n "$(st_get PENDING_PUSH_f$F)" ] || REL="$REL $F"; done; REL=$(echo $REL)
log "== vast_destroy instance $IID ($(st_get LABEL)) | reason: $REASON | wipe $WIPE | folds to give back to the box: [${REL:-none}]"
if [ $YES = 0 ]; then log "DRY: add --yes to wipe + destroy (nothing done)"; exit 0; fi
if [ "$(st_get DESTROYED)" != 1 ]; then
  if [ $WIPE = 1 ]; then
    w=$(ix "pkill -f 'lane[.]sh'; pkill -f 'fleet_train[.]py'; sleep 5; rm -rf $R/fleet_store $R/fleet_runs $R/hf $R/novel $R/rsna_train $R/fast_build; echo left: \$(ls $R)" 2>&1 | tail -n 1)
    log "wipe on the instance: ${w:-ssh failed (the destroy below deletes the disk anyway)}"
  fi
  for i in 1 2 3 4 5; do
    vast_raw destroy instance "$IID" -y > /dev/null; rc=$?
    [ $rc = 3 ] && { log "ALERT: 2FA session needed to destroy $IID. Eesh: vastai tfa send-email, then vastai tfa login -t email -s <secret> -c <code>; then re-run: bash vast_destroy.sh --yes. The on-instance guard self-destroys at $(date -d @$(st_get GUARD_DEADLINE) '+%m-%d %H:%M') at the latest."; touch "$LOGDIR/ALERT_2FA"; exit 3; }
    sleep 15
    IJ=$(vast_raw show instances 2>/dev/null) && S=$(printf '%s' "$IJ" | jq_py "
L = d if isinstance(d, list) else d.get('instances', [])
print(next((str(i.get('actual_status')) for i in L if str(i.get('id')) == '$IID'), 'gone'))" 2>/dev/null) || S=api-error
    case ${S:-api-error} in gone) st_set DESTROYED 1; st_set DESTROYED_AT "$(date '+%F %T')"; log "instance $IID DESTROYED (billing stopped)"; break;; esac
    log "destroy attempt $i: instance still reports '$S'; retrying"; sleep 30
  done
  [ "$(st_get DESTROYED)" = 1 ] || die "instance $IID still exists after 5 destroy attempts: check https://console.vast.ai/instances/"
else log "instance $IID already destroyed at $(st_get DESTROYED_AT)"; fi
rm -f "$LOGDIR/ALERT_2FA"
if [ -n "$REL" ] && [ $KEEPPH = 0 ]; then
  if box_up && sync_box_ops; then bops ph-release $REL | sed 's/^/  box /'; st_set RELEASED "$REL"
  else log "WARN: box unreachable: give folds [$REL] back later with: bash box_placeholders.sh release $REL"; fi
fi
C=$(credit_now 2>/dev/null); C0=$(st_get CREDIT0)
[ -n "$C" ] && [ -n "$C0" ] && log "credit now \$$C (start \$$C0, spent ~\$$("$PYL" -c "print(round($C0 - $C, 2))"))"
log "destroy done"
