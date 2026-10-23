#!/bin/bash
# destroy.sh --yes: destroy the instance in state.env (stops billing; the guard also wipes on its own triggers) and record
# the kit spend. Manual kill switch. Never touches any other instance.
SCRIPT_TAG=destroy; source "$(dirname "$0")/config.sh"
[ "${1:-}" = --yes ] || die "usage: destroy.sh --yes"
IID=$(st_get INSTANCE_ID); [ -n "$IID" ] || die "no instance in state.env"
ix "echo 'laptop destroy.sh' > $R/status/ABORT" 2>/dev/null && log "status/ABORT written (guard wipes the data)"
"$VAST_BIN" destroy instance "$IID" -y 2>&1 | tail -n 1 | sed 's/^/  /'
sleep 10
S=$(vast_raw show instance "$IID" 2>/dev/null | jq_py 'print(d.get("actual_status"))' 2>/dev/null)
log "destroy $IID -> status now '${S:-gone}'"
st_set DESTROYED 1; st_set DESTROYED_AT "$(date '+%F %T')"
sp=$(spent_total 2>/dev/null) && log "kit spend so far \$$sp of \$$CAP_USD"
