#!/bin/bash
# vast_destroy.sh --yes: destroy the rented instance in state.env (stops billing) and record the spend.
SCRIPT_TAG=destroy; source "$(dirname "$0")/config.sh"
[ "${1:-}" = --yes ] || die "usage: vast_destroy.sh --yes"
IID=$(st_get INSTANCE_ID); [ -n "$IID" ] || die "no instance in state.env"
"$VAST_BIN" destroy instance "$IID" -y 2>&1 | tail -n 1 | sed 's/^/  /'
sleep 10
S=$(vast_raw show instance "$IID" 2>/dev/null | jq_py 'print(d.get("actual_status"))' 2>/dev/null)
log "destroy $IID -> status now '${S:-gone}'"
st_set DESTROYED 1
c=$(credit_now 2>/dev/null) && log "credit \$$c | total spend \$$("$PYL" -c "print(round($(st_get CREDIT0) - $c, 2))")"
