#!/bin/bash
# guard.sh (instance; started by the Vast onstart at every container start, embedded by launch.sh; pattern =
# session_2026-10-13/raptor/vast/remote/guard.sh, which self-destroyed instance 55762630 on 10-14 00:37 as designed).
# BACKSTOP: wipes the competition data and self-destroys the instance with the per-instance key Vast injects
# (CONTAINER_ID / CONTAINER_API_KEY, never printed). Checked every GUARD_POLL_S (60 s). Triggers:
#  (a) hard deadline RSNA_GUARD_DEADLINE (epoch, fixed at create from the $15 cap); status/guard_deadline can only TIGHTEN it
#  (b) status/FINAL_FETCH_OK (written by the BOX after it pulled + sha-verified every fold) is 15+ min old
#  (c) status/ABORT exists (box driver or laptop: smoke FAIL, parity mismatch, over-cap projection, ...): immediate
#  (d) bootstrap fuse: no status/RUN_STARTED by RSNA_BOOT_DEADLINE (create + 100 min): push / setup / smoke never finished
#  (e) orphan fuse: every planned fold is terminal and no lane is alive, no FINAL_FETCH_OK, and status/box_seen (touched by
#      the box fetch loop at every poll) is 120+ min old: nobody is fetching, stop paying
# GUARD_MOCK=1 (tests only): no pkill, no API call (prints mock codes); RSNA_ROOT must point at the mock root.
R=${RSNA_ROOT:-/workspace/rsna}; S=$R/status; LOG=$R/guard.log; mkdir -p "$S"
MOCK=${GUARD_MOCK:-0}; POLL=${GUARD_POLL_S:-60}; TRIES=${GUARD_TRIES:-6}; RETRY_S=${GUARD_RETRY_S:-120}
FIN_AGE=${GUARD_FIN_AGE_S:-900}; ORPHAN_AGE=${GUARD_ORPHAN_AGE_S:-7200}
[ -s "$S/guard_deadline0" ] || echo "${RSNA_GUARD_DEADLINE:-0}" > "$S/guard_deadline0"
[ -s "$S/boot_deadline" ] || echo "${RSNA_BOOT_DEADLINE:-0}" > "$S/boot_deadline"
DL0=$(tr -dc 0-9 < "$S/guard_deadline0"); DL0=${DL0:-0}; BOOT=$(tr -dc 0-9 < "$S/boot_deadline"); BOOT=${BOOT:-0}
say(){ echo "[guard $(date -u '+%m-%d %H:%M:%S')Z] $*" >> "$LOG"; }
envv(){ local v; v=$(printenv "$1"); [ -n "$v" ] || v=$(tr '\0' '\n' 2>/dev/null < /proc/1/environ | sed -n "s/^$1=//p" | head -n 1); printf '%s' "$v"; }
PYG=$(command -v python3 || command -v python || echo /opt/conda/bin/python)
api(){   # GET | DELETE | STOP -> prints the HTTP status code only
  [ "$MOCK" = 1 ] && { echo "mock-$1"; return; }
  CID=$(envv CONTAINER_ID) KEY=$(envv CONTAINER_API_KEY) M=$1 "$PYG" -c '
import os, json, urllib.request
cid, key, m = os.environ.get("CID", ""), os.environ.get("KEY", ""), os.environ["M"]
if not cid or not key: print("nokey"); raise SystemExit
url = "https://console.vast.ai/api/v0/instances/%s/" % cid
data, meth = None, m
if m == "STOP": data, meth = json.dumps({"state": "stopped"}).encode(), "PUT"
req = urllib.request.Request(url, data=data, method=meth, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
try: print(urllib.request.urlopen(req, timeout=30).status)
except urllib.error.HTTPError as e: print(e.code)
except Exception as e: print("err-" + type(e).__name__)'
}
age(){ echo $(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || date +%s) )); }
eff_dl(){   # the hard deadline, tightened (never extended) by status/guard_deadline
  local d=$DL0 t; t=$(tr -dc 0-9 2>/dev/null < "$S/guard_deadline")
  if [ -n "$t" ] && [ "$t" -gt 0 ] && { [ "$d" -le 0 ] || [ "$t" -lt "$d" ]; }; then d=$t; fi; echo "$d"; }
lanes_terminal(){   # every planned fold done / failed / skipped and no lane process alive
  local F p; [ -s "$R/kit/plan.env" ] || return 1
  local FOLDS; FOLDS=$(sed -n 's/^FOLDS="\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' "$R/kit/plan.env" | tail -n 1); [ -n "$FOLDS" ] || return 1
  for F in $FOLDS; do [ -e "$S/f$F.done" ] || [ -e "$S/f$F.failed" ] || [ -e "$S/f$F.skipped" ] || return 1; done
  for p in "$S"/lane_*.pid; do [ -e "$p" ] && kill -0 "$(cat "$p")" 2>/dev/null && return 1; done; return 0; }
gp=$(cat "$S/guard.pid" 2>/dev/null | tr -dc 0-9)   # review 10-14: after a container restart PIDs repeat (often our own)
if [ -n "$gp" ] && [ "$gp" != "$$" ] && kill -0 "$gp" 2>/dev/null \
   && [ "$(tr '\0' ' ' < /proc/$gp/cmdline 2>/dev/null)" = "$(tr '\0' ' ' < /proc/$$/cmdline)" ]; then exit 0; fi
echo $$ > "$S/guard.pid"
say "start pid $$ | hard deadline $(date -u -d "@$(eff_dl)" '+%F %T')Z | boot fuse $( [ "$BOOT" -gt 0 ] && date -u -d "@$BOOT" '+%F %T' || echo off)Z | key set: $( [ -n "$(envv CONTAINER_API_KEY)" ] && echo yes || echo NO) | mock $MOCK"
say "self-test: GET own instance with the per-instance key -> http $(api GET) (200 = the backstop can act)"
why=""
while [ -z "$why" ]; do
  now=$(date +%s); DL=$(eff_dl)
  if [ "$DL" -gt 0 ] && [ "$now" -ge "$DL" ]; then why="hard deadline reached"; fi
  [ -e "$S/ABORT" ] && why="ABORT: $(head -c 200 "$S/ABORT" | tr '\n' ' ')"
  if [ -e "$S/FINAL_FETCH_OK" ] && [ "$(age "$S/FINAL_FETCH_OK")" -ge "$FIN_AGE" ]; then why="box final fetch done 15+ min ago"; fi
  if [ "$BOOT" -gt 0 ] && [ "$now" -ge "$BOOT" ] && [ ! -e "$S/RUN_STARTED" ]; then why="bootstrap fuse: no RUN_STARTED by the boot deadline"; fi
  if [ -z "$why" ] && [ ! -e "$S/FINAL_FETCH_OK" ] && lanes_terminal; then
    [ -e "$S/lanes.done" ] || { date +%s > "$S/lanes.done"; say "all planned folds terminal, no lane alive: lanes.done"; }
    ref="$S/box_seen"; [ -e "$ref" ] || ref="$S/lanes.done"
    if [ "$(age "$ref")" -ge "$ORPHAN_AGE" ] && [ "$(age "$S/lanes.done")" -ge "$ORPHAN_AGE" ]; then why="orphan fuse: lanes done, no box fetch for $((ORPHAN_AGE / 60)) min"; fi
  fi
  [ -z "$why" ] && sleep "$POLL"
done
say "TRIGGER: $why -> stop trainers, wipe data, self-destroy"
if [ "$MOCK" = 1 ]; then say "mock: would pkill rstud_lane.sh and fleet_train.py"
else for p in "$S"/lane_*.pid "$S"/f*.tpid; do [ -e "$p" ] && kill "$(cat "$p")" 2>/dev/null; done
  pkill -f 'rstud_lane[.]sh'; pkill -f 'fleet_train[.]py'; sleep 10; fi
rm -rf "$R/fleet_store" "$R/fleet_runs" "$R/hf" "$R/r7t_labels" "$R/rsna_train" "$R/fast_build" "$R/kit" "$R/incoming"
say "wiped: left $(ls "$R" | tr '\n' ' ')"
for i in $(seq 1 "$TRIES"); do
  c=$(api DELETE); say "destroy attempt $i: http $c"
  case $c in 200|204|mock-DELETE) exit 0;; esac
  c=$(api STOP); say "stop attempt $i: http $c"
  sleep "$RETRY_S"
done
say "GAVE UP: the per-instance key could not destroy or stop this instance; only the laptop (2FA) can"
