#!/bin/bash
# guard.sh: on-instance BACKSTOP (started by the Vast onstart at every container start; vast_launch.sh embeds it).
# Self-destroys with the per-instance key Vast injects (CONTAINER_ID / CONTAINER_API_KEY, never printed) when:
#   (a) the hard deadline RSNA_GUARD_DEADLINE (epoch, fixed at create time from the budget) passes, or
#   (b) status/FINAL_FETCH_OK (written by the BOX after it pulled and sha-verified every seed) exists for 15+ min.
# Copy of session_2026-10-09/vastL5/remote/guard.sh with the Raptor paths.
R=${RSNA_ROOT:-/workspace/rsna}; S=$R/status; TRIES=6; RETRY_S=120; LOG=$R/guard.log; mkdir -p "$S"
[ -s "$S/guard_deadline" ] || echo "${RSNA_GUARD_DEADLINE:-0}" > "$S/guard_deadline"
DL=$(cat "$S/guard_deadline")
say(){ echo "[guard $(date -u '+%m-%d %H:%M:%S')Z] $*" >> "$LOG"; }
envv(){ local v; v=$(printenv "$1"); [ -n "$v" ] || v=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -n 1); printf '%s' "$v"; }
PYG=$(command -v python3 || command -v python || echo /opt/conda/bin/python)
api(){
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
if [ -s "$S/guard.pid" ] && kill -0 "$(cat "$S/guard.pid")" 2>/dev/null; then exit 0; fi
echo $$ > "$S/guard.pid"
say "start pid $$ | hard deadline $(date -u -d "@$DL" '+%F %T')Z | key set: $( [ -n "$(envv CONTAINER_API_KEY)" ] && echo yes || echo NO)"
say "self-test: GET own instance -> http $(api GET) (200 = the backstop can act)"
why=""
while [ -z "$why" ]; do
  now=$(date +%s)
  if [ "$DL" -gt 0 ] && [ "$now" -ge "$DL" ]; then why="hard deadline reached"; fi
  if [ -e "$S/FINAL_FETCH_OK" ] && [ $(( now - $(stat -c %Y "$S/FINAL_FETCH_OK") )) -ge 900 ]; then why="box final fetch done 15+ min ago"; fi
  [ -z "$why" ] && sleep 60
done
say "TRIGGER: $why -> stop trainers, wipe data, self-destroy"
pkill -f 'lanes[.]sh'; pkill -f 'train_raptor[.]py'; sleep 10
rm -rf "$R/rsna" "$R/in" "$R/runs" "$R/hf" "$R/kit" "$R/rsna_index.csv"
say "wiped: $(ls "$R" | tr '\n' ' ')"
for i in $(seq 1 $TRIES); do
  c=$(api DELETE); say "destroy attempt $i: http $c"
  case $c in 200|204) exit 0;; esac
  c=$(api STOP); say "stop attempt $i: http $c"
  sleep $RETRY_S
done
say "GAVE UP: only the laptop (2FA) can destroy this instance"
