#!/bin/bash
# config.sh: shared settings + helpers for the RSNA-ONLY Raptor seed rental (path b). Sourced, never run. Laptop side (Git Bash).
# Pattern = session_2026-10-09/vastL5 (vast_raw wrapper, state.env, guard). Secrets: the Vast API key stays in the vastai
# CLI config; only the box's dedicated PUBLIC key (~/.ssh/vast_rsna_ed25519.pub) leaves the box. NOTHING from the box is
# ever pushed to the rental: code + RSNA label csvs go laptop -> instance, the RSNA corpus comes from Kaggle signed URLs.
set -u
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RAP=$(cd "$KIT/.." && pwd)                          # session_2026-10-13/raptor
VAST_BIN=${VAST_BIN:-/c/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/vastai}
export PYTHONIOENCODING=utf-8
PYL=${PYL:-python}
BOX=${BOX:-training@192.168.10.30}
R=/workspace/rsna
IMAGE=${IMAGE:-pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime}   # torch 2.5.1 = the box
DISK_GB=${DISK_GB:-200}                             # 37 GB tars + 59 GB npy + 3 x 4.7 GB epoch ckpts + image
CAP_USD=${CAP_USD:-23.0}                            # main 10-13: hard total cap incl. data-in and failed attempts
FLOOR_USD=${FLOOR_USD:-2.0}                         # credit kept as buffer
RESERVE_USD=${RESERVE_USD:-1.0}                     # kept back for fetch + destroy
MAX_RUN_H=${MAX_RUN_H:-12.0}                        # never plan a longer rental
SEEDS=${SEEDS:-"43 44 45"}
STATE=$KIT/state.env; LOGDIR=$KIT/logs; KH=$KIT/known_hosts_vast
mkdir -p "$LOGDIR"
ts(){ date '+%m-%d %H:%M:%S'; }
log(){ local m; m="[$(ts)] $*"; printf '%s\n' "$m"; printf '%s\n' "$m" >> "$LOGDIR/${SCRIPT_TAG:-kit}.log"; }
die(){ log "ERROR: $*"; exit "${2:-1}"; }
st_get(){ [ -f "$STATE" ] && sed -n "s/^$1=//p" "$STATE" | tail -n 1; return 0; }
st_set(){ touch "$STATE"; grep -v "^$1=" "$STATE" > "$STATE.tmp" || true; printf '%s=%s\n' "$1" "$2" >> "$STATE.tmp"; mv -f "$STATE.tmp" "$STATE"; }
BOX_SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
bx(){ ssh "${BOX_SSH_OPTS[@]}" "$BOX" "$@" < /dev/null; }
ix(){
  local h p; h=$(st_get I_HOST); p=$(st_get I_PORT)
  [ -n "$h" ] && [ -n "$p" ] || { echo "no instance route in state.env" >&2; return 255; }
  ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o LogLevel=ERROR -p "$p" "root@$h" "$@"
}
iscp(){ scp -q -o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o LogLevel=ERROR -P "$(st_get I_PORT)" "$@"; }
vast_raw(){
  local out
  out=$("$VAST_BIN" "$@" --raw 2>&1)
  if printf '%s' "$out" | grep -q "Two Factor Authentication"; then
    echo "VAST: this call needs a 2FA session (Eesh: vastai tfa send-email, then vastai tfa login -t email -s <secret> -c <code>)" >&2
    return 3
  fi
  printf '%s' "$out" | "$PYL" -c '
import sys, json, re
t = sys.stdin.read()
i = [k for k in (t.find("["), t.find("{")) if k >= 0]
def bad(m):
    sys.stderr.write("VAST: " + re.sub(r"[A-Za-z0-9]{24,}", "<redacted>", m)[:300] + "\n"); sys.exit(2)
if not i: bad("no JSON in CLI output: " + (t.strip().splitlines()[-1] if t.strip() else "empty CLI output"))
try: d = json.loads(t[min(i):])
except Exception: bad("unparsable CLI output")
if isinstance(d, dict) and d.get("error"): bad("error %s: %s" % (d.get("status_code"), d.get("msg")))
json.dump(d, sys.stdout)'
}
jq_py(){ "$PYL" -c "import sys, json; d = json.load(sys.stdin); $1"; }
credit_now(){ local j; j=$(vast_raw show user) || return $?; printf '%s' "$j" | jq_py 'print(d.get("credit", ""))'; }
num_ge(){ "$PYL" -c "import sys; sys.exit(0 if float('$1') >= float('$2') else 1)"; }
