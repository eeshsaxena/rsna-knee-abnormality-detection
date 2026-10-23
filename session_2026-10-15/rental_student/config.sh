#!/bin/bash
# config.sh (laptop, Git Bash; sourced, never run): settings + helpers for the rental_student kit (T_b student, seed 1047,
# 5 TTR224 folds, RSNA-only). Pattern = session_2026-10-13/raptor/vast/config.sh (vast_raw wrapper, state.env).
# Secrets: the Vast API key stays in the vastai CLI config; only the box's dedicated PUBLIC key (~/.ssh/vast_rsna_ed25519.pub)
# leaves the box. Money: CAP_USD is the hard TOTAL cap of this kit incl. failed attempts and data-in, measured as the credit
# drop since ledger.env CREDIT_START (recorded once, before the first rental of this kit).
set -u
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VAST_BIN=${VAST_BIN:-/c/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/vastai}
export PYTHONIOENCODING=utf-8
PYL=${PYL:-python}
BOX=${BOX:-training@192.168.10.30}
R=/workspace/rsna
IMAGE=${IMAGE:-pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime}   # torch 2.5.1 = the box rsna env
DISK_GB=${DISK_GB:-80}            # bundle 11.8 GB + 5 folds x ~3.9 GB (epoch ckpts) + pip ~2 GB + image
CAP_USD=${CAP_USD:-15.0}          # workflow rule 10-14: hard total cap incl. failed attempts and data transfer
FLOOR_USD=${FLOOR_USD:-1.5}       # credit that must remain after the worst case
RESERVE_USD=${RESERVE_USD:-0.5}   # kept back from the guard line for destroy latency + storage after stop
MAX_RUN_H=${MAX_RUN_H:-7.0}       # never plan a longer rental
BOOT_MAX_MIN=${BOOT_MAX_MIN:-100} # guard bootstrap fuse: no RUN_STARTED (push + setup + smoke) by then -> self-destroy
STATE=$KIT/state.env; LEDGER=$KIT/ledger.env; LOGDIR=$KIT/logs; KH=$KIT/known_hosts_vast
mkdir -p "$LOGDIR"
ts(){ date '+%m-%d %H:%M:%S'; }
log(){ local m; m="[$(ts)] $*"; printf '%s\n' "$m"; printf '%s\n' "$m" >> "$LOGDIR/${SCRIPT_TAG:-kit}.log"; }
die(){ log "ERROR: $*"; exit "${2:-1}"; }
kv_get(){ [ -f "$1" ] && sed -n "s/^$2=//p" "$1" | tail -n 1; return 0; }
kv_set(){ touch "$1"; grep -v "^$2=" "$1" > "$1.tmp" || true; printf '%s=%s\n' "$2" "$3" >> "$1.tmp"; mv -f "$1.tmp" "$1"; }
st_get(){ kv_get "$STATE" "$1"; }; st_set(){ kv_set "$STATE" "$1" "$2"; }
BOX_SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
bx(){ ssh "${BOX_SSH_OPTS[@]}" "$BOX" "$@" < /dev/null; }
bx_in(){ ssh "${BOX_SSH_OPTS[@]}" "$BOX" "$@"; }
ix(){
  local h p; h=$(st_get I_HOST); p=$(st_get I_PORT)
  [ -n "$h" ] && [ -n "$p" ] || { echo "no instance route in state.env" >&2; return 255; }
  ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o LogLevel=ERROR -p "$p" "root@$h" "$@"
}
vast_raw(){   # JSON on stdout, never the key; rc 3 = a 2FA session is needed, rc 2 = other error
  local out
  out=$("$VAST_BIN" "$@" --raw 2>&1)
  if printf '%s' "$out" | grep -q -i "Two Factor"; then
    echo "VAST: this call needs a 2FA session. Eesh: vastai tfa send-email, then vastai tfa login -t email -s <secret> -c <code>" >&2
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
instances_json(){ vast_raw show instances; }
n_instances(){ instances_json | jq_py 'print(len(d if isinstance(d, list) else d.get("instances", [])))'; }
spent_total(){   # $ spent by this kit so far = ledger CREDIT_START - credit now (0 before the first rental)
  local c0 c; c0=$(kv_get "$LEDGER" CREDIT_START); [ -n "$c0" ] || { echo 0; return 0; }
  c=$(credit_now) || return $?; "$PYL" -c "print(round(max(0.0, $c0 - $c), 3))"; }
num_ge(){ "$PYL" -c "import sys; sys.exit(0 if float('$1') >= float('$2') else 1)"; }
