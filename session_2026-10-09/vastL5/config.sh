#!/bin/bash
# config.sh: shared settings and helpers for the vastL5 kit (L5 all-24-slot TTR288 f0 probe; copy of session_2026-10-08/vast320). Sourced, never run.
# Laptop side (Git Bash). Box = the V100 server (LAN). Instance = the rented Vast machine.
# Secrets: the Vast API key stays in ~/.config/vastai (read only by the vastai CLI); nothing here prints it.
#          Only the box's dedicated PUBLIC key (~/.ssh/vast_rsna_ed25519.pub) ever leaves the box.
set -u
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VAST_BIN=${VAST_BIN:-/c/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/vastai}
export PYTHONIOENCODING=utf-8
PYL=${PYL:-python}                                   # laptop python (JSON parsing only)
BOX=${BOX:-training@192.168.10.30}
R=/workspace/rsna                                    # instance root; mirrors the box layout under it
IMAGE=${IMAGE:-pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime}
DISK_GB=${DISK_GB:-60}                               # store 11.7 GB + 1 fold with epoch ckpts (~4 GB) + image; S1 needs >= 40 GB free
CAP_USD=${CAP_USD:-3.50}                             # hard cap on TOTAL spend for this run (Eesh 10-09: L5 move, cap $3.50)
FLOOR_USD=${FLOOR_USD:-0.50}                         # credit that must remain
RESERVE_USD=${RESERVE_USD:-0.40}                     # kept back for fetch + destroy after a cap trigger
GUARD_MARGIN_H=${GUARD_MARGIN_H:-0.5}                # on-instance self-destroy fires this long after the watcher cap (fetch + destroy take ~15 min)
MAX_RUN_H=${MAX_RUN_H:-9.0}                          # never plan a longer rental than this
POLL_S=${POLL_S:-600}                                # watcher poll period (10 min)
SEED=1045; RUNTAG=$(sed -n 's/^RUNTAG=//p' "$KIT/remote/kit.env")   # ttr288frac75 (remote/kit.env)
V100_FOLD_H=${V100_FOLD_H:-8.1}                      # MODEL V100 time of the 24-slot fold (16-slot MEASURED 5.8 h x 1.4)
STATE=$KIT/state.env; LOGDIR=$KIT/logs; KH=$KIT/known_hosts_vast
mkdir -p "$LOGDIR"

run_name(){ echo "f${1}_s${SEED}_${RUNTAG}"; }
ts(){ date '+%m-%d %H:%M:%S'; }
log(){ local m; m="[$(ts)] $*"; printf '%s\n' "$m"; printf '%s\n' "$m" >> "$LOGDIR/${SCRIPT_TAG:-kit}.log"; }
die(){ log "ERROR: $*"; exit "${2:-1}"; }
num_ge(){ "$PYL" -c "import sys; sys.exit(0 if float('$1') >= float('$2') else 1)"; }   # float a >= b

# ---- state.env: KEY=VALUE lines, read with st_get (never sourced) ----
st_get(){ [ -f "$STATE" ] && sed -n "s/^$1=//p" "$STATE" | tail -n 1; return 0; }
st_set(){ touch "$STATE"; grep -v "^$1=" "$STATE" > "$STATE.tmp" || true; printf '%s=%s\n' "$1" "$2" >> "$STATE.tmp"; mv -f "$STATE.tmp" "$STATE"; }

# ---- box (LAN) ----
BOX_SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
bx(){ ssh "${BOX_SSH_OPTS[@]}" "$BOX" "$@" < /dev/null; }       # no stdin
bx_in(){ ssh "${BOX_SSH_OPTS[@]}" "$BOX" "$@"; }                # with stdin
box_up(){ bx true 2>/dev/null; }
sync_box_ops(){   # copy remote/box_ops.sh + parity.py + box_push.sh + argv_template.txt to ~/vastL5 on the box
  local f
  bx 'mkdir -p ~/vastL5 && chmod 700 ~/vastL5' || return 1
  for f in box_ops.sh box_push.sh parity.py argv_template.txt lane.sh status.sh kit.env; do
    bx_in "cat > ~/vastL5/$f.tmp && mv -f ~/vastL5/$f.tmp ~/vastL5/$f" < "$KIT/remote/$f" || return 1
  done
}
bops(){ bx 'bash ~/vastL5/box_ops.sh' "$@"; }                    # run a box_ops.sh subcommand

# ---- instance (Vast) from the laptop: route chosen by vast_launch.sh ----
ix(){
  local h p; h=$(st_get I_HOST); p=$(st_get I_PORT)
  [ -n "$h" ] && [ -n "$p" ] || { echo "no instance route in state.env" >&2; return 255; }
  ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o LogLevel=ERROR -p "$p" "root@$h" "$@"
}

# ---- vastai CLI wrapper: JSON on stdout, never the key; rc 3 = a 2FA session is needed, rc 2 = other error ----
vast_raw(){
  local out
  out=$("$VAST_BIN" "$@" --raw 2>&1)
  if printf '%s' "$out" | grep -q "Two Factor Authentication"; then
    echo "VAST: this call needs a 2FA session (vastai tfa send-email, then Eesh runs vastai tfa login; RUNBOOK step 0)" >&2
    return 3
  fi
  printf '%s' "$out" | "$PYL" -c '
import sys, json, re
t = sys.stdin.read()
i = [k for k in (t.find("["), t.find("{")) if k >= 0]
def bad(m):
    sys.stderr.write("VAST: " + re.sub(r"[A-Za-z0-9]{24,}", "<redacted>", m)[:300] + "\n"); sys.exit(2)
if not i: bad("no JSON in CLI output: " + t.strip().splitlines()[-1] if t.strip() else "empty CLI output")
try:
    d = json.loads(t[min(i):])
except Exception:
    bad("unparsable CLI output")
if isinstance(d, dict) and d.get("error"):
    bad("error %s: %s" % (d.get("status_code"), d.get("msg")))
json.dump(d, sys.stdout)'
}
jq_py(){ "$PYL" -c "import sys, json; d = json.load(sys.stdin); $1"; }   # jq_py 'print(d["x"])' < json

credit_now(){ local j; j=$(vast_raw show user) || return $?; printf '%s' "$j" | jq_py 'print(d.get("credit", ""))'; }
inst_json(){ vast_raw show instance "$1"; }                                  # rc 3 = 2FA needed
