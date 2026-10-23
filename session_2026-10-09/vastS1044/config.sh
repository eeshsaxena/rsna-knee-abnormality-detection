#!/bin/bash
# config.sh: shared settings and helpers for the vastS1044 kit (TTR224 frac75 seed 1044 folds 1-4; copy of
# session_2026-10-08/vast320 with its own state.env). Sourced by every laptop script, never run directly.
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
DISK_GB=${DISK_GB:-45}                               # store 11.7 GB + 4 folds x ~3.6 GB epoch ckpts (224) + pip ~1 GB = ~28 GB; S1 needs >= 32 GB free
CAP_USD=${CAP_USD:-2.10}                             # hard cap on TOTAL spend for this run (coordinator 10-09 23:40: run cap $2.10, budget = credit - $0.15)
FLOOR_USD=${FLOOR_USD:-0.15}                         # credit that must remain (coordinator 10-09 15:20: remaining credit minus $0.15, still capped at $2.30)
RESERVE_USD=${RESERVE_USD:-0.03}                     # spend line of the on-instance guard = budget - reserve (destroy latency + data-out); watcher cap is GUARD_MARGIN_H earlier
GUARD_MARGIN_H=${GUARD_MARGIN_H:-0.15}               # on-instance self-destroy fires this long after the watcher cap (3-min polls near the cap + fetch ~4 min + destroy ~1 min)
LANE_MARGIN_S=${LANE_MARGIN_S:-360}                   # a lane may start a fold only if it ends this long before the watcher cap
MAX_RUN_H=${MAX_RUN_H:-14.0}                         # sequential folds on ONE GPU: the budget, not this, is the binding limit
POLL_S=${POLL_S:-600}                                # watcher poll period (10 min)
SEED=$(sed -n 's/^SEED=//p' "$KIT/remote/kit.env"); RUNTAG=$(sed -n 's/^RUNTAG=//p' "$KIT/remote/kit.env")   # 1044 / ttr_frac75 (remote/kit.env)
V100_FOLD_H=${V100_FOLD_H:-3.52}                     # MEASURED box V100 time of one TTR224 gc0 fold, 21 ep (f2_s1043_ttr_mix50: 10.07 min/ep)
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
BOXDIR=vastS1044                                     # box-side helper dir ~/vastS1044 (never shared with ~/vast320 or ~/vastL5)
sync_box_ops(){   # copy remote/box_ops.sh + parity.py + box_push.sh + argv_template.txt to ~/$BOXDIR on the box
  local f
  bx "mkdir -p ~/$BOXDIR && chmod 700 ~/$BOXDIR" || return 1
  for f in box_ops.sh box_push.sh parity.py argv_template.txt lane.sh status.sh kit.env; do
    bx_in "cat > ~/$BOXDIR/$f.tmp && mv -f ~/$BOXDIR/$f.tmp ~/$BOXDIR/$f" < "$KIT/remote/$f" || return 1
  done
}
bops(){ bx "bash ~/$BOXDIR/box_ops.sh" "$@"; }                  # run a box_ops.sh subcommand

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

# ---- the concurrent L5 run (session_2026-10-09/vastL5): READ ONLY, never touched ----
# Both runs draw on the same Vast credit. L5's watcher finalises (fetches + destroys L5) when the credit falls to
# L5_TRIG = max(L5 CREDIT0 - (L5 BUDGET - 0.40), 0.50 + 0.40) (~$0.909). This kit therefore (a) subtracts L5's spend
# from the credit drop when it measures its own spend, (b) refuses a launch whose worst case could push the credit to
# L5_TRIG + L5_MARGIN before L5_END_BY, and (c) finalises ITS OWN run (never L5) if the credit nears that line while L5
# is still alive.
L5_STATE=${L5_STATE:-$KIT/../vastL5/state.env}
L5_RESERVE_USD=${L5_RESERVE_USD:-0.10}               # extra held back on top of L5's MODEL remaining spend (l5_holdback) while L5 is alive
L5_MARGIN=${L5_MARGIN:-0.20}                         # keep the credit this far above L5's trigger while L5 is alive
L5_END_BY=${L5_END_BY:-$(date -d 'today 17:15' +%s)} # planning bound for L5's destroy (MODEL ~16:40 IST: ep8/21 at 13:44, 12.1 min/ep, + fetch); the watcher guards live
l5_get(){ [ -f "$L5_STATE" ] && sed -n "s/^$1=//p" "$L5_STATE" | tail -n 1; return 0; }
l5_alive(){ [ -n "$(l5_get INSTANCE_ID)" ] && [ "$(l5_get DESTROYED)" != 1 ] && [ -z "$(st_get L5_GONE_AT)" ]; }   # L5_GONE_AT: our watcher saw it leave the account
l5_holdback(){   # $ held back from this run's budget while L5 lives: L5 $/h x hours to L5_END_BY + L5_RESERVE_USD (0 once L5 is gone)
  l5_alive || { echo 0; return; }
  "$PYL" -c "import time; print(round($(l5_get DPH) * max(0.0, ($L5_END_BY - time.time()) / 3600) + $L5_RESERVE_USD, 3))"; }
l5_trig(){ local c0 b; c0=$(l5_get CREDIT0); b=$(l5_get BUDGET); [ -n "$c0" ] && [ -n "$b" ] || { echo 0.95; return; }
  "$PYL" -c "print(round(max($c0 - ($b - 0.40), 0.90), 4))"; }
l5_end_epoch(){   # when L5 stopped billing (its DESTROYED_AT, or when our watcher saw it gone), or now if still alive
  local d g; if l5_alive; then date +%s; return; fi
  g=$(st_get L5_GONE_AT); d=$(l5_get DESTROYED_AT); [ -n "$d" ] && d=$(date -d "$d" +%s 2>/dev/null)
  if [ -n "$g" ] && { [ -z "$d" ] || [ "$g" -lt "$d" ]; }; then echo "$g"; elif [ -n "$d" ]; then echo "$d"; else date +%s; fi; }
l5_spend_since(){  # $1 = epoch: L5's billing since then (its DPH x overlap), used to separate the two runs in the credit drop
  local dph e; dph=$(l5_get DPH); [ -n "$dph" ] || { echo 0; return; }; e=$(l5_end_epoch)
  "$PYL" -c "print(round(max(0, $e - $1) / 3600 * $dph, 4))"; }
