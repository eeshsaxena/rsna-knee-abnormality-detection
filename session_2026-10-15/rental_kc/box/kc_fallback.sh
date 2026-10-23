#!/bin/bash
# kc_fallback.sh (rental_kc, BOX, CPU only, no GPU; setsid nohup, survives laptop / Claude session loss). Adapted from
# rental_student/box/tb_fallback.sh. NEVER edits ~/queue: qctl-free by default. No em dashes.
#   bash ~/rkc/src/box/kc_fallback.sh --eval     print the state and what it would do (no writes)
#   bash ~/rkc/src/box/kc_fallback.sh --once     evaluate once and act
#   setsid nohup bash ~/rkc/src/box/kc_fallback.sh --loop > /dev/null 2>&1 < /dev/null &   (every 120 s)
# Rental ALIVE = ~/rkc/run.env exists, its driver.sh or fetch_loop.sh pid is alive (cmdline checked), and no ~/rkc/ABORTED,
# FINAL_FETCH_OK or FETCH_GAVE_UP is newer than run.env. While alive: only members the fetch loop already gave back
# (resolved, not ok). When it is NOT alive, for every member the rental reserved (~/rkc/state_<IID>/members):
#   - an orphan EMPTY placeholder ~/fleet_runs/<out> (no .log) is released (rmdir), so a still ACTIVE box line can train it;
#   - a member with no out dir / log whose box line is no longer ACTIVE (the runner REFUSED it while the rental owned it)
#     is LOST for the box: the exact `qctl top` command that restores its original line (name | out | original gate | argv)
#     is written to ~/rkc/REQUEUE_CMDS.sh and logged as REQUEUE NEEDED. With KC_FALLBACK_QCTL=1 (main 10-15 15:05: armed,
#     as tb_fallback 10-14) it runs that qctl top itself and logs every call (QCTL CALL line + rc); else main runs it.
# Exits when the rental is over and every reserved member is delivered, released or listed for re-queue, or at the training
# freeze (10-20 12:00 IST). Log ~/rkc/logs/kc_fallback.log.
set -u
D=${RKC_HOME:-$HOME/rkc}; SRC=${RKC_SRC:-$HOME/rkc/src}; FR=${RKC_FR:-$HOME/fleet_runs}; Q=${RKC_Q:-$HOME/queue/queue.txt}
QCTL=${KCF_QCTL:-bash $HOME/queue/qctl.sh}; LOG=${KCF_LOG:-$D/logs/kc_fallback.log}; RQ=$D/REQUEUE_CMDS.sh
FREEZE=$(date -d '2026-10-20 12:00:00 +0530' +%s); MODE=${1:---eval}; OPS="bash $SRC/box/ops.sh"
mkdir -p "$(dirname "$LOG")"
say(){ echo "[kcf $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; [ "$MODE" = --loop ] || echo "[kcf] $*"; }
alive_pid(){ local p; [ -s "$1" ] || return 1; p=$(cat "$1" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null || return 1
  tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q -- "$2"; }
state(){   # -> ALIVE (0|1) WHY IID
  local RE=$D/run.env m newer="" live=""
  IID=""; ALIVE=0
  if [ ! -f "$RE" ]; then WHY="no rental run.env"; return; fi
  IID=$(sed -n 's/^INSTANCE_ID=//p' "$RE" | tail -n 1)
  for m in ABORTED FINAL_FETCH_OK FETCH_GAVE_UP; do [ -e "$D/$m" ] && [ "$D/$m" -nt "$RE" ] && newer="$newer $m"; done
  alive_pid "$D/driver.pid" driver.sh && live="driver"; alive_pid "$D/fetch/fetch.pid" fetch_loop.sh && live="$live fetch"
  if [ -n "$newer" ]; then WHY="rental $IID ended:$newer"
  elif [ -n "$live" ]; then ALIVE=1; WHY="rental $IID alive:$live"
  else WHY="rental $IID: no live driver / fetch loop and no end marker (died)"; fi
}
tick(){
  state; ALLDONE=1
  local MF=$D/state_$IID/members k n out line st cmd r
  if [ $ALIVE = 1 ]; then [ "$MODE" = --loop ] || say "ALIVE: $WHY"; ALLDONE=0; fi
  [ -s "$MF" ] || { [ $ALIVE = 1 ] || say "$WHY; no reserved members (nothing to do)"; return; }
  for k in $(cat "$MF"); do
    n=$($OPS out $k); out=$FR/$n
    if [ $ALIVE = 1 ]; then   # rental alive: act only on a member its fetch loop already gave back (resolved, not ok)
      r=$(head -c 200 $D/fetch/resolved_$k 2>/dev/null); [ -n "$r" ] && [ "$r" != ok ] || continue
    fi
    if [ -d "$out" ] && [ -z "$(ls -A "$out")" ] && [ ! -e "$out.log" ]; then
      if [ "$MODE" = --eval ]; then say "WOULD release the orphan EMPTY placeholder $out"; ALLDONE=0; continue; fi
      rmdir "$out" && say "$k: released the orphan EMPTY placeholder $out ($WHY)" || { ALLDONE=0; continue; }
    fi
    if [ -e "$out" ] || [ -e "$out.log" ]; then continue; fi                      # delivered (or the box trains it)
    st=$($OPS line-state $k)
    case $st in ACTIVE|RUN) continue;; esac                                          # the box line will run / runs it
    grep -qF "| ~/fleet_runs/$n |" "$RQ" 2>/dev/null && continue                       # already listed
    line=$(grep -E "\| ~/fleet_runs/$n \|" "$Q" | tail -n 1 | sed -E 's/^#[A-Z]+ [0-9-]+( [0-9:]+)? (\([^)]*\) )?//')
    [ -n "$line" ] || { say "$k: lost and no queue line text to restore (state $st): main must re-add it from KC_QCTL_1015.txt"; continue; }
    cmd="bash ~/queue/qctl.sh top '$line'"
    if [ "$MODE" = --eval ]; then say "WOULD write REQUEUE for $k (line $st): ${cmd:0:160}..."; continue; fi
    [ -f "$RQ" ] || printf '#!/bin/bash\n# REQUEUE_CMDS.sh (kc_fallback.sh): members the rental reserved and did not deliver, whose box line was REFUSED.\n# Each restores the original line as the next job; kc_fallback.sh runs it itself only with KC_FALLBACK_QCTL=1 (main 10-15).\n' > "$RQ"
    printf '# %s %s (%s; line was %s)\n%s\n' "$(date '+%F %T')" "$k" "$WHY" "$st" "$cmd" >> "$RQ"
    say "REQUEUE NEEDED $k ($n, line $st): written to $RQ"
    if [ "${KC_FALLBACK_QCTL:-0}" = 1 ]; then
      say "QCTL CALL (armed by main 10-15 15:05, KC_FALLBACK_QCTL=1): $cmd"
      if $QCTL top "$line" >> "$LOG" 2>&1; then say "RE-QUEUED $k via qctl top (rc 0)"; echo "# $(date '+%F %T') executed by kc_fallback.sh (rc 0)" >> "$RQ"
      else say "qctl top for $k refused (rc != 0, see above)"; echo "# $(date '+%F %T') kc_fallback.sh qctl call FAILED: run it by hand" >> "$RQ"; fi
    fi
  done
}
case $MODE in
  --eval|--once) tick; exit 0;;
  --loop)
    if [ -s $D/kc_fallback.pid ] && [ "$(cat $D/kc_fallback.pid)" != $$ ] && alive_pid $D/kc_fallback.pid kc_fallback.sh; then echo "already running"; exit 0; fi
    echo $$ > $D/kc_fallback.pid; say "== loop start pid $$ | queue $Q (read only) | qctl armed: ${KC_FALLBACK_QCTL:-0}"
    while :; do
      tick
      [ "$ALIVE" = 0 ] && [ "$ALLDONE" = 1 ] && [ -n "$IID" ] && { say "rental over and every reserved member resolved: loop exits ($WHY)"; break; }
      [ $(date +%s) -ge $FREEZE ] && { say "training freeze: loop exits"; break; }
      sleep ${KCF_POLL_S:-120}
    done
    rm -f $D/kc_fallback.pid;;
  *) sed -n '2,19p' "$0"; exit 2;;
esac
