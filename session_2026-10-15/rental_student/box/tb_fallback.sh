#!/bin/bash
# tb_fallback.sh (BOX, CPU only, no GPU; setsid nohup, survives laptop / Claude session loss). 10-14 21:40 IST, coordinator
# request: automatic box fallback for the T_b student folds f0-f3 (seed 1047) while the Vast rental owns them.
#   bash ~/rstud/src/box/tb_fallback.sh --eval     print the state + the verdict it would write (no writes)
#   bash ~/rstud/src/box/tb_fallback.sh --once     evaluate once, write the verdict file, re-queue missing folds
#   setsid nohup bash ~/rstud/src/box/tb_fallback.sh --loop > /dev/null 2>&1 < /dev/null &   (every 120 s)
# Verdict file ~/r7t_labels/final/BOX_TB_FALLBACK.verdict (read by the queue runner: the box T_b f0-f3 lines are gated
# `needs PASS from` it). Written atomically, only when it changes:
#   FAIL = a rental currently owns T_b f0-f3: the current ~/rstud/run.env has a live driver.sh or fetch_loop.sh process
#          and no ~/rstud/ABORTED, FINAL_FETCH_OK or FETCH_GAVE_UP newer than run.env (launch phase, smoke, training,
#          fetch). The runner then skips the box lines (a rental placeholder dir also makes them #REFUSED).
#   PASS = no rental owns them: no run.env, or the rental aborted / gave up / finished, or its driver and fetch loop died.
# Re-queue (only on PASS): a fold k in 0-3 with no ~/fleet_runs/f{k}_s1047_ttr_r7tb (nor .log) and no uncommented queue line
# for it (its line was skipped / refused while a rental owned it, and the rental did not deliver it) gets its line back as
# the NEXT job via `qctl top` (argv byte-identical to the frozen sheet ~/r7t_labels/sheets/queue_r7tb.txt, gate = this
# verdict file). The rental side needs nothing new: driver D5 never reserves a fold whose out dir the box already owns.
# Exits when f0-f3 all have out dirs and no rental is alive, or at the training freeze (10-20 12:00 IST).
set -u
D=${RSTUD_HOME:-$HOME/rstud}; FR=${TBF_FR:-$HOME/fleet_runs}; QCTL=${TBF_QCTL:-bash $HOME/queue/qctl.sh}
QUEUE=${QR_QUEUE:-$HOME/queue/queue.txt}; SHEET=${TBF_SHEET:-$HOME/r7t_labels/sheets/queue_r7tb.txt}
V=${TBF_VERDICT:-$HOME/r7t_labels/final/BOX_TB_FALLBACK.verdict}; LOG=${TBF_LOG:-$D/logs/tb_fallback.log}
GATE="needs PASS from ~/r7t_labels/final/BOX_TB_FALLBACK.verdict wait 30"
FREEZE=$(date -d '2026-10-20 12:00:00 +0530' +%s); MODE=${1:---eval}
mkdir -p "$(dirname "$LOG")"
say(){ echo "[tbf $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; [ "$MODE" = --loop ] || echo "[tbf] $*"; }
alive_pid(){   # pidfile pattern -> 0 if that pid is alive and its command line matches
  local p; [ -s "$1" ] || return 1; p=$(cat "$1" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null || return 1
  tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q -- "$2"; }
state(){   # -> ST_VERDICT ST_WHY
  local RE=$D/run.env m newer=""
  if [ ! -f "$RE" ]; then ST_VERDICT=PASS; ST_WHY="no rental run.env"; return; fi
  for m in ABORTED FINAL_FETCH_OK FETCH_GAVE_UP; do [ -e "$D/$m" ] && [ "$D/$m" -nt "$RE" ] && newer="$newer $m"; done
  local IID; IID=$(sed -n 's/^INSTANCE_ID=//p' "$RE" | tail -n 1)
  local live=""; alive_pid "$D/driver.pid" driver.sh && live="driver"; alive_pid "$D/fetch/fetch.pid" fetch_loop.sh && live="$live fetch"
  local rs=no; [ -e "$D/state_$IID/run.ok" ] && rs=yes
  if [ -n "$newer" ]; then ST_VERDICT=PASS; ST_WHY="rental $IID ended:$newer newer than run.env (RUN_STARTED $rs)"
  elif [ -n "$live" ]; then ST_VERDICT=FAIL; ST_WHY="rental $IID owns T_b f0-f3: live$live, RUN_STARTED $rs, no ABORTED / FINAL_FETCH_OK / FETCH_GAVE_UP"
  else ST_VERDICT=PASS; ST_WHY="rental $IID: no live driver / fetch loop and no end marker (died), RUN_STARTED $rs"; fi
}
requeue(){   # on PASS: give back missing folds as the next jobs (f3 first so f0 ends up on top)
  local k out line arg
  for k in 3 2 1 0; do
    out=$FR/f${k}_s1047_ttr_r7tb
    if [ -d "$out" ] && [ -z "$(ls -A "$out")" ] && [ ! -e "$out.log" ]; then   # orphan EMPTY rental placeholder (no rental alive)
      if [ "$MODE" = --eval ]; then say "WOULD release the orphan EMPTY placeholder $out"; continue; fi
      rmdir "$out" && say "released the orphan EMPTY placeholder $out (no rental alive)" || continue
    fi
    [ -e "$out" ] || [ -e "$out.log" ] && continue
    grep -v '^[[:space:]]*#' "$QUEUE" | grep -q "| ~/fleet_runs/f${k}_s1047_ttr_r7tb |" && continue
    line=$(grep "| ~/fleet_runs/f${k}_s1047_ttr_r7tb |" "$SHEET" | head -n 1)
    [ -n "$line" ] || { say "re-queue f$k: no sheet line in $SHEET"; continue; }
    arg=$(printf '%s' "$line" | awk -F'|' -v g=" $GATE " 'BEGIN{OFS="|"} {$3=g; print}')
    if [ "$MODE" = --eval ]; then say "WOULD re-queue f$k as the next job: ${arg:0:120}..."; continue; fi
    if $QCTL top "$arg" >> "$LOG" 2>&1; then say "RE-QUEUED f$k as the next job (qctl top; no out dir, no live line, verdict PASS)"
    else say "re-queue f$k: qctl refused (see above); retry next tick"; fi
  done
}
tick(){
  state
  local cur=""; [ -f "$V" ] && cur=$(head -c 4 "$V")
  if [ "$MODE" = --eval ]; then say "EVAL: would write $ST_VERDICT ($ST_WHY); current file: ${cur:-none}"
  elif [ "$cur" != "$ST_VERDICT" ]; then
    printf '%s %s | %s\n' "$ST_VERDICT" "$ST_WHY" "$(date '+%F %T %Z') tb_fallback.sh" > "$V.tmp" && mv -f "$V.tmp" "$V"
    say "verdict ${cur:-none} -> $ST_VERDICT: $ST_WHY"
  fi
  [ "$ST_VERDICT" = PASS ] && requeue
}
case $MODE in
  --eval|--once) tick; exit 0;;
  --loop)
    if [ -s $D/tb_fallback.pid ] && [ "$(cat $D/tb_fallback.pid)" != $$ ] && alive_pid $D/tb_fallback.pid tb_fallback.sh; then echo "already running"; exit 0; fi
    echo $$ > $D/tb_fallback.pid; say "== loop start pid $$ | verdict $V | queue $QUEUE"
    while :; do
      tick
      all=1; for k in 0 1 2 3; do [ -e "$FR/f${k}_s1047_ttr_r7tb" ] || all=0; done
      if [ $all = 1 ] && [ "$ST_VERDICT" = PASS ]; then say "f0-f3 all have out dirs and no rental is alive: loop exits"; break; fi
      [ $(date +%s) -ge $FREEZE ] && { say "training freeze: loop exits"; break; }
      sleep ${TBF_POLL_S:-120}
    done
    rm -f $D/tb_fallback.pid;;
  *) sed -n '2,8p' "$0"; exit 2;;
esac
