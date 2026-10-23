#!/bin/bash
# raise_budget.sh: once L5 is gone, lift this run's budget from min(cap, credit - floor - L5 hold-back) to
# min(cap, launch credit - L5's billing since our launch - floor) (coordinator 10-09 15:20: "once L5 is destroyed, the
# full remaining credit minus $0.15 is available, still with the $2.30 cap"), and move every time cap with it:
#   state.env BUDGET / DEADLINE_EPOCH / GUARD_DEADLINE, the on-instance guard (restarted with the new deadline),
#   plan.env DEADLINE_EPOCH (lanes restarted so they read it; running trainers are NOT touched, the lanes wait for them).
# Called by vast_watch.sh (once, BUDGET_RAISED=1). Never touches L5. Laptop side.
SCRIPT_TAG=raise; source "$(dirname "$0")/config.sh"
[ "$(st_get BUDGET_RAISED)" = 1 ] && exit 0
l5_alive && { log "L5 still alive: no raise"; exit 0; }
IID=$(st_get INSTANCE_ID); [ -n "$IID" ] && [ "$(st_get DESTROYED)" != 1 ] || exit 0
L0=$(st_get LAUNCH_EPOCH); DPH=$(st_get DPH); XF=$(st_get XFER); C0=$(st_get CREDIT0); B=$(st_get BUDGET); CR=$(st_get CAP_RUN); CR=${CR:-$CAP_USD}
L5S=$(l5_spend_since "$L0")
NB=$("$PYL" -c "print(round(min($CR, $C0 - $L5S - $FLOOR_USD), 2))")
if ! num_ge "$NB" "$("$PYL" -c "print($B + 0.05)")"; then log "budget \$$B stays (lifted value \$$NB is not higher)"; st_set BUDGET_RAISED 1; exit 0; fi
GH=$("$PYL" -c "print(min(($NB - $RESERVE_USD - $XF) / $DPH, $MAX_RUN_H + $GUARD_MARGIN_H))")
GDL=$("$PYL" -c "print(int($L0 + $GH * 3600))"); DL=$("$PYL" -c "print(int($L0 + ($GH - $GUARD_MARGIN_H) * 3600))")
log "L5 gone: budget \$$B -> \$$NB (cap \$$CR, launch credit \$$C0 - L5 since launch \$$L5S - floor \$$FLOOR_USD) | watcher cap $(date -d @$(st_get DEADLINE_EPOCH) '+%H:%M') -> $(date -d @$DL '+%H:%M') | guard -> $(date -d @$GDL '+%H:%M')"
# 1 on-instance guard: new deadline file, restart the guard process (it reads the file at start)
G=$(ix "echo $GDL > $R/status/guard_deadline; p=\$(cat $R/status/guard.pid 2>/dev/null); [ -n \"\$p\" ] && kill \$p 2>/dev/null; sleep 1; rm -f $R/status/guard.pid; (setsid nohup bash $R/guard.sh >> $R/guard.out 2>&1 < /dev/null &); sleep 3; tail -n 2 $R/guard.log" 2>&1 | tr '\n' ' ')
case $G in *"$(date -u -d @$GDL '+%F %T')"*) log "guard restarted: $G";; *) log "WARN: guard restart not confirmed ($G): budget NOT raised"; exit 1;; esac
st_set GUARD_DEADLINE "$GDL"; st_set DEADLINE_EPOCH "$DL"; st_set BUDGET "$NB"; st_set BUDGET_RAISED 1
# 2 lanes: new lane deadline; restart lane.sh (only if the kit is on the instance); trainers keep running
LDL=$(( DL - LANE_MARGIN_S ))
if ix "test -f $R/kit/plan.env"; then
  ix "sed -i 's/^DEADLINE_EPOCH=.*/DEADLINE_EPOCH=$LDL/' $R/kit/plan.env && grep DEADLINE $R/kit/plan.env" | sed 's/^/  inst /'
  if [ -n "$(st_get TRAIN_STARTED)" ]; then
    ix "for p in $R/status/lane_*.pid; do [ -e \$p ] && kill \$(cat \$p) 2>/dev/null; rm -f \$p; done; sleep 2; pgrep -fa 'fleet_train[.]py' | cut -c1-80" | sed 's/^/  inst trainers still alive: /'
    bash "$KIT/vast_train.sh" > "$LOGDIR/raise_train.out" 2>&1 && log "lanes restarted with lane deadline $(date -d @$LDL '+%H:%M')" || log "WARN: lane restart failed (see raise_train.out); the watcher restarts dead lanes once"
  fi
fi
log "budget raise done"
