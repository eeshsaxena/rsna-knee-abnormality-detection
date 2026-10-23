#!/bin/bash
# status.sh [manifest F | touch_seen] (instance, read only except touch_seen). No args: one line per planned fold
#   FOLD f<F> state=<pending|running|done|failed|skipped>[+fetched] gpu=<g> ep=<n>/<E> sit=<last s/it> val=<last raw val> trmin=<last TRAIN-ONLY>
# then GPU lines, LANES_ALIVE=<n>, ALL_TERMINAL=0|1. "done" = closing line + <N>.sha256 + status/f<F>.done.
# manifest F: the fold's <N>.sha256 (what the box verifies). touch_seen: touch status/box_seen (the guard's orphan fuse).
R=${RSNA_ROOT:-/workspace/rsna}; S=$R/status; FR=$R/fleet_runs; . $R/kit/kit.env; . $R/kit/plan.env 2>/dev/null
closing(){ tail -n 1 "$1" 2>/dev/null | tr -d '\r' | grep -Eq '^\{"raw": [^,]+, "ema": [^}]+\}$'; }
case ${1:-} in
  manifest) cat $FR/f${2}_s${SEED}_${RUNTAG}.sha256; exit $?;;
  touch_seen) touch $S/box_seen; exit 0;;
esac
echo "STATUS $(date '+%F %T %Z') | folds [${FOLDS:-?}] | lane deadline $(date -d @${DEADLINE_EPOCH:-0} '+%m-%d %H:%M')"
nt=0; n=0
for F in ${FOLDS:-}; do
  n=$((n + 1)); N=f${F}_s${SEED}_${RUNTAG}; L=$FR/$N.log; st=pending; g=-
  [ -e $S/f$F.running ] && { st=running; g=$(awk '{print $2}' $S/f$F.running); }
  [ -e $S/f$F.skipped ] && st=skipped; [ -e $S/f$F.failed ] && st=failed
  [ -e $S/f$F.done ] && closing $L && [ -s $FR/$N.sha256 ] && st=done
  case $st in done|failed|skipped) nt=$((nt + 1));; esac
  [ -e $S/f$F.fetched_ok ] && st="$st+fetched"
  ep=$(grep -c -E '^ep [0-9]+ raw \|' $L 2>/dev/null); ep=${ep:-0}
  sit=$(grep -oE '^ep [0-9]+ it [0-9]+/[0-9]+ loss [^ ]+ [0-9.]+s/it' $L 2>/dev/null | tail -n 1 | awk '{print $NF}')
  val=$(grep -E '^ep [0-9]+ raw \|' $L 2>/dev/null | tail -n 1 | sed -E 's/.*val AUC vs [^ ]+ ([0-9.]+).*/\1/')
  tm=$(grep -E 'TRAIN-ONLY' $L 2>/dev/null | tail -n 1 | sed -E 's/.*TRAIN-ONLY ([0-9.]+) min.*/\1/')
  echo "FOLD f$F state=$st gpu=$g ep=$ep/${EPOCHS:-21} sit=${sit:--} val=${val:--} trmin=${tm:--}"
done
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU /'
echo "CPU load $(cut -d' ' -f1-3 /proc/loadavg) of $(nproc) | disk free $(df -h $R | tail -n 1 | awk '{print $4}')"
alive=0; for p in $S/lane_*.pid; do [ -e "$p" ] && kill -0 "$(cat "$p")" 2>/dev/null && alive=$((alive + 1)); done
echo "LANES_ALIVE=$alive"
if [ "$n" -gt 0 ] && [ "$nt" -eq "$n" ] && [ "$alive" -eq 0 ]; then echo ALL_TERMINAL=1; else echo ALL_TERMINAL=0; fi
