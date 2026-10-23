#!/bin/bash
# status.sh [touch_seen] (rental_kc instance, read only except touch_seen). No args: one line per planned member
#   MEMBER <key> state=<pending|running|done|failed|stopped|skipped>[+fetched] gpu=<g> ep=<n>/13 it=<it>/<iters> sit=<last s/it> trmin=<last TRAIN-ONLY>
# then SMOKE lines (if any), GPU lines, LANES_ALIVE=<n>, ALL_TERMINAL=0|1. done = closing line + <out>.sha256 + status/<key>.done.
R=${RSNA_ROOT:-/workspace/rsna}; S=$R/status; FR=$R/fleet_runs; K=$R/kit; . $K/kit.env; . $K/plan.env 2>/dev/null
closing(){ tail -n 1 "$1" 2>/dev/null | tr -d '\r' | grep -Eq '^\{"raw": [^,]+, "ema": [^}]+\}$'; }
[ "${1:-}" = touch_seen ] && { touch $S/box_seen; exit 0; }
line(){ local L=$1 ep it sit tm
  ep=$(grep -c -E '^ep [0-9]+ raw ' $L 2>/dev/null); ep=${ep:-0}
  it=$(grep -oE '^ep [0-9]+ it [0-9]+/[0-9]+' $L 2>/dev/null | tail -n 1 | awk '{print $4}')
  sit=$(grep -oE '^ep [0-9]+ it [0-9]+/[0-9]+ loss [^ ]+ [0-9.]+s/it' $L 2>/dev/null | tail -n 1 | awk '{print $NF}')
  tm=$(grep -E 'TRAIN-ONLY' $L 2>/dev/null | tail -n 1 | sed -E 's/.*TRAIN-ONLY ([0-9.]+) min.*/\1/')
  echo "ep=$ep/${EPOCHS:-13} it=${it:--} sit=${sit:--} trmin=${tm:--}"; }
echo "STATUS $(date '+%F %T %Z') | members [${MEMBERS:-}]"
nt=0; n=0
for k in ${MEMBERS:-}; do
  n=$((n + 1)); N=$(awk -F'\t' -v k=$k '$1==k {print $2}' $K/members.tsv); L=$FR/$N.log; st=pending; g=-
  [ -e $S/$k.running ] && { st=running; g=$(awk '{print $2}' $S/$k.running); }
  [ -e $S/$k.failed ] && st=failed; [ -e $S/$k.stopped ] && st=stopped; [ -e $S/$k.skipped ] && st=skipped
  [ -e $S/$k.done ] && closing $L && [ -s $FR/$N.sha256 ] && st=done
  case $st in done|failed|stopped|skipped) nt=$((nt + 1));; esac
  [ -e $S/$k.fetched_ok ] && st="$st+fetched"
  echo "MEMBER $k state=$st gpu=$g $(line $L)"
done
for f in $S/smoke_*.running $S/smoke_*.ended; do [ -e "$f" ] || continue; k=$(basename $f); k=${k#smoke_}; k=${k%.*}
  N=$(awk -F'\t' -v k=$k '$1==k {print $2}' $K/members.tsv); echo "SMOKE $k $(basename $f | sed 's/.*[.]//') $(line $R/smoke/$N.log)"; done
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU /'
echo "CPU load $(cut -d' ' -f1-3 /proc/loadavg) of $(nproc) | disk free $(df -h $R | tail -n 1 | awk '{print $4}')"
alive=0; for p in $S/lane_*.pid; do [ -e "$p" ] && kill -0 "$(cat "$p")" 2>/dev/null && alive=$((alive + 1)); done
echo "LANES_ALIVE=$alive"
if [ "$n" -gt 0 ] && [ "$nt" -eq "$n" ] && [ "$alive" -eq 0 ]; then echo ALL_TERMINAL=1; else echo ALL_TERMINAL=0; fi
