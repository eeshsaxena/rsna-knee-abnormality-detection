#!/bin/bash
# status.sh [manifest F]: instance side, read only.
#   (no args)    one screen of progress for the planned folds + GPUs; last line ALL_DONE=0|1
#   manifest F   md5 list (paths relative to fleet_runs) of what vast_fetch.sh copies for fold F:
#                top-level *_best.pt + oof_*.csv, epochs/oof_*.csv (+ epochs/*.pt if EPOCHPT=1), and the .log
R=${RSNA_ROOT:-/workspace/rsna}; . $R/kit/kit.env; . $R/kit/plan.env 2>/dev/null; S=$R/status
closing(){ tail -n 1 "$1" 2>/dev/null | tr -d '\r' | grep -Eq '^\{"raw": [^,]+, "ema": [^}]+\}$'; }
if [ "${1:-}" = manifest ]; then
  N=f${2}_s${SEED}_${RUNTAG}; cd $R/fleet_runs || exit 1
  { find $N -maxdepth 1 -type f \( -name '*_best.pt' -o -name 'oof_*.csv' \)
    find $N/epochs -maxdepth 1 -type f -name 'oof_*.csv' 2>/dev/null
    [ "${EPOCHPT:-0}" = 1 ] && find $N/epochs -maxdepth 1 -type f -name '*.pt' 2>/dev/null
    [ -f $N.log ] && echo $N.log; } | sort | xargs -r md5sum
  exit 0
fi
echo "STATUS $(date '+%F %T %Z') | folds [${FOLDS:-?}] | deadline $(date -d @${DEADLINE_EPOCH:-0} '+%H:%M')"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null | sed 's/^/GPU /'
echo "CPU load $(cut -d' ' -f1-3 /proc/loadavg) of $(nproc) vCPU | disk free $(df -h $R | tail -n 1 | awk '{print $4}')"
alive=0; for p in $S/lane_*.pid; do [ -e "$p" ] && kill -0 "$(cat "$p")" 2>/dev/null && alive=$((alive + 1)); done
nd=0; n=0
for F in ${FOLDS:-}; do
  n=$((n + 1)); N=f${F}_s${SEED}_${RUNTAG}; L=$R/fleet_runs/$N.log; st=pending
  [ -e $S/f$F.running ] && st=running; [ -e $S/f$F.skipped ] && st=skipped; [ -e $S/f$F.failed ] && st=failed
  closing $L && st=done
  [ -e $S/f$F.fetched_ok ] && st="$st+fetched"
  ep=$(grep -c '^ep [0-9]* raw |' $L 2>/dev/null); ep=${ep:-0}
  it=$(grep -E '^ep [0-9]+ it ' $L 2>/dev/null | tail -n 1 | cut -c1-60)
  ev=$(grep -E '^ep [0-9]+ raw \|' $L 2>/dev/null | tail -n 1 | sed -E 's/.*val AUC vs [^ ]+ ([0-9.]+).*/\1/')
  tm=$(grep -E 'TRAIN-ONLY' $L 2>/dev/null | tail -n 1 | sed -E 's/.*TRAIN-ONLY ([0-9.]+) min.*/\1/')
  echo "FOLD f$F state=$st epochs=$ep/21 last_raw_val=${ev:--} train_min/ep=${tm:--} last=\"${it:--}\""
  case $st in done*|failed*|skipped*) nd=$((nd + 1));; esac
done
echo "LANES_ALIVE=$alive"
if [ "$n" -gt 0 ] && [ "$nd" -eq "$n" ] && [ "$alive" -eq 0 ]; then echo ALL_DONE=1; else echo ALL_DONE=0; fi
