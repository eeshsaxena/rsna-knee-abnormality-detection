#!/bin/bash
# status.sh: BOX side, read only. Per fold: last epoch line (raw/ema val AUC, epoch minutes) and the latest s/it
# from the log on the Modal volume; the app state; the workspace month-to-date cost.
M=~/modalenv/bin/modal; T=~/modal_e2/logs/status_tmp; mkdir -p $T
echo "== $(date '+%m-%d %H:%M:%S %Z')"
$M app list 2>/dev/null | grep -E "rsna-e2" | sed 's/  */ /g' | cut -c1-160
for F in ${E2_FOLDS:-0 1 2 3 4}; do
  N=f${F}_s1046_ttr288frac75
  $M volume get rsna-e2 /runs/$N.log - 2>/dev/null | grep -v "Finished downloading" > $T/$N.log
  if [ -s $T/$N.log ]; then
    v=$(grep -E "^ep [0-9]+ (raw|ema) .*val AUC" $T/$N.log | tail -n 2 | sed -E 's/^ep ([0-9]+) (raw|ema).*val AUC vs [^ ]+ ([0-9.]+).*epoch ([0-9.]+) min/ep\1 \2 \3 \4m/' | tr '\n' ' ')
    s=$(grep -E "^ep [0-9]+ it " $T/$N.log | tail -n 1 | cut -c1-60)
    c=$(tail -n 1 $T/$N.log | grep -E '^\{"raw"' || true)
    echo "f$F: ${c:+DONE $c | }$v| last: $s"
  else
    echo "f$F: no log on the volume yet"
  fi
done
$M billing summary 2>/dev/null | grep -E "Cost" | sed 's/  */ /g'
