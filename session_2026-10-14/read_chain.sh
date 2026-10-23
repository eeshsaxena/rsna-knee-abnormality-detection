#!/bin/bash
# read_chain.sh REF...: as each pending submission scores, snapshot the LB and run the frozen-team read vs the previous snapshot.
SP="C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/01efcc80-088f-477c-8a8a-b504ddf28d6a/scratchpad"
FR=/c/Users/eeshs/Downloads/rsna-knee/session_2026-10-13/frozen_read.py
LOG=/c/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/read_chain.log
prev=${PREV:-$SP/lb14b}; todo="$*"; k=0
while [ -n "$todo" ]; do
  for ref in $todo; do
    s=$(kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null | python -c "
import sys, csv
rows = {r['ref']: r for r in csv.DictReader(sys.stdin)}
r = rows.get('$ref'); print((r['status'].replace('SubmissionStatus.','') + ' ' + r['publicScore'] + ' ' + r['description'][:30]) if r else '')")
    if echo "$s" | grep -qE "COMPLETE|ERROR"; then
      k=$((k+1)); sleep 120; d=$SP/lb14r$k; rm -rf $d; mkdir -p $d; (cd $d && kaggle competitions leaderboard rsna-knee-abnormality-detection -d -p . >/dev/null 2>&1; unzip -o -q *.zip)
      echo "[$(date '+%m-%d %H:%M')] $ref $s" | tee -a $LOG
      python $FR $prev $d 2>&1 | tee -a $LOG; prev=$d
      todo=$(echo $todo | tr ' ' '\n' | grep -v "^$ref$" | tr '\n' ' ' | sed 's/ *$//')
    fi
  done
  [ -n "$todo" ] && sleep 240
done
echo "ALL READ" | tee -a $LOG
