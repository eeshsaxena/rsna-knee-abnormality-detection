#!/bin/bash
# api_chain.sh REF...: when each pending submission scores, take an API LB snapshot and print our team-best id / rank (api_frozen.py vs previous).
L=/c/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe; LOG=/c/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/api_chain.log
cd $L; prev=${PREV:-api_lb_1014T1255.json}; todo="$*"; k=0
while [ -n "$todo" ]; do
  for ref in $todo; do
    s=$(kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null | python -c "
import sys, csv
rows = {r['ref']: r for r in csv.DictReader(sys.stdin)}
r = rows.get('$ref'); print((r['status'].replace('SubmissionStatus.','') + ' ' + r['publicScore'] + ' ' + r['description'][:30]) if r else '')")
    if echo "$s" | grep -qE "COMPLETE|ERROR"; then
      k=$((k+1)); sleep 90; new=api_lb_chain_$(date +%H%M)_$k.json; python fetch_api_lb.py $new > /dev/null 2>&1
      echo "[$(date '+%m-%d %H:%M')] $ref $s" | tee -a $LOG; python api_frozen.py $prev $new 2>&1 | tee -a $LOG; prev=$new
      todo=$(echo $todo | tr ' ' '\n' | grep -v "^$ref$" | tr '\n' ' ' | sed 's/ *$//')
    fi
  done
  [ -n "$todo" ] && sleep 180
done
echo "ALL READ" | tee -a $LOG
