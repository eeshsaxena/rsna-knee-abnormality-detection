#!/bin/bash
# watch_r7oof.sh SLUG:OUTDIR [SLUG:OUTDIR ...] (r7teacher/oof, 10-14; laptop Git Bash, read-only on Kaggle). Polls every kernel's
# status every 120 s, tolerant of network / API errors (an error just means "poll again"), appends every status change to
# watch_r7oof.log, and downloads each kernel's output into its OUTDIR as soon as it is terminal (COMPLETE / ERROR / CANCEL*;
# kaggle kernels output keeps only the LATEST version, so this runs before any re-push of the same slug). Exits when every
# kernel is terminal and fetched (or after MAX_H hours, default 10). Prints one final summary line per kernel.
cd "$(dirname "$0")"
MAX_H=${MAX_H:-10}; T0=$(date +%s)
declare -A last done
while :; do
  all=1
  for kv in "$@"; do
    k=${kv%%:*}; o=${kv#*:}
    [ "${done[$k]:-}" = 1 ] && continue
    all=0
    s=$(timeout 60 kaggle kernels status "eeshsaxena/$k" 2>&1 | tail -n 1 | tr -d '\r')
    case "$s" in
      *COMPLETE*|*ERROR*|*CANCEL*)
        echo "$(date '+%F %T') $k TERMINAL $s" >> watch_r7oof.log
        mkdir -p "$o"
        for i in 1 2 3 4 5; do
          timeout 1800 kaggle kernels output "eeshsaxena/$k" -p "$o" >> watch_r7oof.log 2>&1 && [ -s "$o/r7_oof.csv" -o -n "$(ls $o 2>/dev/null)" ] && break
          echo "$(date '+%F %T') $k fetch attempt $i failed, retry" >> watch_r7oof.log; sleep 60
        done
        echo "$(date '+%F %T') $k FETCHED -> $o ($(ls $o | wc -l) files; r7_oof.csv $( [ -s $o/r7_oof.csv ] && wc -l < $o/r7_oof.csv || echo missing) lines)" >> watch_r7oof.log
        done[$k]=1;;
      *RUNNING*|*QUEUED*)
        [ "${last[$k]:-}" != "$s" ] && echo "$(date '+%F %T') $k $s" >> watch_r7oof.log; last[$k]=$s;;
      *)
        echo "$(date '+%F %T') $k poll error/unknown: ${s:0:160}" >> watch_r7oof.log;;
    esac
  done
  [ $all = 1 ] && break
  [ $(( $(date +%s) - T0 )) -gt $(( MAX_H * 3600 )) ] && { echo "$(date '+%F %T') WATCH TIMEOUT ${MAX_H} h" >> watch_r7oof.log; break; }
  sleep 120
done
for kv in "$@"; do k=${kv%%:*}; grep " $k " watch_r7oof.log | tail -n 2; done
echo "WATCH_R7OOF_DONE"
