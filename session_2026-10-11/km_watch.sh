#!/bin/bash
# km_watch.sh: polls kmrsuman's two D2 notebooks every 5 min; makes the EMPTY box placeholders as soon as the box is
# reachable; exits when the box placeholders are done AND (both notebooks finished OR a status changes to error).
L=/c/Users/eeshs/Downloads/rsna-knee/session_2026-10-11/km_watch.log
ph=0; prev=""
for i in $(seq 1 160); do
  st=""
  for s in f12 f34; do st="$st $s:$(kaggle kernels status kmrsuman/rsna-train-ttr-frac75-s1044-$s 2>&1 | tail -1 | grep -oE 'RUNNING|COMPLETE|ERROR|CANCEL[A-Z_]*|QUEUED' | head -1)"; done
  [ "$st" != "$prev" ] && echo "[$(date '+%m-%d %H:%M')] status$st" | tee -a $L; prev="$st"
  if [ $ph = 0 ] && ssh -o ConnectTimeout=8 -o BatchMode=yes training@192.168.10.30 true 2>/dev/null; then
    r=$(ssh -o BatchMode=yes training@192.168.10.30 'for F in 1 2 3 4; do d=~/fleet_runs/f${F}_s1044_ttr_frac75; if [ -e $d ] || [ -e $d.log ]; then echo "EXISTS f$F"; else mkdir $d && echo "placeholder f$F"; fi; done' < /dev/null 2>&1 | tr '\n' ' ')
    echo "[$(date '+%m-%d %H:%M')] box: $r" | tee -a $L; ph=1
  fi
  echo "$st" | grep -q ERROR && { echo "ERROR seen" | tee -a $L; exit 2; }
  [ $ph = 1 ] && echo "$st" | grep -q "f12:COMPLETE" && echo "$st" | grep -q "f34:COMPLETE" && { echo "both COMPLETE" | tee -a $L; exit 0; }
  [ $ph = 1 ] && [ $i -ge 12 ] && echo "$st" | grep -q "f12:RUNNING" && echo "$st" | grep -q "f34:RUNNING" && [ ! -f /tmp/km_early_done ] && { touch /tmp/km_early_done; echo "past 1 h, both RUNNING, placeholders made" | tee -a $L; exit 0; }
  sleep 300
done
