#!/bin/bash
# run_probes.sh (10-17, Claude): for each A16 probe: commit on Kaggle, gate, and SUBMIT only on PROBE_CHECK PASS 16/16 (idempotent).
set -u
cd "$(dirname "$0")/../.." || exit 1
D=session_2026-10-17/probes
for spec in "s615 v224r6hws6f15j 0.5 0.35 0.15" "wr45 v224r6hwwr45j 0.45 0.35 0.2" "a25 v224r6hwa25j 0.5 0.25 0.2"; do
  set -- $spec; P=$1; TAG=$2; WR=$3; A2=$4; WS=$5; OUT=$D/out_${TAG}_v1
  python session_2026-10-15/v224f/running.py 6 > $D/running_$TAG.txt
  N=$(grep RUNNING_COUNT $D/running_$TAG.txt | awk '{print $2}')
  while [ "${N:-9}" -gt 1 ]; do sleep 60; python session_2026-10-15/v224f/running.py 6 > $D/running_$TAG.txt; N=$(grep RUNNING_COUNT $D/running_$TAG.txt | awk '{print $2}'); done
  kaggle kernels push -p $D/kernel_$TAG > $D/push_$TAG.log 2>&1
  grep -q "successfully pushed" $D/push_$TAG.log || { echo "$P PUSH FAILED"; continue; }
  python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$TAG $OUT 50 > $OUT.log 2>&1
  py -3.12 -I $D/probe_check.py $OUT --wr $WR --a224 $A2 --ws6 $WS > $D/gate_${TAG}_v1.txt 2>&1
  LAST=$(tail -1 $D/gate_${TAG}_v1.txt | tr -d '\r'); echo "$P gate: $LAST"
  [ "$LAST" = "PROBE_CHECK PASS 16/16" ] || { echo "$P NOT submitted (gate)"; continue; }
  V=$(grep -o "Kernel version [0-9]*" $D/push_$TAG.log | awk '{print $3}')
  python - "$P" "$TAG" "$V" <<'PYEOF'
import sys, datetime
from kaggle.api.kaggle_api_extended import KaggleApi
p, tag, v = sys.argv[1], sys.argv[2], int(sys.argv[3])
pre = 'PROBE-A16-' + p.upper()
a = KaggleApi(); a.authenticate()
subs = a.competition_submissions('rsna-knee-abnormality-detection')
if any(str(getattr(s, 'description', '') or '').startswith(pre) for s in subs): print(p, 'already submitted'); sys.exit(0)
day = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
if sum(1 for s in subs if str(getattr(s, 'date', '')).startswith(day)) >= 5: print(p, '5 submissions today: skip'); sys.exit(0)
r = a.competition_submit_code(file_name='submission.csv', competition='rsna-knee-abnormality-detection',
    message='%s: HW best (57262115) one-line probe %s; PROBE_CHECK 16/16; PREREG_10-15 A16' % (pre, p), kernel='eeshsaxena/rsna-sub-' + tag, kernel_version=v)
line = '[%s IST] A16 %s SUBMITTED (v%d): %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), p, v, r)
print(line); open(r'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-14\submits.log', 'a', encoding='utf-8').write(line + '\n')
PYEOF
done
echo RUN_PROBES_DONE
