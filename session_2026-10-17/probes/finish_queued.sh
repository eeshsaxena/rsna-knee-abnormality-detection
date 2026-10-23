#!/bin/bash
# finish_queued.sh (10-17, Claude): the two A16 probes whose commits were still QUEUED at the 50-min wait: wait up to 240 min each, fetch, gate,
# submit on PROBE_CHECK PASS 16/16 only (same idempotent submit as run_probes.sh). No new push.
set -u
cd "$(dirname "$0")/../.." || exit 1
D=session_2026-10-17/probes
for spec in "s615 v224r6hws6f15j 0.5 0.35 0.15" "a25 v224r6hwa25j 0.5 0.25 0.2"; do
  set -- $spec; P=$1; TAG=$2; WR=$3; A2=$4; WS=$5; OUT=$D/out_${TAG}_v1
  rm -rf $OUT
  python session_2026-10-15/v224f/wait_fetch.py eeshsaxena/rsna-sub-$TAG $OUT 240 > $OUT.log 2>&1
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
echo FINISH_QUEUED_DONE
