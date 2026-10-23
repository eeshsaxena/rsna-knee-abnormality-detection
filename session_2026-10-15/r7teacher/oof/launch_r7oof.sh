#!/bin/bash
# launch_r7oof.sh (r7teacher/oof, 10-14, PREREG_TEACHER A1 prep; laptop Git Bash). NOT RUN BY ITS AUTHOR. Run it ONLY after Eesh
# replied "OK R7" (his Kaggle GPU quota: 2 sessions x ~3.1-3.9 h on T4 x2, 6.5-9 GPU-h incl. boot, MODEL). Pushes the two
# commit-tested full-run kernels (cells 0, 2, 3, 4 byte-identical to the pushed commit v1 / proof v2 kernels; only the config
# cell 1 differs: mode full, shard i of 2, budget 8.5 h, chunk 600). Nothing is submitted; no dataset is created.
#   bash launch_r7oof.sh parallel   (default) both shards now on two slugs: rsna-r7-oof (shard 0) + rsna-r7-oof-b (shard 1)
#                                   needs RUNNING_COUNT 0 (uses both GPU slots for ~4 h; no other commit possible meanwhile)
#   bash launch_r7oof.sh seq0       shard 0 only (keeps one slot free for a canary-fix commit); needs RUNNING_COUNT <= 1
#   bash launch_r7oof.sh seq1       shard 1 on slug rsna-r7-oof-b (after shard 0's output is DOWNLOADED; disjoint shards,
#                                   so no --skip-from is needed unless a session spilled over)
# Quota refusal symptom (RUNNING then ERROR within ~2-3 min with an EMPTY log, lesson 10-08: account has < ~12 h free):
#   re-push the same folder with a timeout above the 8.5 h internal deadline:  kaggle kernels push -p <folder> -t 32400
#   if that is refused too, rebuild with a shorter budget and a matching timeout (the internal guard must end first):
#   python build_r7oof.py --mode full --shard-index I --shard-count 2 --budget-h 5.5 [--slug rsna-r7-oof-b] ; push -t 21600
#   (a cut session is resumed later with --skip-from <earlier r7_oof.csv>; oof/RESULT.md sec 5).
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-parallel}
V=../../v224f
S0=kernel_r7oof_full_s0of2; S1=kernel_r7oof_full_s1of2_parallel
want0=358beef2217a4f1a; want1=202d67c776a4a13c                     # notebook sha256[:16] as built 10-14 13:43-13:45
nbsha(){ python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])" "$1/rsna_r7_oof.ipynb"; }
[ "$(nbsha $S0)" = $want0 ] || { echo "STOP: $S0 notebook changed since the A1 prep"; exit 1; }
[ "$(nbsha $S1)" = $want1 ] || { echo "STOP: $S1 notebook changed since the A1 prep"; exit 1; }
grep -q '"id": "eeshsaxena/rsna-r7-oof"' $S0/kernel-metadata.json && grep -q '"id": "eeshsaxena/rsna-r7-oof-b"' $S1/kernel-metadata.json \
  || { echo "STOP: slugs"; exit 1; }
RC=$(python $V/running.py 8 | tee -a launch_r7oof.log | tail -n 1)
N=$(echo "$RC" | awk '{print $2}')
echo "$(date '+%F %T') mode $MODE | $RC" | tee -a launch_r7oof.log
case $MODE in
  parallel) [ "$N" = 0 ] || { echo "STOP: $N kernel(s) running; parallel needs 0 (use seq0)"; exit 1; }
            kaggle kernels push -p $S0 2>&1 | tee push_full_s0.log
            kaggle kernels push -p $S1 2>&1 | tee push_full_s1.log;;
  seq0)     [ "$N" -le 1 ] || { echo "STOP: $N kernels running"; exit 1; }
            kaggle kernels push -p $S0 2>&1 | tee push_full_s0.log;;
  seq1)     [ -s out_full_s0/r7_oof.csv ] || { echo "STOP: download shard 0 first (kaggle kernels output keeps only the latest version)"; exit 1; }
            [ "$N" -le 1 ] || { echo "STOP: $N kernels running"; exit 1; }
            kaggle kernels push -p $S1 2>&1 | tee push_full_s1.log;;
  *) echo "usage: launch_r7oof.sh parallel|seq0|seq1"; exit 2;;
esac
cat <<EOF
PUSHED ($MODE). Poll + fetch (read-only; run as background tasks of the main session, they die with it):
  python $V/wait_fetch.py eeshsaxena/rsna-r7-oof   out_full_s0 600
  python $V/wait_fetch.py eeshsaxena/rsna-r7-oof-b out_full_s1 600      (parallel / seq1)
Within ~3 min check both are RUNNING with a growing log (quota-refusal symptom above). Expected end: ~3.1-3.9 h + boot (MODEL).
Then: bash after_r7oof.sh out_full_s0 out_full_s1
EOF
