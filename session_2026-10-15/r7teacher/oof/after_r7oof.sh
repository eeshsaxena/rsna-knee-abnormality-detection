#!/bin/bash
# after_r7oof.sh OUT_DIR [OUT_DIR ...] (r7teacher/oof, 10-14, PREREG_TEACHER A1 prep; laptop Git Bash). NOT RUN BY ITS AUTHOR.
# Run when the R7-OOF sessions are downloaded (launch_r7oof.sh). Verifies each session, merges them, checks what the A1 builder
# needs (every row: finite z|<label> x 12, precision full), copies the merged file to the box and runs on_r7oof.sh there (box CPU:
# L0 per-fold leak gate, REAL label build, LABEL_GATE, labels_gate.sh verdict). Installs NOTHING in the box queue: on PASS it
# prints the T_b install command (labels/kits/box/install_r7tb_top.sh) for the main session.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: after_r7oof.sh out_full_s0 out_full_s1 [more resume sessions]"; exit 2; }
B=training@192.168.10.30
for d in "$@"; do
  python verify_commit.py "$d" > "verify_$(basename $d).log" 2>&1 || true          # it prints ALL_PASS / SOME_FAIL last
  tail -n 1 "verify_$(basename $d).log" | grep -q '^ALL_PASS' && echo "verify $d: ALL_PASS" \
    || { echo "STOP: verify_commit.py $d did not print ALL_PASS (see verify_$(basename $d).log)"; exit 1; }
done
M=${OUTCSV:-r7_oof_ALL.csv}
python merge_sessions.py $M "$@" | tee ${M%.csv}_merge.log
OUTCSV=$M python - <<'EOF'
import pandas as pd, numpy as np
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
     'Contusion', 'Fracture']
import os
d = pd.read_csv(os.environ.get('OUTCSV', 'r7_oof_ALL.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
z = d[['z|' + c for c in T]].astype(float).values
bad = (~np.isfinite(z)).any(1) | (d['precision'].astype(str) != 'full').values
pool_missing = 4407 - len(d)
print(f'A1 pre-check: rows {len(d)} | rows without finite z| or not full precision {int(bad.sum())} | train studies missing {pool_missing}')
assert bad.sum() == 0, 'rows without full-precision z|: re-run them (build_r7oof.py --skip-from ...) before the label build'
print('A1 PRE-CHECK PASS (the builder will still enforce the 1 % missing cap on the 4,333 non-gold pool rows)')
EOF
md5sum $M | tee ${M%.csv}.md5
[ -n "${DRY:-}" ] && { echo "DRY: stop before the box steps"; exit 0; }
ssh -o BatchMode=yes $B 'mkdir -p ~/r7t_labels/incoming' < /dev/null
scp -q r7_oof_ALL.csv $B:r7t_labels/incoming/r7_oof_ALL.csv
ssh -o BatchMode=yes $B "cd ~/r7t_labels/incoming && md5sum r7_oof_ALL.csv" < /dev/null
ssh -o BatchMode=yes $B 'CUDA_VISIBLE_DEVICES= taskset -c 34-39 bash ~/r7t_labels/on_r7oof.sh ~/r7t_labels/incoming/r7_oof_ALL.csv' < /dev/null \
  | tee on_r7oof_REAL.log
echo "Next (main session, only on L0 PASS/DROP + LABEL_GATE + labels_gate PASS): bash ../labels/kits/box/install_r7tb_top.sh (it copies itself to the box and installs the 5 T_b lines ahead of L13)"
