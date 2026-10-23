#!/bin/bash
# on_r7oof.sh R7OOF_CSV (r7teacher/labels, 2026-10-14; BOX, CPU, seconds): run the moment job 1's merged R7-OOF file
# (merge_sessions.py output, copied to the box) exists. It does NOT touch the queue and rents nothing; it prints the
# commands the main session runs next.
#   1 copy the file to ~/r7t_labels/real/R7OOF.csv (refuses to overwrite a DIFFERENT file: one teacher per PREREG)
#   2 check PREREG_TEACHER.md against PREREG_TEACHER.sha256 (the builder checks it again)
#   3 REAL build (PREREG_TEACHER + Amendment A1): L0 per-fold leak gate (A1 FX3; STOP = exit 4, DROP = one fold's R7 rows
#     removed and printed), then labels_r7t_a.csv / labels_r7t_b.csv from the RAW z| logits (A1 FX1/FX2) + receipt + diag +
#     LABEL_GATE (PREREG sec 4)
#   4 labels_gate.sh for the table the label gate chose (writes the verdict the queue lines / the rental need)
set -u
SRC=${1:?usage: on_r7oof.sh /path/to/merged_r7_oof.csv}
D=$HOME/r7t_labels; PY=$HOME/miniconda3/envs/rsna/bin/python
mkdir -p $D/real
if [ -e $D/real/R7OOF.csv ]; then cmp -s "$SRC" $D/real/R7OOF.csv || { echo "REFUSE: a different $D/real/R7OOF.csv already exists"; exit 1; }
else cp "$SRC" $D/real/R7OOF.csv; fi
echo "R7OOF.csv sha256 $(sha256sum < $D/real/R7OOF.csv | cut -c1-16).. rows $(($(wc -l < $D/real/R7OOF.csv) - 1))"
[ "$(sha256sum < $D/PREREG_TEACHER.md | cut -c1-64)" = "$(cut -c1-64 $D/PREREG_TEACHER.sha256)" ] || { echo "STOP: PREREG_TEACHER.md does not match its sha256"; exit 1; }
[ -s $D/fleet/gold_fleetOF_mean.csv ] || { echo "STOP: fleet gold mean missing (re-run collect_fleet_oof.py after the D2 gold preds)"; exit 1; }
cd $D && nice -n 19 $PY build_teacher_labels.py --teacher real/R7OOF.csv --teacher_tag R7OOF \
  --fleet_oof fleet/oof_fleetOF_mean.csv --fleet_gold fleet/gold_fleetOF_mean.csv --out_dir $D/real \
  --prereg $D/PREREG_TEACHER.md --prereg_sha $D/PREREG_TEACHER.sha256 2>&1 | tee $D/real/build.log
RC=${PIPESTATUS[0]}
echo; echo "=== L0 per-fold leak gate (PREREG_TEACHER A1 FX3) ==="; grep -E '^L0 ' $D/real/build.log || echo "(no L0 line: the build stopped before it)"
[ "$RC" = 4 ] && { echo "L0 STOP (exit 4): the R7-teacher line stops here (PREREG_TEACHER A1 FX3). Report; install nothing."; exit 4; }
[ "$RC" = 0 ] || { echo "STOP: the REAL build failed rc $RC (see $D/real/build.log)"; exit 1; }
grep -E '^A1 tie report|^teacher logits' $D/real/build.log
TT=$($PY -c "import json; print(json.load(open('$D/real/teacher_labels_receipt.json'))['label_gate']['train_table'] or 'NONE')")
if [ "$TT" = NONE ]; then echo "LABEL GATE STOP: no table may be trained (PREREG_TEACHER sec 4). Report and end the line."; exit 2; fi
bash $D/labels_gate.sh "$TT"
T=${TT#labels_r7t_}; T=${T%.csv}; SH=$D/sheets/queue_r7t$T.txt
cat <<EOF

NEXT (main session decides; nothing below has been run):
  BOX route (~17.6 h for 5 folds, MEASURED 3.52 h/fold): put the 5 student lines at the TOP of the queue, f0 first
  (the L13 s1044 folds behind them are a filler seed):
    for F in 4 3 2 1 0; do bash ~/queue/qctl.sh top "\$(sed -n "\$((F + 1))p" $SH)"; done; bash ~/queue/qctl.sh check
  RENTED route (~4-5 h, ~\$8-10 on 4 x 4090): Downloads/rsna-knee/session_2026-10-15/r7teacher/labels/kits/vast_r7t/RUNBOOK.md
  (do not run both routes for the same table: the rental's box placeholders make the queue REFUSE those lines anyway).
  After 5/5 folds: student_gate.py (PREREG sec 6):
    cd $D && CUDA_VISIBLE_DEVICES= NTHR=6 taskset -c 34-39 nice -n 19 $PY student_gate.py --student 'f{F}_s$([ $T = b ] && echo 1047 || echo 1048)_ttr_r7t$T' \\
      --table real/$TT --receipt real/teacher_labels_receipt.json --teacher real/R7OOF.csv --infer
EOF
