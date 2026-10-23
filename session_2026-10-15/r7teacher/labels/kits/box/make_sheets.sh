#!/bin/bash
# make_sheets.sh (r7teacher/labels, 2026-10-14): writes the box queue sheets for the R7-teacher student (NOT installed;
# the main session installs one with `bash ~/queue/qctl.sh install` or appends its lines, after the REAL label build).
# argv = the box TTR224 RSNA-only fold recipe (21 ep, gc0, seed-specific out dir) token for token, == vastS1044
# argv_template.txt with ~ paths, only --labels / --seed / --fold / --out differ. Gate = labels_gate.sh (REAL receipt,
# PREREG sha, label gate chose this table, md5 unchanged).
#   T_b (PRIMARY)  seed 1047, out ~/fleet_runs/fF_s1047_ttr_r7tb, labels ~/r7t_labels/real/labels_r7t_b.csv
#   T_a (fallback) seed 1048, out ~/fleet_runs/fF_s1048_ttr_r7ta, labels ~/r7t_labels/real/labels_r7t_a.csv
# PREREG_TEACHER Amendment A1 FX4: --eval_labels = the student's OWN table (was v4), so the trainer's raw-best checkpoint and
# its OOF follow the AUC vs the table on the held-out val rows (rewards generalising the teacher, cannot reward copying it);
# the ckpt args then carry sel_labels == train_labels == the table name, which student_gate.py G0 requires.
# usage: bash make_sheets.sh OUTDIR [SIM]   (SIM: point --labels at the SIM tables, gate '-', out dirs under r7tdry: for --check only)
set -u
O=${1:?outdir}; SIM=${2:-}
for T in b a; do
  S=$([ $T = b ] && echo 1047 || echo 1048); TAG=ttr_r7t$T; TB=labels_r7t_$T.csv
  LAB="~/r7t_labels/real/$TB"; GATE="needs PASS from ~/r7t_labels/real/labels_gate_labels_r7t_$T.verdict wait 120 via bash ~/r7t_labels/labels_gate.sh $TB"
  [ -n "$SIM" ] && { LAB="~/r7t_labels/sim/tables/SIM_$TB"; GATE="-"; TAG=${TAG}dry; }
  f=$O/queue_r7t$T${SIM:+_SIMCHECK}.txt; : > "$f"
  for F in 0 1 2 3 4; do
    echo "R7T$T S$S f$F (TTR224 s$S R7-teacher T_$T, 21 ep, gc0, sel on own table; PREREG_TEACHER A1) | ~/fleet_runs/f${F}_s${S}_${TAG} | $GATE | --model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 21 --seed $S --save_epochs 6-21 --workers 10 --store ~/fleet_store --labels $LAB --folds ~/rsna_train/folds5.csv --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels $LAB --gold ~/rsna_train/gold74_eval_set.csv --fold $F --out ~/fleet_runs/f${F}_s${S}_${TAG} --grad_ckpt 0" >> "$f"
  done
  echo "wrote $f"
done
