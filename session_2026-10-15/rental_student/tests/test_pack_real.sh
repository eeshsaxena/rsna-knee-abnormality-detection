#!/bin/bash
# real_pos.sh (box CPU, seconds; ~/rstud_tests/realpos): POSITIVE REAL-mode control of pack_check (a launch must not be
# blocked by the real finalize_labels.sh format) + REAL negatives for the review checks
set -u
T=$HOME/rstud_tests/realpos; rm -rf $T; mkdir -p $T/final $T/real; PY=$HOME/miniconda3/envs/rsna/bin/python; PC=$HOME/rstud/src/box/pack_check.py
SIM=$HOME/r7t_labels/sim_a1/final/b1; FRC=$HOME/r7t_labels/fleet/fleet_oof_receipt_fleetOF.json; n=0; bad=0
cp $SIM/SIM_labels_r7t_b.csv $T/final/labels_r7t_b.csv; cp $T/final/labels_r7t_b.csv $T/real/labels_r7t_b.csv
PS=$(cut -c1-64 $HOME/r7t_labels/PREREG_TEACHER.sha256)
$PY - $SIM/SIM_teacher_labels_receipt.json $T/final/teacher_labels_receipt.json $PS <<'P'
import json, sys
r = json.load(open(sys.argv[1])); r.update(real=True, teacher_tag='R7OOF', builder_md5='a7f8823d7216fb0a55d5fc7821607c78', prereg_sha256=sys.argv[3])
r['inputs']['teacher']['path'] = 'real/R7OOF.csv'; r['label_gate']['train_table'] = 'labels_r7t_b.csv'; r['l0'] = r.get('l0') or {'verdict': 'PASS'}
json.dump(r, open(sys.argv[2], 'w'), indent=1)
P
# LABELS_READY in finalize_labels.sh's exact line format
S=$(sha256sum $T/final/labels_r7t_b.csv | cut -c1-64); M=$(md5sum $T/final/labels_r7t_b.csv | cut -c1-32)
{ echo "# R7-teacher student label tables (PREREG_TEACHER + A1 + A2), written $(date '+%F %T') by finalize_labels.sh"; echo "prereg_sha256 $PS"
  echo "teacher_tag R7OOF real True"; echo "teacher_sha256 $(printf 'c%.0s' $(seq 1 64))"; echo "L0 PASS spread 0.0025 auc_per_fold [0.9] outlier None"
  echo "label_gate delta_vs_frac75 {'a': 0.001, 'b': 0.002} passes {'a': True, 'b': True} bar -0.01"
  echo "labels_gate T_b: PASS labels_r7t_b.csv md5 $M teacher R7OOF prereg ${PS:0:12}"; echo "labels_gate T_a: PASS labels_r7t_a.csv md5 $(printf 'd%.0s' $(seq 1 32)) teacher R7OOF prereg ${PS:0:12}"
  echo "table T_b labels_r7t_b.csv sha256 $S md5 $M seed 1047"; echo "table T_a labels_r7t_a.csv sha256 $(printf 'e%.0s' $(seq 1 64)) md5 $(printf 'd%.0s' $(seq 1 32)) seed 1048"
  echo "receipt teacher_labels_receipt.json sha256 $(sha256sum $T/final/teacher_labels_receipt.json | cut -c1-64)"; } > $T/final/LABELS_READY
echo "PASS labels_r7t_b.csv md5 $M teacher R7OOF prereg ${PS:0:12}" > $T/real/labels_gate_labels_r7t_b.verdict
run(){ $PY $PC table $T/final/labels_r7t_b.csv --mode REAL --ready $T/final/LABELS_READY --receipt ${RCPT:-$T/final/teacher_labels_receipt.json} \
        --fleet_receipt $FRC --real_copy $T/real/labels_r7t_b.csv --verdict ${VER:-$T/real/labels_gate_labels_r7t_b.verdict} > $T/out 2>&1; rc=$?
     [ $rc = 0 ] && echo PASS || { [ $rc = 3 ] && echo STOP || echo ERR$rc; }; }
chk(){ n=$((n + 1)); [ "$2" = "$3" ] && echo "ok | $1 | $3 | $(tail -n 1 $T/out | cut -c1-200)" || { echo "WRONG | $1 | want $2 got $3 | $(tail -n 1 $T/out | cut -c1-300)"; bad=$((bad + 1)); }; }
chk "REAL positive (finalize format, real receipt fields)" PASS $(run)
sed -i "s/^prereg_sha256 .*/prereg_sha256 $(printf 'f%.0s' $(seq 1 64))/" $T/final/LABELS_READY; chk "REAL: build PREREG sha != LABELS_READY prereg_sha256" STOP $(run)
sed -i "s/^prereg_sha256 .*/prereg_sha256 $PS/" $T/final/LABELS_READY
VER=$T/real/none.verdict; chk "REAL: no labels_gate verdict file" STOP $(run); VER=""
$PY -c "import json; r=json.load(open('$T/final/teacher_labels_receipt.json')); r['builder_md5']='0'*32; json.dump(r, open('$T/rc_b.json','w'))"
RCPT=$T/rc_b.json; chk "REAL: receipt from a non-frozen builder" STOP $(run); RCPT=""
echo x >> $T/real/labels_r7t_b.csv; chk "REAL: real/ copy differs from final/" STOP $(run)
echo "== $n REAL checks, $bad WRONG"; rm -rf $T
