#!/bin/bash
# a1_dryrun.sh (r7teacher/labels, 2026-10-14; BOX, CPU only: CUDA hidden, nice 19, ionice idle, cores ${CORES:-34-39}).
# PREREG_TEACHER Amendment A1 dry run, end to end, with the REALISTIC simulated teacher (A1 FX6). Never touches the GPU, the
# queue (~/queue) or ~/fleet_runs; writes only under ~/r7t_labels/{sim,sim_a1,dry,gate} and a temporary /tmp dir.
#   R0  regression: the A1 builder on the pre-A1 SIM teacher (no z| columns) reproduces the pre-A1 SIM tables byte for byte
#   S1  realistic SIM teacher (DT order, real |z| magnitudes of the 44 real rows, stacker-like p| on Effusion / Baker's)
#   B0  the PRE-A1 builder on it (shows the clip bug: tied cells, L1)
#   B1  the A1 builder on it -> sim/tables (the SIM tables the SIMCHECK sheets and the gate use)
#   F2  the same teacher with another stacker seed (p| of Effusion / Baker's changed, z| identical): tables must be identical
#   L1/L2 L0 negative controls: an in-fold leak on hash fold 2 (DROP expected) and on folds 1 + 2 (STOP, exit 4, expected)
#   RG  REAL-path guards in a temp dir (REAL tag without z| refused; PREREG without the A1 code freeze refused; wrong sha refused;
#       a temp PREREG listing this builder's md5 passes) -> temp dir deleted
#   TR  trainer argv on the real fleet_train.py (CPU, 31 studies, 1 epoch) with the A1 SIMCHECK sheets: rc 0, sel_labels == table
#   G   student gate stand-ins with the A1 SIM teacher + tables: E2 (did not learn the teacher: G2 FAIL, G3' PASS) and DT (the
#       teacher itself: G3 FAIL, G3' FAIL)
set -u
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
C=${CORES:-34-39}; D=$HOME/r7t_labels; PY=$HOME/miniconda3/envs/rsna/bin/python; O=$D/sim_a1; mkdir -p $O
RUN="taskset -c $C nice -n 19 ionice -c3 $PY"
cd $D || exit 1
log(){ echo "[a1 $(date +%H:%M:%S)] $*"; }
BUILD="--fleet_oof fleet/oof_fleetOF_mean.csv --fleet_gold fleet/gold_fleetOF_mean.csv"
log "code md5: $(md5sum build_teacher_labels.py student_gate.py sim_r7_oof.py r7_absz_ref44.json make_sheets.sh on_r7oof.sh dry_run_student.sh | awk '{printf "%s %s | ", substr($1,1,8), $2}')"

log "R0 regression: A1 builder on the pre-A1 SIM teacher (plain p| layout, no z|)"
$RUN build_teacher_labels.py --teacher sim/SIM_R7OOF_K14DT_job1fmt.csv --teacher_tag SIM:K14-DT $BUILD --out_dir $O/r0 > $O/r0.log 2>&1; echo "rc $?"
grep -E '^L0 |^LABEL_GATE|^TEACHER_LABELS_OK' $O/r0.log
log "R0 table md5 (pre-A1 SIM: T_a 8ec090db, T_b 8fbf5787): $(md5sum $O/r0/SIM_labels_r7t_a.csv $O/r0/SIM_labels_r7t_b.csv | awk '{printf "%s ", substr($1,1,8)}')"

log "S1 realistic SIM teacher (A1 FX6)"
$RUN sim_r7_oof.py --oof sim/oof_DT.csv --gold_mean sim/gold_simDT_mean.csv --absz_ref r7_absz_ref44.json --out sim/SIM_R7OOF_K14DT_A1.csv
$RUN sim_r7_oof.py --oof sim/oof_DT.csv --gold_mean sim/gold_simDT_mean.csv --absz_ref r7_absz_ref44.json --stacker_seed 12 --out $O/SIM_R7OOF_K14DT_A1_s12.csv
$RUN sim_r7_oof.py --oof sim/oof_DT.csv --gold_mean sim/gold_simDT_mean.csv --absz_ref r7_absz_ref44.json --leak_folds 2 --leak ${LEAK:-2.0} --out $O/SIM_R7OOF_K14DT_A1_leak1.csv
$RUN sim_r7_oof.py --oof sim/oof_DT.csv --gold_mean sim/gold_simDT_mean.csv --absz_ref r7_absz_ref44.json --leak_folds 1 2 --leak ${LEAK:-2.0} --out $O/SIM_R7OOF_K14DT_A1_leak2.csv

log "B0 PRE-A1 builder on the realistic SIM teacher"
$RUN pre_a1/build_teacher_labels.py --teacher sim/SIM_R7OOF_K14DT_A1.csv --teacher_tag SIM:K14-DT $BUILD --out_dir $O/b0 > $O/b0.log 2>&1; echo "rc $?"
grep -E '^L1 target|^LABEL_GATE|^mean per-label' $O/b0.log

log "B1 A1 builder on the realistic SIM teacher -> sim/tables (pre-A1 SIM tables kept in sim/tables_pre_a1)"
[ -d sim/tables_pre_a1 ] || cp -rp sim/tables sim/tables_pre_a1
$RUN build_teacher_labels.py --teacher sim/SIM_R7OOF_K14DT_A1.csv --teacher_tag SIM:K14-DT $BUILD --out_dir sim/tables > $O/b1.log 2>&1; echo "rc $?"
grep -E '^teacher logits|^L0 |^A1 tie|^L1 target|^LABEL_GATE|^mean per-label|^TEACHER_LABELS_OK|^WROTE' $O/b1.log

log "F2 stacker-seed invariance (p| of Effusion / Baker's changed, z| identical)"
$RUN build_teacher_labels.py --teacher $O/SIM_R7OOF_K14DT_A1_s12.csv --teacher_tag SIM:K14-DT $BUILD --out_dir $O/f2 > $O/f2.log 2>&1; echo "rc $?"
for t in a b; do cmp -s sim/tables/SIM_labels_r7t_$t.csv $O/f2/SIM_labels_r7t_$t.csv && echo "F2 T_$t IDENTICAL" || echo "F2 T_$t DIFFERENT"; done
cmp -s sim/SIM_R7OOF_K14DT_A1.csv $O/SIM_R7OOF_K14DT_A1_s12.csv && echo "F2 teacher files identical (test void)" || echo "F2 teacher files differ (as intended)"

log "L1 L0 negative control: leak on hash fold 2"
$RUN build_teacher_labels.py --teacher $O/SIM_R7OOF_K14DT_A1_leak1.csv --teacher_tag SIM:K14-DT $BUILD --out_dir $O/leak1 > $O/leak1.log 2>&1; echo "rc $?"
grep -E '^L0 |^LABEL_GATE|^WROTE|^TEACHER_LABELS_OK' $O/leak1.log
log "L2 L0 negative control: leak on hash folds 1 + 2"
$RUN build_teacher_labels.py --teacher $O/SIM_R7OOF_K14DT_A1_leak2.csv --teacher_tag SIM:K14-DT $BUILD --out_dir $O/leak2 > $O/leak2.log 2>&1; echo "rc $? (4 expected)"
grep -E '^L0' $O/leak2.log; ls $O/leak2

log "RG REAL-path guards (temp dir, deleted after)"
TD=$(mktemp -d /tmp/r7t_a1_rg.XXXXXX)
$PY - "$D/sim/SIM_R7OOF_K14DT_A1.csv" "$TD" <<'EOF'
import sys, pandas as pd
src, td = sys.argv[1:3]
d = pd.read_csv(src, dtype=str).drop(columns=['teacher_tag'])
d.to_csv(f'{td}/REAL_like.csv', index=False)                                   # SIM values under a REAL tag (test only)
d.drop(columns=[c for c in d.columns if c.startswith('z|')]).to_csv(f'{td}/REAL_like_noz.csv', index=False)
EOF
ME=$(md5sum build_teacher_labels.py | cut -c1-32)
printf '# temp PREREG for the guard test\n## Amendment A1\nbuild_teacher_labels.py %s\n' "$ME" > $TD/P_ok.md; sha256sum < $TD/P_ok.md | cut -c1-64 > $TD/P_ok.sha256
printf '# temp PREREG without the code freeze\n## Amendment A1\n' > $TD/P_nomd5.md; sha256sum < $TD/P_nomd5.md | cut -c1-64 > $TD/P_nomd5.sha256
echo 0000000000000000000000000000000000000000000000000000000000000000 > $TD/P_bad.sha256
g(){ $RUN build_teacher_labels.py --teacher $1 --teacher_tag R7OOF $BUILD --out_dir $TD/out_$2 --prereg $3 --prereg_sha $4 > $TD/$2.log 2>&1; echo "RG $2 rc $? | $(grep -E 'AssertionError|^LABEL_GATE|^TEACHER_LABELS_OK' $TD/$2.log | tail -n 1 | cut -c1-200)"; }
g $TD/REAL_like_noz.csv noz $TD/P_ok.md $TD/P_ok.sha256
g $TD/REAL_like.csv nomd5 $TD/P_nomd5.md $TD/P_nomd5.sha256
g $TD/REAL_like.csv badsha $TD/P_ok.md $TD/P_bad.sha256
g $TD/REAL_like.csv ok $TD/P_ok.md $TD/P_ok.sha256
grep -E '^builder md5|^L0 ' $TD/ok.log
rm -rf "$TD"; log "RG temp dir removed"

log "TR trainer argv dry run (A1 SIMCHECK sheets; CPU)"
[ -f $D/dry/dry_run.log ] && mv -n $D/dry/dry_run.log $D/dry/dry_run_pre_a1.log
CORES=$C bash $D/dry_run_student.sh
cat $D/dry/dry_run.log

log "G student gate stand-ins (A1 SIM teacher + tables)"
for S in 'f{F}_s1046_ttr288frac75' 'f{F}_s1043_ttr_frac75oaiT123v00'; do
  NTHR=6 $RUN student_gate.py --student "$S" --table sim/tables/SIM_labels_r7t_b.csv --receipt sim/tables/SIM_teacher_labels_receipt.json \
    --teacher sim/SIM_R7OOF_K14DT_A1.csv --standin --out $D/gate 2>&1 | grep -vE '^ *"|^ *[{}]|^ *\]' | tail -n 14
done
log "A1 DRY RUN DONE"
