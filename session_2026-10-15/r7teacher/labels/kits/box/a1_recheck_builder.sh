#!/bin/bash
# a1_recheck_builder.sh (r7teacher/labels, 10-14; BOX, CPU only, nice 19, cores ${CORES:-34-39}). Re-runs every BUILDER step of
# a1_dryrun.sh with the FINAL A1 builder (after the DROP-branch fix: tables may hold v4's hard 0 / 1 on rows whose only source
# is v4), into ~/r7t_labels/sim_a1/final/, and checks that the realistic-SIM tables are byte-identical to the ones the trainer
# argv dry run and the student gate stand-ins used (sim/tables, built by the pre-fix A1 builder), so those results carry over.
set -u
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
C=${CORES:-34-39}; D=$HOME/r7t_labels; PY=$HOME/miniconda3/envs/rsna/bin/python; O=$D/sim_a1/final; mkdir -p $O
RUN="taskset -c $C nice -n 19 ionice -c3 $PY"; cd $D || exit 1
BUILD="--fleet_oof fleet/oof_fleetOF_mean.csv --fleet_gold fleet/gold_fleetOF_mean.csv"
log(){ echo "[a1re $(date +%H:%M:%S)] $*"; }
log "builder md5 $(md5sum build_teacher_labels.py | cut -c1-32)"
b(){ $RUN build_teacher_labels.py --teacher $1 --teacher_tag SIM:K14-DT $BUILD --out_dir $O/$2 > $O/$2.log 2>&1; echo "rc $?"; }
log "R0 (pre-A1 SIM teacher, plain layout)"; b sim/SIM_R7OOF_K14DT_job1fmt.csv r0
log "R0 md5 $(md5sum $O/r0/SIM_labels_r7t_a.csv $O/r0/SIM_labels_r7t_b.csv | awk '{printf "%s ", substr($1,1,8)}') (pre-A1: 8ec090db 8fbf5787)"
log "B1 (realistic SIM teacher)"; b sim/SIM_R7OOF_K14DT_A1.csv b1
grep -E '^L0 |^WROTE|^LABEL_GATE|^TEACHER_LABELS_OK' $O/b1.log | cut -c1-260
for t in a b; do cmp -s $O/b1/SIM_labels_r7t_$t.csv sim/tables/SIM_labels_r7t_$t.csv && echo "B1 T_$t IDENTICAL to sim/tables (TR + G results carry over)" || echo "B1 T_$t DIFFERENT"; done
log "F2 (stacker seed 12)"; b sim_a1/SIM_R7OOF_K14DT_A1_s12.csv f2
for t in a b; do cmp -s $O/b1/SIM_labels_r7t_$t.csv $O/f2/SIM_labels_r7t_$t.csv && echo "F2 T_$t IDENTICAL" || echo "F2 T_$t DIFFERENT"; done
log "L1 (leak on hash fold 2: DROP expected, tables written)"; b sim_a1/SIM_R7OOF_K14DT_A1_leak1.csv leak1
grep -E '^L0 |^WROTE|^LABEL_GATE|^TEACHER_LABELS_OK' $O/leak1.log | cut -c1-300
log "L2 (leak on hash folds 1 + 2: STOP exit 4 expected)"; b sim_a1/SIM_R7OOF_K14DT_A1_leak2.csv leak2
grep -E '^L0' $O/leak2.log | cut -c1-300; ls $O/leak2
log "RG REAL-path guards (temp dir, deleted after)"
TD=$(mktemp -d /tmp/r7t_a1_rg.XXXXXX)
$PY - "$D/sim/SIM_R7OOF_K14DT_A1.csv" "$TD" <<'EOF'
import sys, pandas as pd
src, td = sys.argv[1:3]
d = pd.read_csv(src, dtype=str).drop(columns=['teacher_tag'])
d.to_csv(f'{td}/REAL_like.csv', index=False)
d.drop(columns=[c for c in d.columns if c.startswith('z|')]).to_csv(f'{td}/REAL_like_noz.csv', index=False)
EOF
ME=$(md5sum build_teacher_labels.py | cut -c1-32)
printf '# temp PREREG for the guard test\n## Amendment A1\nbuild_teacher_labels.py %s\n' "$ME" > $TD/P_ok.md; sha256sum < $TD/P_ok.md | cut -c1-64 > $TD/P_ok.sha256
printf '# temp PREREG without the code freeze\n## Amendment A1\n' > $TD/P_nomd5.md; sha256sum < $TD/P_nomd5.md | cut -c1-64 > $TD/P_nomd5.sha256
echo 0000000000000000000000000000000000000000000000000000000000000000 > $TD/P_bad.sha256
g(){ $RUN build_teacher_labels.py --teacher $1 --teacher_tag R7OOF $BUILD --out_dir $TD/out_$2 --prereg $3 --prereg_sha $4 > $TD/$2.log 2>&1; echo "RG $2 rc $? | $(grep -E 'AssertionError|^TEACHER_LABELS_OK' $TD/$2.log | tail -n 1 | cut -c1-170)"; }
g $TD/REAL_like_noz.csv noz $TD/P_ok.md $TD/P_ok.sha256
g $TD/REAL_like.csv nomd5 $TD/P_nomd5.md $TD/P_nomd5.sha256
g $TD/REAL_like.csv badsha $TD/P_ok.md $TD/P_bad.sha256
g $TD/REAL_like.csv ok $TD/P_ok.md $TD/P_ok.sha256
rm -rf "$TD"; log "RG temp dir removed"
log "A1 BUILDER RECHECK DONE"
