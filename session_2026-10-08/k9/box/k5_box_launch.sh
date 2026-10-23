#!/bin/bash
# K9 box CPU tests (CUDA hidden, nice 19, never touches the GPU). usage:
#   K9T=k9_test bash ~/k9_test/k5/box/k5_box_launch.sh <a0_main|a0_solo|a0_kf|all|kfneg> [ids csv, default ids/a0.csv]
# = the K7 launcher for the three K9 notebooks (rsna_sub_k9_fleet / rsna_sub_k9_solo / rsna_sub_k9_kf) + --expect_tta
# ~/$K9T/K5_BUILD.json. a0_kf runs the KF twin with a synthesized, map-consistent parent (k2_box_stage.py --kf_parent ok)
# and --expect_outer_flat 0.6 (K8-KF assertions + the independent flat-parent recompute). kfneg = two NEGATIVE KF runs
# (--kf_parent perturb / missing): the KF self-check must fail BEFORE any worker starts, submission.csv ABSENT (strict),
# so each ends "BOXSTAGE PASS" (consistent fail-loud) with why naming the KF check, in ~1-2 min.
# K7-KEY: manifest names are the SHIPPED (staged) names; re-keyed members (K9: every member, f{F}_s1044t288_... and f{F}_s1043f75_...) map to their
# collected file and OOF through K5_BUILD.json picks. Each test is detached; logs ~/$K9T/runs/<NAME>_<HHMM>.log.
# Per-member OOF parity (pass A rows) for every manifest member whose OOF holds an A0 study (the box DICOM mirror =
# fold 0 + gold: the fold-0 members D f0 and C f0 share the same 2 A0 studies; folds 1-4 have no DICOMs on the box).
set -u
K9T=${K9T:-k9_test}; export K9T                      # box test area (env), default ~/k9_test
cd ~/$K9T || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
IDS=${2:-ids/a0.csv}
[ -f "$IDS" ] || { echo "missing $IDS (run k5_ids.py)"; exit 1; }
[ -f ds/rsna-knee-cnxt-fleet/fleet_manifest.json ] || { echo "dataset not staged"; exit 1; }
[ -f K5_BUILD.json ] || { echo "K5_BUILD.json not staged (k9_stage_box.sh)"; exit 1; }
OOFARGS=$($PYK - "$IDS" <<'EOF'
import csv, json, os, sys
H = os.path.expanduser('~'); T = os.environ.get('K9T', 'k9_test')
man = json.load(open(f'{H}/{T}/ds/rsna-knee-cnxt-fleet/fleet_manifest.json'))['checkpoints']
picks = json.load(open(f'{H}/{T}/K5_BUILD.json'))['picks']
oof_by_file = {r['file']: r['oof'] for r in json.load(open(f'{H}/{T}/members.json'))['members']}
mem = {p['staged']: oof_by_file[p['file']] for p in picks}
miss = [n for n in man if n not in mem]
assert not miss, f'no OOF for {miss}'
ids = {r['StudyInstanceUID'] for r in csv.DictReader(open(sys.argv[1]))}
have = [n for n in man if ids & {r['StudyInstanceUID'] for r in csv.DictReader(open(mem[n]))}]
for n in man:
    if n not in have:
        print(f'NOTE {n}: no A0 study in its OOF (no DICOMs for its fold on the box): per-member OOF parity skipped '
              f'here (npz identities + decode probe still cover it; Kaggle commit gates it)', file=sys.stderr)
assert have, 'no manifest member has an A0 study in its OOF'
print(' '.join(f'--oof_member {n}={mem[n]}' for n in have))
EOF
) || { echo "OOF args failed"; exit 1; }
T=$(date +%H%M)
run() {   # name, args...
  local name=$1; shift
  CUDA_VISIBLE_DEVICES= nohup setsid nice -n 19 $PYK -W ignore k5/box/k2_box_stage.py --cpu "$@" --out runs/${name}_$T \
    --expect_tta K5_BUILD.json > runs/${name}_$T.log 2>&1 < /dev/null &
  echo "started $name pid $! log runs/${name}_$T.log"
}
# FIX 10-08: the A0 wall window was --age_h 7.9 (0.7 h + 300 s grace to the 8.6 h wall). A 10-member kit with 5 x TTR288
# needs ~45-60 min on the loaded box CPU (the microbench alone ~35 min), so the K8 sim A0 hit the wall at 47.0 min
# (fail-loud, submission.csv deleted). The A0 tests correctness; T4 timing is modelled from the commit microbench.
A0_AGE_H=${A0_AGE_H:-7.0}                            # 1.6 h window on CPU
COMMON="--ids $IDS --fleet_ds ds/rsna-knee-cnxt-fleet --age_h $A0_AGE_H --threads 4"
case $1 in
  a0_main|all) run A0_main --nb kernel/rsna_sub_k9_fleet.ipynb $COMMON $OOFARGS ;;&
  a0_solo|all) run A0_solo --nb kernel/rsna_sub_k9_solo.ipynb $COMMON $OOFARGS ;;&
  a0_kf|all)   run A0_kf --nb kernel/rsna_sub_k9_kf.ipynb $COMMON $OOFARGS --kf_parent ok --expect_outer_flat 0.6 ;;&
  kfneg)       run A0_kfneg_perturb --nb kernel/rsna_sub_k9_kf.ipynb $COMMON --kf_parent perturb --expect_outer_flat 0.6
               run A0_kfneg_missing --nb kernel/rsna_sub_k9_kf.ipynb $COMMON --kf_parent missing --expect_outer_flat 0.6 ;;
  *) ;;
esac
