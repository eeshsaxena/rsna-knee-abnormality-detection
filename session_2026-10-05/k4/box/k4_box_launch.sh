#!/bin/bash
# K4 box CPU tests (CUDA hidden, nice 19, never touches the GPU). usage:
#   bash ~/k4_test/k4/box/k4_box_launch.sh <a0_main|a0_solo|a0_w50|all> [ids csv, default ids/a0.csv]
# Each test is detached; logs in ~/k4_test/runs/<NAME>_<HHMM>.log; result line "BOXSTAGE PASS n/n".
# Members + OOFs come from ~/k4_test/members.json (k4_pick.py --copy): every member that is in the staged dataset
# ~/k4_test/ds/rsna-knee-cnxt-fleet/fleet_manifest.json gets a per-member OOF parity check (--oof_member); M1
# (f0_s42 family 1, only with build_k4.py --with_m1) is checked against ~/fleet_runs/f0_s42/oof_f0_s42_raw.csv.
set -u
cd ~/k4_test || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
IDS=${2:-ids/a0.csv}
[ -f "$IDS" ] || { echo "missing $IDS (run k4_ids.py)"; exit 1; }
[ -f ds/rsna-knee-cnxt-fleet/fleet_manifest.json ] || { echo "dataset not staged"; exit 1; }
OOFARGS=$($PYK - "$IDS" <<'EOF'
import csv, json, os, sys
H = os.path.expanduser('~')
man = json.load(open(f'{H}/k4_test/ds/rsna-knee-cnxt-fleet/fleet_manifest.json'))['checkpoints']
mem = {r['file']: r['oof'] for r in json.load(open(f'{H}/k4_test/members.json'))['members']}
mem['f0_s42_raw_cnxt320x24_cv8810_best.pt'] = f'{H}/fleet_runs/f0_s42/oof_f0_s42_raw.csv'
miss = [n for n in man if n not in mem]
assert not miss, f'no OOF for {miss}'
# VERIFY fix 10-05: the box DICOM mirror holds fold 0 + gold only, so a fold 1-4 member's OOF shares no A0 study and
# its per-member parity check would FAIL on 0 studies. Parity args only for members with >= 1 A0 study in their OOF.
ids = {r['StudyInstanceUID'] for r in csv.DictReader(open(sys.argv[1]))}
have = [n for n in man if ids & {r['StudyInstanceUID'] for r in csv.DictReader(open(mem[n]))}]
for n in man:
    if n not in have:
        print(f'NOTE {n}: no A0 study in its OOF (no DICOMs for its fold on the box): per-member OOF parity skipped '
              f'here (npz mean identity + decode probe still cover it; Kaggle commit gates it)', file=sys.stderr)
assert have, 'no manifest member has an A0 study in its OOF'
print(' '.join(f'--oof_member {n}={mem[n]}' for n in have))
EOF
) || { echo "OOF args failed"; exit 1; }
T=$(date +%H%M)
run() {   # name, args...
  local name=$1; shift
  CUDA_VISIBLE_DEVICES= nohup setsid nice -n 19 $PYK -W ignore k4/box/k2_box_stage.py --cpu "$@" --out runs/${name}_$T \
    > runs/${name}_$T.log 2>&1 < /dev/null &
  echo "started $name pid $! log runs/${name}_$T.log"
}
case $1 in
  a0_main|all) run A0_main --nb kernel/rsna_sub_k4_fleet.ipynb --ids $IDS --fleet_ds ds/rsna-knee-cnxt-fleet \
      --age_h 7.9 --threads 4 $OOFARGS ;;&
  a0_solo|all) run A0_solo --nb kernel/rsna_sub_k4_solo.ipynb --ids $IDS --fleet_ds ds/rsna-knee-cnxt-fleet \
      --age_h 7.9 --threads 4 $OOFARGS ;;&
  a0_w50|all) [ -f kernel/rsna_sub_k4_w50.ipynb ] && run A0_w50 --nb kernel/rsna_sub_k4_w50.ipynb --ids $IDS \
      --fleet_ds ds/rsna-knee-cnxt-fleet --age_h 7.9 --threads 4 $OOFARGS ;;&
  *) ;;
esac
