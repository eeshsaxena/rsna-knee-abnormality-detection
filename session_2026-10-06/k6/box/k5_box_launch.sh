#!/bin/bash
# K5 box CPU tests (CUDA hidden, nice 19, never touches the GPU). usage:
#   bash ~/k6_test/k5/box/k5_box_launch.sh <a0_main|a0_solo|a0_w50|all> [ids csv, default ids/a0.csv]
# = k4_box_launch.sh for the K5 notebooks (rsna_sub_k6_*.ipynb) + --expect_tta ~/k6_test/K5_BUILD.json (the build's
# TTA map; k2_check verifies it from the receipt). Each test is detached; logs ~/k6_test/runs/<NAME>_<HHMM>.log;
# result line "BOXSTAGE PASS n/n". Per-member OOF parity (pass A rows) for every manifest member whose OOF holds an A0
# study (fold 0: J1, M1); M1's OOF = ~/fleet_runs/f0_s42/oof_f0_s42_raw.csv (read only).
set -u
cd ~/k6_test || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
IDS=${2:-ids/a0.csv}
[ -f "$IDS" ] || { echo "missing $IDS (run k5_ids.py)"; exit 1; }
[ -f ds/rsna-knee-cnxt-fleet/fleet_manifest.json ] || { echo "dataset not staged"; exit 1; }
[ -f K5_BUILD.json ] || { echo "K5_BUILD.json not staged (k5_stage_box.sh)"; exit 1; }
OOFARGS=$($PYK - "$IDS" <<'EOF'
import csv, json, os, sys
H = os.path.expanduser('~')
man = json.load(open(f'{H}/k6_test/ds/rsna-knee-cnxt-fleet/fleet_manifest.json'))['checkpoints']
mem = {r['file']: r['oof'] for r in json.load(open(f'{H}/k6_test/members.json'))['members']}
mem['f0_s42_raw_cnxt320x24_cv8810_best.pt'] = f'{H}/fleet_runs/f0_s42/oof_f0_s42_raw.csv'
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
case $1 in
  a0_main|all) run A0_main --nb kernel/rsna_sub_k6_fleet.ipynb --ids $IDS --fleet_ds ds/rsna-knee-cnxt-fleet \
      --age_h 7.9 --threads 4 $OOFARGS ;;&
  a0_solo|all) run A0_solo --nb kernel/rsna_sub_k6_solo.ipynb --ids $IDS --fleet_ds ds/rsna-knee-cnxt-fleet \
      --age_h 7.9 --threads 4 $OOFARGS ;;&
  a0_w50|all) [ -f kernel/rsna_sub_k6_w50.ipynb ] && run A0_w50 --nb kernel/rsna_sub_k6_w50.ipynb --ids $IDS \
      --fleet_ds ds/rsna-knee-cnxt-fleet --age_h 7.9 --threads 4 $OOFARGS ;;&
  *) ;;
esac
