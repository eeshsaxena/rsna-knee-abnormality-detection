#!/bin/bash
# K7 box CPU tests (CUDA hidden, nice 19, never touches the GPU). usage:
#   K7T=k7_test bash ~/k7_test/k5/box/k5_box_launch.sh <a0_main|a0_solo|all> [ids csv, default ids/a0.csv]
# = the K6 launcher for the K7 notebooks (rsna_sub_k7_fleet / rsna_sub_k7_solo) + --expect_tta ~/k7_test/K5_BUILD.json.
# K7-KEY: manifest names are the SHIPPED (staged) names; a re-keyed member (TTR288: f0_s1044t288_...) maps to its
# collected file and OOF through K5_BUILD.json picks (staged -> oof). Each test is detached; logs
# ~/k7_test/runs/<NAME>_<HHMM>.log; result line "BOXSTAGE PASS n/n". Per-member OOF parity (pass A rows) for every
# manifest member whose OOF holds an A0 study (fold-0 members: A f0, B f0, C f0 share the same A0 studies).
set -u
K7T=${K7T:-k7_test}; export K7T                      # box test area (env), default ~/k7_test
cd ~/$K7T || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
IDS=${2:-ids/a0.csv}
[ -f "$IDS" ] || { echo "missing $IDS (run k5_ids.py)"; exit 1; }
[ -f ds/rsna-knee-cnxt-fleet/fleet_manifest.json ] || { echo "dataset not staged"; exit 1; }
[ -f K5_BUILD.json ] || { echo "K5_BUILD.json not staged (k7_stage_box.sh)"; exit 1; }
OOFARGS=$($PYK - "$IDS" <<'EOF'
import csv, json, os, sys
H = os.path.expanduser('~'); T = os.environ.get('K7T', 'k7_test')
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
case $1 in
  a0_main|all) run A0_main --nb kernel/rsna_sub_k7_fleet.ipynb --ids $IDS --fleet_ds ds/rsna-knee-cnxt-fleet \
      --age_h 7.9 --threads 4 $OOFARGS ;;&
  a0_solo|all) run A0_solo --nb kernel/rsna_sub_k7_solo.ipynb --ids $IDS --fleet_ds ds/rsna-knee-cnxt-fleet \
      --age_h 7.9 --threads 4 $OOFARGS ;;&
  *) ;;
esac
