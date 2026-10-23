#!/bin/bash
# K9N box CPU A0 (CUDA hidden, nice 19, never touches the GPU; writes only ~/k9n_test). usage:
#   bash ~/k9n_test/k9n/box/k9n_box_launch.sh <stub|chain|neg|all|NAME>
# stub : A0n_stub_a / A0n_stub_b   N stage on 4 fold-0 studies (ids/n4.csv), fleet stage synthesized (strict, clean)
# chain: A0n_chain_a               the full K9N-a cells 31 + 32 on the K9 A0 pair (ids/a0.csv): real 10-member fleet on
#                                  CPU (~50 min) then the N stage; k9n_check part A = every K9 gate on the fleet part
# neg  : 7 negatives (fleet stub, 2 studies): missing_weights, bad_sha, fleet_fail, time_skip, wall_kill, child_crash,
#        child_sha; each must end K9NBOX PASS (N NOT clean, submission.csv ABSENT, why names the cause, no child left)
# Each test is detached (setsid nohup bash -c, ssh returns at once); logs ~/k9n_test/runs/<NAME>_<HHMM>.log.
set -u
T=k9n_test
cd ~/$T || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
[ -f K9N_BUILD.json ] && [ -f K5_BUILD.json ] && [ -d ds/rsna-knee-cnxt-fleet ] || { echo "not staged (k9n_stage_box.sh)"; exit 1; }
head -n 3 ids/a0.csv > ids/neg2.csv
H=$(date +%H%M)
run() {   # name, args...
  local name=$1; shift
  local args="$*"
  setsid nohup bash -c "cd ~/$T && CUDA_VISIBLE_DEVICES= nice -n 19 $PYK -W ignore k9n/box/k9n_box_stage.py --k9n_build K9N_BUILD.json --weights_dir wts --out runs/${name}_$H $args" > runs/${name}_$H.log 2>&1 < /dev/null &
  echo "started $name (pid $!) log runs/${name}_$H.log"
}
STUB_A="--nb kernel/rsna_sub_k9n_a.ipynb --tag a --fleet stub"
case $1 in
  stub|all)  run A0n_stub_a $STUB_A --ids ids/n4.csv
             run A0n_stub_b --nb kernel/rsna_sub_k9n_b.ipynb --tag b --fleet stub --ids ids/n4.csv ;;&
  chain|all) run A0n_chain_a --nb kernel/rsna_sub_k9n_a.ipynb --tag a --fleet real --fleet_ds ds/rsna-knee-cnxt-fleet \
               --ids ids/a0.csv --expect_tta K5_BUILD.json --expect_manifest ds/rsna-knee-cnxt-fleet/fleet_manifest.json ;;&
  neg|all)   for n in missing_weights bad_sha fleet_fail time_skip wall_kill child_crash child_sha; do
               run A0n_neg_$n $STUB_A --ids ids/neg2.csv --neg $n; done ;;
  *) ;;
esac
