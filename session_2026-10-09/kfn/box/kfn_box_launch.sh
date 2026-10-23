#!/bin/bash
# kfn box CPU A0 (CUDA hidden, nice 19, never touches the GPU; writes only ~/kfn_test). usage:
#   bash ~/kfn_test/kfn/box/kfn_box_launch.sh <chain|stub|nonly|gold|neg|all>
# chain: A0fn_chain   the full FN kernel (FN-IDS + real 10-member K9 fleet on CPU + N) on the K9 A0 pair (ids/a0.csv)
# stub : A0fn_stub    FN-IDS + stubbed strict fleet + N on 4 fold-0 studies (ids/n4.csv)
# nonly: A0no_n4      the N-only kernel on the same 4 studies
# gold : A0no_gold12  the N-only kernel on 12 gold58 studies (n >= 11 gate path; label-order sanity via kfn_gold_sanity.py)
# neg  : negatives, run ONE AFTER ANOTHER in one detached job (light load): FN stub: missing_weights bad_sha fleet_fail
#        time_skip wall_kill child_crash child_sha ids_fail; N-only: missing_weights bad_sha child_crash ids_fail;
#        FN real: fleet_missing (fleet dataset not attached; no inference)
# Each job is detached (setsid nohup bash -c, ssh returns at once); logs ~/kfn_test/runs/<NAME>_<HHMM>.log.
set -u
T=kfn_test
cd ~/$T || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
[ -f FN_BUILD.json ] && [ -f K5_BUILD.json ] && [ -d ds/rsna-knee-cnxt-fleet ] || { echo "not staged (kfn_stage_box.sh)"; exit 1; }
head -n 3 ids/a0.csv > ids/neg2.csv
H=$(date +%H%M)
STAGE="CUDA_VISIBLE_DEVICES= nice -n 19 $PYK -W ignore kfn/box/kfn_box_stage.py --fn_build FN_BUILD.json --weights_dir wts"
FN="--nb kernel/rsna_sub_k9fn.ipynb --kernel fn"
NO="--nb kernel/rsna_sub_nonly.ipynb --kernel nonly"
REAL="--fleet real --fleet_ds ds/rsna-knee-cnxt-fleet --expect_tta K5_BUILD.json --expect_manifest ds/rsna-knee-cnxt-fleet/fleet_manifest.json"
run() {   # name, args...
  local name=$1; shift
  setsid nohup bash -c "cd ~/$T && $STAGE --out runs/${name}_$H $*" > runs/${name}_$H.log 2>&1 < /dev/null &
  echo "started $name (pid $!) log runs/${name}_$H.log"
}
case $1 in
  chain|all) run A0fn_chain $FN $REAL --ids ids/a0.csv ;;&
  stub|all)  run A0fn_stub $FN --fleet stub --ids ids/n4.csv ;;&
  nonly|all) run A0no_n4 $NO --ids ids/n4.csv ;;&
  gold|all)  run A0no_gold12 $NO --ids ids/gold_a.csv ;;&
  neg|all)
    CMDS=""
    for n in missing_weights bad_sha fleet_fail time_skip wall_kill child_crash child_sha ids_fail; do
      CMDS="$CMDS $STAGE --out runs/A0fn_neg_${n}_$H $FN --fleet stub --ids ids/neg2.csv --neg $n > runs/A0fn_neg_${n}_$H.log 2>&1;"
    done
    for n in missing_weights bad_sha child_crash ids_fail; do
      CMDS="$CMDS $STAGE --out runs/A0no_neg_${n}_$H $NO --ids ids/neg2.csv --neg $n > runs/A0no_neg_${n}_$H.log 2>&1;"
    done
    CMDS="$CMDS $STAGE --out runs/A0fn_neg_fleet_missing_$H $FN $REAL --ids ids/neg2.csv --neg fleet_missing > runs/A0fn_neg_fleet_missing_$H.log 2>&1;"
    setsid nohup bash -c "cd ~/$T && $CMDS echo NEGS DONE" > runs/A0_negs_$H.log 2>&1 < /dev/null &
    echo "started negatives (pid $!) log runs/A0_negs_$H.log" ;;
  *) ;;
esac
