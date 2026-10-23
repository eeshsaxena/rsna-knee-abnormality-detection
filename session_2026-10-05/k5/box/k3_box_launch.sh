#!/bin/bash
# K3 box CPU tests (CUDA hidden, nice 19). usage: bash ~/k3_test/k3/box/k3_box_launch.sh <what> [M2 file name]
#   what = a0_main | a0_ttrsolo | g58_main | all     (each one detached; logs in ~/k3_test/runs/)
# M1 = the K2 member; M2 = the shipped TTR file (default: the one f0_s1042_*_best.pt in ~/k3_test/ds/rsna-knee-cnxt-fleet)
# OOF for M2 = ~/k3_test/oof/oof_f0_s1042_<tag>_cv<NNNN>.csv: a copy taken together with the ckpt (same mtime).
set -u
cd ~/k3_test || exit 1
PYK=~/miniconda3/envs/kmimic/bin/python
M1=f0_s42_raw_cnxt320x24_cv8810_best.pt
M2=${2:-$(cd ds/rsna-knee-cnxt-fleet && ls f0_s1042_*_best.pt)}
TAG2=$(echo "$M2" | cut -d_ -f3)
CV2=$(echo "$M2" | sed 's/.*_cv\([0-9]*\)_best.pt/cv\1/')
OOF1=$HOME/fleet_runs/f0_s42/oof_f0_s42_raw.csv
OOF2=$HOME/k3_test/oof/oof_f0_s1042_${TAG2}_${CV2}.csv
T=$(date +%H%M)
[ -f "$OOF2" ] || { echo "missing $OOF2"; exit 1; }
run() {   # name, args...
  local name=$1; shift
  CUDA_VISIBLE_DEVICES= nohup setsid nice -n 19 $PYK -W ignore k3/box/k2_box_stage.py --cpu "$@" --out runs/${name}_$T \
    > runs/${name}_$T.log 2>&1 < /dev/null &
  echo "started $name pid $! log runs/${name}_$T.log"
}
case $1 in
  a0_main|all) run A0_main --nb kernel/rsna_sub_k3_fleet.ipynb --ids two.csv --fleet_ds ds/rsna-knee-cnxt-fleet \
      --age_h 7.9 --threads 4 --oof_member $M1=$OOF1 --oof_member $M2=$OOF2 ;;&
  a0_ttrsolo|all) run A0_ttrsolo --nb kernel/rsna_sub_k3_ttrsolo.ipynb --ids two.csv --fleet_ds ds/rsna-knee-cnxt-fleet \
      --age_h 7.9 --threads 4 --oof $OOF2 --oof_member $M2=$OOF2 ;;&
  g58_main|all) run G58_main --nb kernel/rsna_sub_k3_fleet.ipynb --ids gold58.csv --fleet_ds ds/rsna-knee-cnxt-fleet \
      --parent_csv $HOME/kmimic/runs/gold58_gpu/localLB_parent_preds.csv --age_h 7.0 --threads 8 ;;&
  *) ;;
esac
