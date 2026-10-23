#!/bin/bash
# run_e2e_all.sh (10-15, Claude subagent, kcstage): box CPU end-to-end of the BUILT stand-in kernel's 3 KC cells (e2e_cells.py) in the
# three cases + the box parity tool on the clean output. Cores 28-39, nice 19, ionice c3, CUDA hidden; RSNA train studies only.
#   bash run_e2e_all.sh NOTEBOOK TAG
set -u
NB=$1; TAG=$2
K=~/kcstage_1015
PY=~/miniconda3/envs/kmimic/bin/python
E="taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3"
mkdir -p $K/e2e_$TAG
for c in guard clean s6fcanary; do
  rm -rf $K/e2e_$TAG/$c
  age=6.0; [ $c = guard ] && age=7.2
  $E $PY -u $K/tools/e2e_cells.py --nb $NB --ds $K/ds_kc_standin --root $K/fakeroot_n5 --work $K/e2e_$TAG/$c --case $c --age-h $age --threads 5 \
     > $K/e2e_$TAG/$c.log 2>&1
  tail -n 1 $K/e2e_$TAG/$c.log
done
$E $PY -u $K/tools/kc_box_parity.py --out-dir $K/e2e_$TAG/clean/working --code $K/code \
   --members $K/standin/kc_standin_m1_s11_raw_fp32.pt $K/standin/kc_standin_m2_s12_raw_fp16.pt \
   --manifest $K/ds_kc_standin/KC_MANIFEST.json --json $K/e2e_$TAG/clean_box_parity.json --threads 10 > $K/e2e_$TAG/parity.log 2>&1
tail -n 1 $K/e2e_$TAG/parity.log
echo E2E_ALL_DONE
