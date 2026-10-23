#!/bin/bash
# run_final_box.sh (10-15, Claude subagent, kcstage): every box-side test on the FINAL built stand-in kernels (box CPU, cores 28-39,
# nice 19, ionice c3, CUDA hidden; RSNA train studies only): e2e cells (S6F and S6F-J forms: guard / clean / s6fcanary + box parity),
# the skip-control kernel's guard case, the Linux process-group kill of the RUN cell, the driver's stop-time kill.
set -u
K=~/kcstage_1015
PY=~/miniconda3/envs/kmimic/bin/python
E="taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1"
sha256sum $K/kernels/*.ipynb $K/kc_driver.py $K/kc_serve.py
bash $K/tools/run_e2e_all.sh $K/kernels/rsna_sub_v224s6fkcstandin.ipynb s6f
bash $K/tools/run_e2e_all.sh $K/kernels/rsna_sub_v224s6fkcstandinj.ipynb s6fj
rm -rf $K/e2e_ctl && mkdir -p $K/e2e_ctl
$E $PY -u $K/tools/e2e_cells.py --nb $K/kernels/rsna_sub_v224s6fkcstandinskip.ipynb --ds $K/ds_kc_standin --root $K/fakeroot_n5 \
   --work $K/e2e_ctl/guard --case guard --age-h 6.0 > $K/e2e_ctl/guard.log 2>&1; tail -n 1 $K/e2e_ctl/guard.log
$E $PY -u $K/tools/test_kill_linux.py --nb $K/kernels/rsna_sub_v224s6fkcstandin.ipynb > $K/meas/test_kill_linux.txt 2>&1; tail -n 1 $K/meas/test_kill_linux.txt
$E OMP_NUM_THREADS=2 $PY -u $K/tools/test_driver_stop.py --driver $K/kc_driver.py --serve $K/kc_serve.py --code $K/code --root $K/fakeroot_n5 \
   --members $K/standin/kc_standin_m1_s11_raw_fp32.pt $K/standin/kc_standin_m2_s12_raw_fp16.pt > $K/meas/test_driver_stop.txt 2>&1
tail -n 1 $K/meas/test_driver_stop.txt
echo FINAL_BOX_DONE
