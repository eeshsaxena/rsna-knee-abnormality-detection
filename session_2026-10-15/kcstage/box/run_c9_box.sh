#!/bin/bash
# run_c9_box.sh (10-15, Claude subagent, kcstage; KCS2): the two n 20 C9 runs on the SERVED path, box CPU, sequential, light
# (cores 28-39, nice 19, ionice c3, CUDA hidden, THREADS threads). (1) the 2 stand-in members against the stand-in KC_MANIFEST
# (the JSON the mock gate uses); (2) a dry run of the real-member path on f0 s2001's first saved RAW ckpt (ep3, BOX_PLAN C9).
cd ~ || exit 1
T=${THREADS:-4}
C="taskset -c 28-39 nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T"
PY=~/miniconda3/envs/kmimic/bin/python
COMMON="--code kcstage_1015/code --serve kcstage_1015/kc_serve.py --kpp kc/code/kc_prob_parity.py --store kc_store/store --dicom eda_data --n 20 --threads $T"
mkdir -p kcstage_1015/c9
$C $PY -u kcstage_1015/tools/c9_served.py $COMMON \
  --members kcstage_1015/standin/kc_standin_m1_s11_raw_fp32.pt kcstage_1015/standin/kc_standin_m2_s12_raw_fp16.pt \
  --manifest kcstage_1015/ds_kc_standin/KC_MANIFEST.json --out kcstage_1015/c9/c9_served_standin_n20.json > kcstage_1015/c9/c9_served_standin_n20.txt 2>&1
$C $PY -u kcstage_1015/tools/c9_served.py $COMMON \
  --members fleet_runs/f0_s2001_kc256_r7tb/epochs/f0_s2001_raw_kcttr256-256x12_ep3_best.pt \
  --out kcstage_1015/c9/c9_served_f0ep3_n20.json > kcstage_1015/c9/c9_served_f0ep3_n20.txt 2>&1
echo RUN_C9_BOX_DONE > kcstage_1015/c9/run_c9_box.done
