#!/bin/bash
# mock_e2e_4fold.sh (10-14 20:10: copy of mock_e2e.sh for the 2-GPU, 4-fold plan A: NG=2, FOLDS 0-3, 2 trainers per mock GPU) (BOX CPU only, nice 19, cores 28-39; GPU, ~/queue and ~/fleet_runs untouched; everything under ~/rstud_mock).
# End-to-end dry run of the box side + instance side WITHOUT renting: pack.sh DRY on a 31-study subset -> driver.sh with
# the transport replaced by tests/mock_ssh.sh and the instance root = ~/rstud_mock/inst/workspace/rsna: D2 push (tar
# streams), D3 setup.sh (manifest, OAI assertion, parity; pip/GPU steps mocked), D4 parity compare, D5 placeholders (mock
# fleet_runs), D6 lanes in the 4-GPU smoke layout (4 "GPUs" = 4 x 3 CPU cores; f4 paired with f0), D7 decide.py (seq
# expected on CPU: pairing halves the speed) incl. stop_fold + any-lane, RUN_STARTED, D8 fetch loop: pull + verify (real
# trainer outputs, ckpt args vs the box reference run) + place + receipts + FINAL_FETCH_OK.
# Trainer overrides (mock only, appended): --device cpu --min_cover 0 --time_epochs 1. RUN_EPOCHS=1 for decide.py.
set -u
M=$HOME/rstud_mock; rm -rf $M; mkdir -p $M/inst/workspace/rsna $M/fleet_runs
ls $HOME/novel/frac75/dry/store | sed 's/\.npz$//' > $M/subset.txt
SUBSET=$M/subset.txt bash $HOME/rstud/src/box/pack.sh DRY $M/bundle > $M/pack.out 2>&1 || { echo "pack failed"; tail -n 5 $M/pack.out; exit 1; }
: > $M/queue.txt
now=$(date +%s)
cat > $M/run.env <<E
INSTANCE_ID=mock
OFFER=mock
GPU_NAME=mock-cpu
NG=2
DPH=2.0
DIN=0.25
SPENT_BEFORE=0
CAP=15.0
LAUNCH_EPOCH=$now
GUARD_DEADLINE=$((now + 4 * 3600))
BOOT_DEADLINE=$((now + 6000))
LANE_DEADLINE=$((now + 3 * 3600))
FOLD_EST_S=900
I_HOST=mock
I_PORT=0
I_ROOT=$M/inst/workspace/rsna
FOLDS="0 1 2 3"
SMOKE_MIN=3
SMOKE_MAX=25
PYI=$HOME/miniconda3/envs/rsna/bin/python
E
export RSTUD_HOME=$M RSTUD_ENV=$M/run.env BUNDLE=$M/bundle RSTUD_SSH="bash $HOME/rstud/src/tests/mock_ssh.sh" RSTUD_FR=$M/fleet_runs \
       RSTUD_Q=$M/queue.txt RSTUD_ALLOW_DRY=1 RSTUD_POLL_S=60 SMOKE_POLL_S=30 RUN_EPOCHS=1 DECIDE_MIN_IT=0 VERIFY_SKIP=decode_probe,amp FETCH_H=0.05
export MOCK_ENV='SETUP_MOCK=1 MOCK_CPU=1 MOCK_CPUSETS="28-33 34-39" EXTRA_ARGS="--device cpu --min_cover 0 --time_epochs 1" OMP_T=3 LANE_ANY_DELAY_S=30 LANE_ANY_POLL_S=20 PARITY_PRE="nice -n 19 taskset -c 28-39"'
setsid -f bash $HOME/rstud/src/box/driver.sh > $M/driver.out 2>&1 < /dev/null
echo "mock e2e started $(date '+%T'): logs $M/driver.log, $M/fetch/fetch.log"
