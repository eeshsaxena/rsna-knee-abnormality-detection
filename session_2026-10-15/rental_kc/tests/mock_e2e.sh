#!/bin/bash
# mock_e2e.sh (rental_kc tests; BOX CPU only, nice 19, cores 30-39; GPU, ~/queue, ~/fleet_runs and ~/kc/gate untouched;
# everything under ~/rkc_mock). End-to-end dry run of the box + instance side WITHOUT renting: pack.sh on a 40-study store
# subset with a mock code dir (box code2 + the KC files + the shimmed trainer text) and the dread TEST table -> driver.sh with
# the transport = tests/mock_ssh.sh and the instance root ~/rkc_mock/inst/workspace/rsna: D2 push + push-rate gate, D3
# setup.sh (manifest, OAI assertion, parity on the real code; pip / GPU steps mocked), D4 parity compare, D5 2-phase smoke
# (fake trainer holds after epoch-0 progress; stop_member kills by verified PID), D6 gate wait (the mock writes the early G1
# file after GATE_AFTER_S), D7 GO (S read FAIL -> s2002d2, argv-check vs the mock queue, code-check, decide.py, placeholders
# in the mock fleet_runs, lanes), D8 fetch + verify_member.py (recipe ref = a fake f0 ckpt) + place + receipts + FINAL_FETCH_OK.
# No em dashes.
set -u
M=$HOME/rkc_mock; rm -rf $M; mkdir -p $M/inst/workspace/rsna $M/fleet_runs $M/gate $M/code2
cp -p $HOME/fast_build/code2/*.py $M/code2/
cp -p $HOME/kc/code/fleet_train_kc.py $HOME/kc/code/kc_common.py $HOME/kc/code/kc_g1.py $M/code2/
cp $HOME/kc/code/shim/ft_shimmed.py.txt $M/code2/fleet_train.py
ls $HOME/kc_store/store | head -n 40 | sed 's/\.npz$//' > $M/subset.txt
grep -o "'[^']*'" $HOME/kc/code/KC_QCTL_1015.txt | grep -F '| ~/fleet_runs/' | tr -d "'" > $M/queue.txt
echo "mock queue: $(wc -l < $M/queue.txt) lines"
export RKC_Q=$M/queue.txt RKC_CODE=$M/code2
RKC_DR=$HOME/kc/smoke/labels_dread_test.csv RKC_DRC=$HOME/kc/smoke/labels_dread_test.csv.receipt.json RKC_SUBSET=$M/subset.txt \
  bash $HOME/rkc/src/box/pack.sh $M/bundle > $M/pack.out 2>&1 || { echo "pack failed"; tail -n 8 $M/pack.out; exit 1; }
tail -n 3 $M/pack.out
F0=$(grep "| ~/fleet_runs/f0_s2001_kc256_r7tb |" $M/queue.txt | sed 's/.*| //' | sed -e "s#--out ~/fleet_runs/#--out $M/fleet_runs/#" -e "s#--gate_out ~/kc/gate/#--gate_out $M/gate/#" -e "s#~/#$HOME/#g")
( cd $M/code2 && FAKE_EP_S=0.1 nice -n 19 $HOME/miniconda3/envs/rsna/bin/python $HOME/rkc/src/tests/fake_trainer.py $F0 > $M/fleet_runs/f0_s2001_kc256_r7tb.log 2>&1 ) || { echo "fake f0 failed"; tail -n 3 $M/fleet_runs/f0_s2001_kc256_r7tb.log; exit 1; }
rm -f $M/gate/kc_f0.verdict $M/gate/kc_f0.verdict.json      # the fake f0 wrote its end-hook verdict; the real f0 is still running at GO
echo "FAIL S not positive (mock)" > $M/gate/s_read.verdict
now=$(date +%s)
cat > $M/run.env <<E
INSTANCE_ID=mock
OFFER=mock
GPU_NAME=mock-cpu
NG=2
DPH=1.1
DIN=0.1
SPENT_BEFORE=0
CAP=7.4
LAUNCH_EPOCH=$now
GUARD_DEADLINE=$((now + 6 * 3600 + 1800))
BOOT_DEADLINE=$((now + 3 * 3600))
GATE_LATEST=$((now + 2 * 3600))
I_HOST=mock
I_PORT=0
I_ROOT=$M/inst/workspace/rsna
PYI=$HOME/rkc/src/tests/fake_py.sh
SMOKE1_MIN=1
SMOKE2_MIN=1
CANDIDATES=""
E
( sleep ${GATE_AFTER_S:-200}; echo "EARLY_G1 ablation_pass=True epoch=4 | mock" > $M/gate/kc_f0_ep3_ablation.txt ) > /dev/null 2>&1 < /dev/null &
export RKC_HOME=$M RKC_ENV=$M/run.env BUNDLE=$M/bundle RKC_SSH="bash $HOME/rkc/src/tests/mock_ssh.sh" RKC_FR=$M/fleet_runs \
       RKC_POLL_S=20 KC_GATE_DIR=$M/gate GATE_POLL_S=5 PUSH_POLL_S=2 PUSH_GATE_S=1 PUSH_MIN_MBS=0 FAKE_EP_S=1 FAKE_HOLD_SMOKE=1
export MOCK_ENV='SETUP_MOCK=1 MOCK_CPU=1 MOCK_CPUSETS="38 39" OMP_T=1 PARITY_PRE="nice -n 19 taskset -c 30-39"'
setsid -f bash $HOME/rkc/src/box/driver.sh > $M/driver.out 2>&1 < /dev/null
echo "mock e2e started $(date '+%T'): logs $M/driver.log, $M/fetch/fetch.log"
