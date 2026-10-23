#!/bin/bash
# test_fetch_neg.sh (BOX CPU; after mock_e2e.sh; everything under ~/rstud_mock2): negative paths of fetch_loop.sh on a mock
# instance built from the e2e outputs: f0 done but a file corrupted after its sha256 list -> 3 failed verifies ->
# verify_failed (kept aside, placeholder released); f1 done but holding fold-2 checkpoints (wrong fold) -> verify_failed;
# f2 failed and f3 skipped on the instance -> resolved, placeholders released; f4 still running inside the last 40 min
# before the guard deadline -> partial pull to <N>_rentpartial_* (never a student), placeholder released; then
# FINAL_FETCH_OK. Nothing may land in a placeholder.
set -u
M=$HOME/rstud_mock; M2=$HOME/rstud_mock2; rm -rf $M2; mkdir -p $M2/inst/workspace/rsna $M2/fleet_runs
I=$M2/inst/workspace/rsna; cp -a $M/inst/workspace/rsna/kit $I/; mkdir -p $I/status $I/fleet_runs
SRC=$M/inst/workspace/rsna/fleet_runs; N(){ echo f$1_s1047_ttr_r7tb; }
cp -a $SRC/$(N 0) $SRC/$(N 0).log $SRC/$(N 0).receipt.json $SRC/$(N 0).sha256 $I/fleet_runs/
echo "tamper" >> $I/fleet_runs/$(N 0)/oof_f0_s1047_raw.csv; touch $I/status/f0.done
mkdir -p $I/fleet_runs/$(N 1); cp -a $SRC/$(N 2)/. $I/fleet_runs/$(N 1)/; cp $SRC/$(N 2).log $I/fleet_runs/$(N 1).log; cp $SRC/$(N 2).receipt.json $I/fleet_runs/$(N 1).receipt.json
( cd $I/fleet_runs && { find $(N 1) -type f; echo $(N 1).log; echo $(N 1).receipt.json; } | LC_ALL=C sort | xargs -d '\n' sha256sum > $(N 1).sha256 ); touch $I/status/f1.done
echo "rc 1 after 3 min: RuntimeError: mock" > $I/status/f2.failed; echo "estimated 130 min > time left 90 min" > $I/status/f3.skipped
mkdir -p $I/fleet_runs/$(N 4); cp $SRC/$(N 4)/oof_f4_s1047_raw.csv $I/fleet_runs/$(N 4)/; head -n 5 $SRC/$(N 4).log > $I/fleet_runs/$(N 4).log; echo "$(date +%s) 0" > $I/status/f4.running
: > $M2/queue.txt; cp $M/bundle/BUNDLE.json $M2/BUNDLE.json 2>/dev/null; mkdir -p $M2/bundle; cp $M/bundle/BUNDLE.json $M2/bundle/
now=$(date +%s); sed -e "s#^I_ROOT=.*#I_ROOT=$I#" -e "s#^GUARD_DEADLINE=.*#GUARD_DEADLINE=$((now + 1800))#" $M/run.env > $M2/run.env
for F in 0 1 2 3 4; do RSTUD_FR=$M2/fleet_runs RSTUD_Q=$M2/queue.txt bash $HOME/rstud/src/box/ops.sh ph-create $F > /dev/null; done
RSTUD_HOME=$M2 RSTUD_ENV=$M2/run.env BUNDLE=$M2/bundle RSTUD_SSH="bash $HOME/rstud/src/tests/mock_ssh.sh" RSTUD_FR=$M2/fleet_runs RSTUD_Q=$M2/queue.txt \
  RSTUD_POLL_S=2 RSTUD_MAX_POLLS=6 VERIFY_SKIP=decode_probe bash $HOME/rstud/src/box/fetch_loop.sh; echo "fetch_loop rc $?"
grep -E 'RESOLVED|FINAL|ALL RESOLVED|verify/place failed|VERIFY_FAIL' $M2/fetch/fetch.log | sed 's/"ckpts": \[[^]]*\]//' | cut -c1-260
echo "--- mock fleet_runs after the loop:"; ls -la $M2/fleet_runs | tail -n +4 | awk '{print $NF}'
echo "--- instance FINAL_FETCH_OK: $(ls $I/status/FINAL_FETCH_OK 2>/dev/null || echo absent) | kept aside: $(ls -d $M2/fetch/bad_* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
