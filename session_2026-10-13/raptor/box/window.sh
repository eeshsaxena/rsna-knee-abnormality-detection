#!/bin/bash
# window.sh (box): ONE GPU window at the next queue job boundary for the Raptor seed. Never kills or signals anything.
#  1 wait until the queue runner logs DONE / FAILED for the RUNNING job ($JOB), then touch ~/queue/PAUSE inside the runner's
#    120 s free-check gap (PAUSE only blocks NEW launches);
#  2 abort (and remove PAUSE) if the runner launched anything meanwhile or the GPU does not become idle;
#  3 smoke (timeout 30 min) -> verdict;
#  4 full run ONLY if: SMOKE_GO, ~/raptor_store/CORPUS_OK present, PREREG sha matches, ~/raptor_store/FULL_APPROVED present;
#    epochs + channels_last from the verdict; foreground; resume-capable (last.pt);
#  5 ALWAYS remove ~/queue/PAUSE at the end (run finished, aborted or skipped), so the queue resumes.
set -u
JOB=${JOB:-f0_s1044_ttr_frac75oaiT123v00}
D=$HOME/raptor_store/smoke; mkdir -p $D; L=$D/window.log; QR=$HOME/fleet_runs/queue_runner.log
PY=$HOME/miniconda3/envs/rsna/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L; }
cleanup(){ rm -f $HOME/queue/PAUSE; say "removed ~/queue/PAUSE (exit)"; }
n0=$(wc -l < $QR)
say "armed (pid $$): waiting for DONE/FAILED of $JOB after queue_runner.log line $n0"
while :; do
  tail -n +$((n0 + 1)) $QR | grep -F "/fleet_runs/$JOB |" | grep -qE "\] (DONE|FAILED) " && break
  sleep 5
done
touch $HOME/queue/PAUSE; trap cleanup EXIT; say "job ended -> touched ~/queue/PAUSE"
sleep 30
if tail -n +$((n0 + 1)) $QR | grep -q "\] LAUNCH "; then say "runner launched a job before PAUSE took effect: ABORT window"; exit 1; fi
apps=1
for i in $(seq 1 30); do apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$apps" = 0 ] && break; sleep 10; done
[ "$apps" = 0 ] || { say "GPU still has $apps compute app(s) after 5 min: ABORT window"; exit 1; }
say "GPU idle, runner paused -> smoke start"
timeout 1800 bash $HOME/raptor2/smoke_run.sh > $D/smoke.out 2>&1; rc=$?
say "smoke rc $rc | $(tail -n 2 $D/verdict.txt 2>/dev/null | tr '\n' ' ')"
grep -qx SMOKE_GO $D/verdict.txt 2>/dev/null || { say "no SMOKE_GO: no full run"; exit 0; }
[ -f $HOME/raptor_store/CORPUS_OK ] || { say "CORPUS_OK missing: smoke only, full run at a later boundary"; exit 0; }
[ -f $HOME/raptor_store/FULL_APPROVED ] || { say "FULL_APPROVED missing: smoke only"; exit 0; }
( cd $HOME/raptor2 && sha256sum -c PREREG_RAPTOR_SEED.sha256 >/dev/null 2>&1 ) || { say "PREREG sha mismatch: no full run"; exit 0; }
EP=$(grep -o 'EPOCHS [0-9]*' $D/verdict.txt | awk '{print $2}'); CL=$(grep -o 'CHANNELS_LAST [01]' $D/verdict.txt | awk '{print $2}')
R=$HOME/raptor_store/run_s42; mkdir -p $R
say "FULL RUN launch: epochs $EP channels_last $CL -> $R"
cd $HOME/raptor2 && OMP_NUM_THREADS=2 nice -n 19 taskset -c 8-15 $PY train_raptor.py \
  --index $HOME/raptor_store/rsna_index.csv $HOME/raptor_store/oai_index.csv \
  --labels_rsna $HOME/raptor2/labels_raptor_rsna_blendjev_sol.csv --labels_oai $HOME/oai_build/oai_labels_raptorT1.csv \
  --out $R --epochs $EP --channels_last $CL --workers 6 > $R/run.out 2>&1
rc=$?
say "FULL RUN ended rc $rc | $(grep -E '^=== (ep|SWA)' $R/train.log | tail -n 2 | tr '\n' ' ')"
exit 0
