#!/bin/bash
# window3.sh (box): after the S42 full run, keep the SAME queue gap for ~3 min of GPU: serving-geometry gold58 of S (swa.pt) and N
# (serve_check.py, V100 fp16, 94 windows, mirror) and the PREREG_RAPTOR_SEED gate (gate_raptor.py, CPU). Re-takes ~/queue/PAUSE the
# second window2.sh exits (the runner needs 2 free checks 120 s apart), ALWAYS removes it on exit. Never kills anything.
set -u
D=$HOME/raptor_store/smoke; L=$D/window.log; QR=$HOME/fleet_runs/queue_runner.log; PY=$HOME/miniconda3/envs/rsna/bin/python
RUN=$HOME/raptor_store/run_s42
say(){ echo "[$(date '+%m-%d %H:%M:%S')] w3 $*" >> $L; }
cleanup(){ rm -f $HOME/queue/PAUSE; say "removed ~/queue/PAUSE (exit)"; }
W2=${W2:-656448}
say "armed (pid $$): waiting for window2.sh pid $W2 to exit"
while kill -0 $W2 2>/dev/null; do sleep 1; done
touch $HOME/queue/PAUSE; trap cleanup EXIT; n0=$(wc -l < $QR); say "window2.sh exited -> re-touched ~/queue/PAUSE"
[ -f $RUN/swa.pt ] && grep -q 'SERVE-LOAD OK' $RUN/train.log || { say "no finished swa.pt in $RUN: nothing to score"; exit 0; }
sleep 5
if tail -n +$((n0 + 1)) $QR | grep -q "\] LAUNCH "; then say "runner launched a job: ABORT"; exit 1; fi
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$apps" = 0 ] || { say "GPU busy: ABORT"; exit 1; }
cd $HOME/raptor2 && timeout 900 nice -n 19 taskset -c 8-15 $PY serve_check.py --index $HOME/raptor_store/rsna_index.csv \
  --ck $RUN/swa.pt $HOME/kfn70_test/wts/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt --out $RUN/serve > $RUN/serve.log 2>&1
say "serve rc $? | $(grep SERVE $RUN/serve.log | cut -c1-90 | tr '\n' ' ')"
EP=$(grep -o '"epochs": [0-9]*' $RUN/train.log | head -n 1 | awk '{print $2}')
CUDA_VISIBLE_DEVICES= nice -n 19 taskset -c 8-15 $PY gate_raptor.py --run $RUN --s_pred $RUN/serve/run_s42__swa.pt.npy \
  --n_pred $RUN/serve/wts__raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt.npy --planned_epochs ${EP:-16} --out $HOME/raptor2/GATE_RESULT.txt > $RUN/gate.log 2>&1
say "gate: $(grep RAPTOR_SEED_GATE $HOME/raptor2/GATE_RESULT.txt)"
exit 0
