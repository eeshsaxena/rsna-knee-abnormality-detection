#!/bin/bash
# window2.sh (box): SAME queue gap as window.sh. The V100 smoke measured peak memory 15.1 GiB (> the 14.5 GB smoke bar; ~0.7 GiB
# headroom on the 16 GB card, too tight for a 12-20 h run). If the smoke failed ONLY on memory, re-take PAUSE the second
# window.sh exits (the runner needs 2 free checks 120 s apart, so it cannot launch in between), run a short smoke C with
# --grad_ckpt 1, and if C passes the SAME smoke bars (no NaN, s/step <= .85, mem <= 14.5 GB, swa + serve-load OK) launch the full
# run with --grad_ckpt 1 (16 ep if the projection <= 26 h, else 12). ALWAYS removes ~/queue/PAUSE on exit. Never kills anything.
set -u
D=$HOME/raptor_store/smoke; L=$D/window.log; QR=$HOME/fleet_runs/queue_runner.log; PY=$HOME/miniconda3/envs/rsna/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] w2 $*" >> $L; }
cleanup(){ rm -f $HOME/queue/PAUSE; say "removed ~/queue/PAUSE (exit)"; }
W1=${W1:-550546}
say "armed (pid $$): waiting for window.sh pid $W1 to exit"
while kill -0 $W1 2>/dev/null; do sleep 1; done
touch $HOME/queue/PAUSE; trap cleanup EXIT; n0=$(wc -l < $QR); say "window.sh exited -> re-touched ~/queue/PAUSE"
ONLYMEM=$($PY - <<'PYV'
import json, re, os
t = open(os.path.expanduser('~/raptor_store/smoke/verdict.txt')).read()
a = json.loads(re.search(r'^A channels_last=1: (.*)$', t, re.M).group(1))
srv = 'SERVE ' in t
ok = (a['nonfinite'] == 0 and not a['abort'] and not a['tb'] and a['swa'] and a['sps_med'] is not None and a['sps_med'] <= .85 and srv
      and (a['skipped'] or 0) <= 2 and a['mem'] is not None and a['mem'] > 14.5 and 'SMOKE_NOGO' in t)
print('yes' if ok else 'no')
PYV
)
[ "$ONLYMEM" = yes ] || { say "smoke did not fail on memory alone (or passed): window2 does nothing"; exit 0; }
sleep 5
if tail -n +$((n0 + 1)) $QR | grep -q "\] LAUNCH "; then say "runner launched a job: ABORT"; exit 1; fi
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$apps" = 0 ] || { say "GPU busy: ABORT"; exit 1; }
CL=$(grep -o 'CHANNELS_LAST [01]' $D/verdict.txt | awk '{print $2}'); CL=${CL:-1}
say "smoke C start: --grad_ckpt 1 --channels_last $CL, 400 study-steps"
[ -e $D/C_gc1 ] && mv $D/C_gc1 $D/C_gc1_old_$(date +%m%d_%H%M%S)
cd $HOME/raptor2 && OMP_NUM_THREADS=2 timeout 900 nice -n 19 taskset -c 8-15 $PY train_raptor.py \
  --index $HOME/raptor_store/rsna_index.csv $HOME/raptor_store/oai_index.csv \
  --labels_rsna $HOME/raptor2/labels_raptor_rsna_blendjev_sol.csv --labels_oai $HOME/oai_build/oai_labels_raptorT1.csv \
  --out $D/C_gc1 --smoke_steps 400 --channels_last $CL --grad_ckpt 1 --workers 6 > $D/C_gc1.out 2>&1
rc=$?
V=$($PY - <<'PYC'
import re, os, statistics
t = open(os.path.expanduser('~/raptor_store/smoke/C_gc1/train.log')).read() if os.path.exists(os.path.expanduser('~/raptor_store/smoke/C_gc1/train.log')) else ''
sps = [float(x) for x in re.findall(r' ([0-9.]+)s/step', t)]; mem = [float(x) for x in re.findall(r'peak mem ([0-9.]+) GB', t)]
nf = [int(x) for x in re.findall(r'nonfinite_loss (\d+)', t)]; ev = re.findall(r'eval (\d+)s\)', t)
s = statistics.median(sps[1:]) if len(sps) > 1 else None
ok = s is not None and s <= .85 and mem and max(mem) <= 14.5 and nf and max(nf) == 0 and 'SERVE-LOAD OK' in t and 'ABORT' not in t
proj = (s * 6748 * 16 + 16 * int(ev[0] if ev else 60)) / 3600 if s else 99
print(f"{'GO' if ok else 'NOGO'} {s} {max(mem) if mem else None} {round(proj, 1)}")
PYC
)
say "smoke C rc $rc -> $V (verdict s/step peak_mem_GB projection_16ep_h)"
set -- $V; [ "$1" = GO ] || { say "smoke C NOGO: no full run"; exit 0; }
EP=16; "$PY" -c "import sys; sys.exit(0 if float('$4') <= 26 else 1)" || EP=12
[ -f $HOME/raptor_store/CORPUS_OK ] && [ -f $HOME/raptor_store/FULL_APPROVED ] || { say "CORPUS_OK / FULL_APPROVED missing"; exit 0; }
( cd $HOME/raptor2 && sha256sum -c PREREG_RAPTOR_SEED.sha256 > /dev/null 2>&1 ) || { say "PREREG sha mismatch"; exit 0; }
R=$HOME/raptor_store/run_s42; mkdir -p $R
say "FULL RUN launch: epochs $EP channels_last $CL grad_ckpt 1 (smoke C s/step $2, mem $3 GB, projection 16 ep $4 h) -> $R"
cd $HOME/raptor2 && OMP_NUM_THREADS=2 nice -n 19 taskset -c 8-15 $PY train_raptor.py \
  --index $HOME/raptor_store/rsna_index.csv $HOME/raptor_store/oai_index.csv \
  --labels_rsna $HOME/raptor2/labels_raptor_rsna_blendjev_sol.csv --labels_oai $HOME/oai_build/oai_labels_raptorT1.csv \
  --out $R --epochs $EP --channels_last $CL --grad_ckpt 1 --workers 6 > $R/run.out 2>&1
rc=$?
say "FULL RUN ended rc $rc | $(grep -E '^=== (ep|SWA)' $R/train.log | tail -n 2 | tr '\n' ' ')"
exit 0
