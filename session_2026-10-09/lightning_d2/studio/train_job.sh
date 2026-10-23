#!/bin/bash
# train_job.sh MODE F CAP_S PRE: runs INSIDE the Lightning Job (machine started from the Studio rsna-d2-nb snapshot).
#   MODE dry    facts + input md5 re-check + artifact probe, no training (CPU test job)
#   MODE train  the exact D2 argv (meta/argv_template.txt = the Vast kit's, argv-check vs the box queue line
#               "Q2 frac75 f3": only --seed 1044 and --out differ), trainer under `timeout CAP_S`.
#   PRE         credits already spent before this job (for the in-job cost guard)
# Lightning job facts (dry job rsna-d2-dry1, 10-09): /teamspace/jobs/<job> is read-only; the job ARTIFACTS are the
# files created or changed under /teamspace/studios/this_studio, uploaded when the command ends. So the trainer writes
# to a LOCAL root ($LR: symlinks to the snapshot inputs + a real local fleet_runs), and only the fetch set (log,
# *_best.pt, oof_*.csv, epochs/oof_*.csv; never epochs/*.pt) is copied to $ART = this_studio/ld2_out every 5 min and
# once more after the trainer exits; then an md5 manifest is written and the script exits, so the job (and its
# machine) ends right after the closing line. Cost guard (every 60 s): projected total = PRE + RATE x (SETUP_H +
# elapsed + remaining + 0.05 h); if it exceeds CAP the trainer is stopped (TERM), outputs are copied and kept.
set -u
MODE=$1; F=$2; CAP_S=${3:-6000}; PRE=${4:-0.40}
RATE=${LD2_RATE:-2.14}; CAP=${LD2_CAP:-4.3}; SETUP_H=${LD2_SETUP_H:-0.13}
H=/teamspace/studios/this_studio
R=$H/rsna; M=$H/rsna_d2/meta; P=$H/rsna_d2/venv/bin/python
N=f${F}_s1044_ttr_frac75
ART=$H/ld2_out; LR=/tmp/ldR
CLOSE_RE='^\{"raw": [^,]+, "ema": [^}]+\}$'
say(){ echo "[ld2job $(date '+%F %T')] $*"; }
say "MODE $MODE fold $F cap ${CAP_S}s pre $PRE rate $RATE credit cap $CAP | host $(hostname) | ART $ART | local root $LR"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv,noheader 2>&1 | head -2 | sed 's/^/GPU /'
echo "NPROC $(nproc) RAM_GB $(free -g | awk '/Mem/{print $2}') SHM $(df -h /dev/shm | tail -n 1 | awk '{print $4}') LOCALFREE $(df -h /tmp | tail -n 1 | awk '{print $4}')"
[ -f $H/rsna_d2/PARITY_OK ] || { say "FAIL: no PARITY_OK in the snapshot"; exit 3; }
t0=$(date +%s)
rm -rf /tmp/ld2_md5; mkdir -p /tmp/ld2_md5; split -n l/8 -d $M/store_md5.txt /tmp/ld2_md5/c_
ls /tmp/ld2_md5/c_* | R=$R xargs -P 8 -I{} sh -c 'cd $R/fleet_store && md5sum -c --quiet {}' || { say "FAIL: store md5 in the job"; exit 4; }
(cd $R && sed 's#  code2/#  fast_build/code2/#' $M/small_md5.txt | md5sum -c --quiet -) || { say "FAIL: small md5 in the job"; exit 4; }
say "inputs md5 IDENTICAL to the box: 4407 store + $(wc -l < $M/small_md5.txt) code/csv/HF files ($(( $(date +%s) - t0 ))s)"
$P -c 'import torch, sys; print("TORCH", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda, "py", sys.version.split()[0], torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
mkdir -p $ART
if [ "$MODE" = dry ]; then echo "dry $(date '+%F %T')" > $ART/dry_probe.txt; say "DRY DONE"; exit 0; fi

# ---- train ----
rm -rf $LR; mkdir -p $LR/fleet_runs
for d in fast_build novel rsna_train hf fleet_store; do ln -s $R/$d $LR/$d; done
ARGV=$(sed -e "s#@R@#$LR#g" -e "s#@F@#$F#g" $M/argv_template.txt)
$P - "$F" $ARGV <<'EOF' || { say "FAIL: argv check"; exit 5; }
import sys
F, a = sys.argv[1], sys.argv[2:]
g = dict(zip(a[::2], a[1::2]))
assert len(a) == 38, len(a)
assert g["--seed"] == "1044" and g["--fold"] == F and g["--epochs"] == "21" and g["--save_epochs"] == "6-21", g
assert g["--model"] == "ttr" and g["--grad_ckpt"] == "0" and g["--workers"] == "10" and g["--train_slices"] == "16", g
assert g["--out"].endswith("/fleet_runs/f%s_s1044_ttr_frac75" % F), g
assert g["--labels"].endswith("/novel/frac75/labels_frac75_logit.csv") and g["--eval_labels"].endswith("/rsna_train/llm_labels_v4_blend.csv"), g
print("ARGV_OK 38 tokens: seed 1044 fold %s epochs 21 save 6-21 ttr (224 default) gc0 workers 10" % F)
EOF
OUTD=$LR/fleet_runs/$N; LOG=$LR/fleet_runs/$N.log; A=$ART/fleet_runs; mkdir -p $A/$N/epochs
sync_out(){   # $1 = full (plain cp) or upd (cp -u)
  local o=-u; [ "${1:-upd}" = full ] && o=""
  [ -f $LOG ] && cp $o $LOG $A/
  for f in $OUTD/*_best.pt $OUTD/oof_*.csv; do [ -f "$f" ] && cp $o "$f" $A/$N/; done
  for f in $OUTD/epochs/oof_*.csv; do [ -f "$f" ] && cp $o "$f" $A/$N/epochs/; done
  for f in $A/$N/*_best.pt; do [ -e "$f" ] && [ ! -e "$OUTD/$(basename "$f")" ] && rm -f "$f"; done
  return 0
}
guard(){   # prints PROJ <total> <remaining_h> <basis>; exit 0
  $P - "$LOG" "$T0" "$PRE" "$RATE" "$SETUP_H" <<'EOF'
import sys, re, time
log, t0, pre, rate, setup = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
try: L = open(log, errors="replace").read().splitlines()
except Exception: L = []
el = (time.time() - t0) / 3600
ema = [float(m.group(2)) for m in (re.match(r"ep (\d+) ema \|.* epoch ([0-9.]+) min", l) for l in L) if m]
its = [(int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))) for m in (re.match(r"ep (\d+) it (\d+)/(\d+) loss \S+ (\S+)s/it", l) for l in L) if m]
if ema:
    ep_h = sum(ema[-3:]) / len(ema[-3:]) / 60; done = len(ema); rem = (21 - done) * ep_h; basis = "epochs %d, %.1f min/ep" % (done, ep_h * 60)
elif len(its) >= 3 and its[-1][1] >= 400 and its[-2][0] == its[-1][0]:
    # the trainer prints a cumulative mean s/it within the epoch (it 0 includes ~35 s of startup), so use the interval
    # rate between the last two lines: (s2 * i2 - s1 * i1) / (i2 - i1)
    (e1, i1, n, s1), (e2, i2, _, s2) = its[-2], its[-1]
    s = (s2 * i2 - s1 * i1) / max(i2 - i1, 1); ep_h = n * s * 1.27 / 3600; rem = max(0.0, 21 * ep_h - el)
    basis = "ep0 it %d interval %.3f s/it (est)" % (i2, s)
else:
    print("PROJ NA NA warmup"); sys.exit(0)
tot = pre + rate * (setup + el + rem + 0.05)
print("PROJ %.3f %.3f %s" % (tot, rem, basis))
EOF
}
say "LAUNCH $N: $P fleet_train.py $ARGV"
T0=$(date +%s)
( cd $LR/fast_build/code2 && exec env HF_HOME=$LR/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
    timeout --signal=TERM --kill-after=60 $CAP_S $P fleet_train.py $ARGV ) > $LOG 2>&1 < /dev/null &
TP=$!
tail -n +1 -F $LOG --pid=$TP 2>/dev/null &
( while kill -0 $TP 2>/dev/null; do sleep 300; sync_out upd; done ) &
SP=$!
ABORT=""
while kill -0 $TP 2>/dev/null; do
  sleep 60
  g=$(guard 2>/dev/null); set -- $g
  if [ "${1:-}" = PROJ ] && [ "${2:-NA}" != NA ]; then
    say "COST_GUARD projected total ${2} credits (cap $CAP) | remaining ${3} h | ${4:-} ${5:-} ${6:-} ${7:-}"
    # abort on a completed-epoch projection > CAP, or on the ep-0 estimate only if it is > CAP x 1.10
    if $P -c "import sys; t, c, est = float('$2'), float('$CAP'), '${4:-}' == 'ep0'; sys.exit(0 if t > c * (1.10 if est else 1.0) else 1)"; then
      ABORT="projected $2 > cap $CAP"; say "ABORT: $ABORT: stopping the trainer, keeping its outputs"
      pkill -TERM -f "fleet_train[.]py .*--out $OUTD( |\$)"; sleep 30; pkill -KILL -f "fleet_train[.]py .*--out $OUTD( |\$)"; break
    fi
  fi
done
wait $TP; rc=$?
DT=$(( $(date +%s) - T0 ))
sleep 2; kill $SP 2>/dev/null
CL=$(tail -n 1 $LOG | tr -d '\r')
say "trainer exit rc $rc after $(( DT / 60 )) min | last line: $CL"
sync_out full
(cd $A && { find $N -maxdepth 1 -type f \( -name '*_best.pt' -o -name 'oof_*.csv' \); find $N/epochs -maxdepth 1 -type f -name 'oof_*.csv'; echo $N.log; } | LC_ALL=C sort | xargs md5sum) > $A/$N.manifest.txt
(cd $LR/fleet_runs && md5sum -c --quiet $A/$N.manifest.txt) && say "artifact copies md5 == local trainer outputs ($(wc -l < $A/$N.manifest.txt) files)" || say "WARN: artifact copy md5 differs from local"
du -sh $OUTD $A/$N 2>/dev/null | sed 's/^/DU /'
if [ -z "$ABORT" ] && [ $rc -eq 0 ] && echo "$CL" | grep -Eq "$CLOSE_RE"; then
  echo "rc=0 train_s=$DT closing=$CL" > $A/$N.DONE; say "DONE $N: $CL"; exit 0
fi
echo "rc=$rc train_s=$DT abort=${ABORT:-none} last=$CL" > $A/$N.FAILED; say "FAILED/ABORTED $N rc $rc ${ABORT}"; exit 1
