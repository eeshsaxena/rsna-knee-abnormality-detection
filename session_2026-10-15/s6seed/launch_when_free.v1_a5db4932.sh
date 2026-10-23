#!/bin/bash
# launch_when_free.sh (10-15, BOX_PLAN_1015 sec 3.5 PAUSE protocol + Amendment 7 pre-GPU gates) for the external S6 member s43.
# Written by the S6 prep agent; ARMED BY MAIN ONLY (it creates and removes ~/queue/PAUSE and starts a GPU job):
#   setsid nohup bash ~/s6seed/launch_when_free.sh --px <px320 dir> [--pause-after <path> | --pause-at 'YYYY-mm-dd HH:MM'] \
#       > ~/s6seed/logs/launch_when_free.log 2>&1 < /dev/null &
# Sequence:
#   0. pre-flight (no PAUSE yet, CPU only): kit shas (ct_s6v100.py, init, targets, px index), gate files written by main
#      (gates/PX_PARITY.verdict and gates/A7.verdict must start with PASS), ct_s6v100 --check-targets on CPU (PASS, pool size)
#   1. wait for the trigger (--pause-after: a file appears, e.g. the job ahead's last-epoch checkpoint; --pause-at: a time;
#      neither: now), then touch ~/queue/PAUSE (PAUSE holds NEW queue launches only; a PAUSE that already existed is left alone)
#   2. wait for 2 consecutive free-GPU checks 120 s apart (no compute app in nvidia-smi, no fleet_train.py process)
#   3. 300-step smoke (fp16 + GradScaler, all-data, separate out dir): gates rc 0, finite loss on every step, s/step median <= 1.8,
#      peak reserved memory < 15.5 GB, scaler skipped <= 25 of 300
#   4. launch the real run (setsid nohup, its own log), wait until its process holds the GPU, then rm ~/queue/PAUSE
#   Any failure after step 1: rm the PAUSE we created (the queue resumes), write gates/LAUNCH.verdict FAIL <reason>, exit 1.
#   An EXIT trap does the same if the script dies, so PAUSE is never left behind without a job on the GPU.
# Test hooks (tests/test_launch_when_free.sh): S6_QUEUE_DIR, S6_NVSMI, S6_FTCOUNT_CMD, S6_FREE_INTERVAL, S6_DRYRUN=1 (stub smoke / stub run; then the
# queue dir must NOT be ~/queue).  No em dashes.
set -u
KIT="${S6_KIT:-$HOME/s6seed}"
Q="${S6_QUEUE_DIR:-$HOME/queue}"
NVSMI="${S6_NVSMI:-nvidia-smi}"
PY="${S6_PY:-$HOME/miniconda3/envs/rsna/bin/python}"
FREE_INTERVAL="${S6_FREE_INTERVAL:-120}"
APPEAR_TIMEOUT="${S6_APPEAR_TIMEOUT:-900}"
DRY="${S6_DRYRUN:-0}"
PX=""; PAUSE_AFTER=""; PAUSE_AT=""; RUN_TAG="s6s43"; MAX_SPS="1.8"; MAX_MEM="15.5"; MAX_SKIP="25"
while [ $# -gt 0 ]; do
  case "$1" in
    --px) PX="$2"; shift 2;;
    --pause-after) PAUSE_AFTER="$2"; shift 2;;
    --pause-at) PAUSE_AT="$2"; shift 2;;
    --tag) RUN_TAG="$2"; shift 2;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
GATES="$KIT/gates"; mkdir -p "$GATES" "$KIT/logs" "$KIT/runs"
VERDICT="$GATES/LAUNCH_${RUN_TAG}.verdict"
log() { echo "$(date '+%F %T') $*"; }
CREATED_PAUSE=0; LAUNCHED=0
fail() { log "FAIL $*"; echo "FAIL $(date '+%F %T') $*" > "$VERDICT"; cleanup_pause; exit 1; }
cleanup_pause() {
  if [ "$CREATED_PAUSE" = "1" ] && [ "$LAUNCHED" = "0" ]; then rm -f "$Q/PAUSE" && log "removed $Q/PAUSE (no S6 job launched)"; CREATED_PAUSE=0; fi
}
trap 'cleanup_pause' EXIT
trap 'log "signal: stopping"; exit 1' INT TERM HUP

if [ "$DRY" = "1" ] && [ "$(readlink -f "$Q")" = "$(readlink -f "$HOME/queue")" ]; then echo "S6_DRYRUN=1 must not use the real queue dir"; exit 2; fi
[ -n "$PX" ] || fail "no --px"
[ -f "$PX/index.json" ] && [ -f "$PX/px.u8.npy" ] && [ -f "$PX/meta.f32.npy" ] || fail "px cache incomplete: $PX"

# ---- recipe (nartaa E_prior_s42_fall run_config, mapped to ct_s6v100.py; see recipe_s6s43.json) ----
OUT="$KIT/runs/$RUN_TAG"
ARGS=(--px "$PX" --targets "$KIT/targets_s6seed.csv" --arms E_prior --backbone convnext_small
      --init "$KIT/pretrained/convnext_small_in22k_ft_in1k.safetensors" --seeds 1 --seed-base 43 --folds 5 --only-folds 0
      --epochs 16 --batch 2 --train-windows 64 --crop 288 --slices 3 --slice-jitter 0 --ema 0 --bn-recal-batches 0 --grad-ckpt
      --phys-mm 3.5 --lr 1e-4 --freeze-blocks 16 --llrd 1.0 --head-lr 1e-3 --wd 0.05 --workers 8 --eval-workers 3 --ckpt-every 1
      --eval-stride 3 --train-sampling uniform --cover-stride 3 --spatial-attn --aug crop --aug-res 0
      --i-have-user-authorisation-to-finetune)
ENVS=(S6_ALL_DATA=1 S6_AMP=fp16 S6_PX_GAP="${S6_PX_GAP_RUN:-16}" S6_LOG_EVERY=200 S6_MAX_SKIP_FRAC=0.02 PYTHONUNBUFFERED=1)

# ---- 0. pre-flight (CPU only, before PAUSE) ----
sha() { sha256sum "$1" | cut -c1-64; }
want_ct="$(cat "$KIT/RECEIPT_ct_s6v100.sha256" 2>/dev/null | cut -c1-64)"
[ -n "$want_ct" ] && [ "$(sha "$KIT/ct_s6v100.py")" = "$want_ct" ] || fail "ct_s6v100.py sha differs from RECEIPT_ct_s6v100.sha256"
[ "$(sha "$KIT/pretrained/convnext_small_in22k_ft_in1k.safetensors")" = "2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9" ] || fail "init sha"
[ -f "$KIT/targets_s6seed.csv" ] || fail "no targets_s6seed.csv"
for g in PX_PARITY A7; do
  head -c 4 "$GATES/$g.verdict" 2>/dev/null | grep -q '^PASS' || fail "gate file $GATES/$g.verdict missing or not PASS (main writes it)"
done
log "pre-flight: shas OK, gates PX_PARITY + A7 PASS; check-targets on CPU"
CT_LOG="$KIT/logs/${RUN_TAG}_check_targets.log"
if [ "$DRY" = "1" ]; then echo "pool 4349 gold 58 folds [1,1,1,1,1]" > "$CT_LOG"; echo "check-targets OK" >> "$CT_LOG"
else
  ( cd "$KIT" && env CUDA_VISIBLE_DEVICES= "${ENVS[@]}" nice -n 10 "$PY" -u ct_s6v100.py "${ARGS[@]}" --out "$KIT/runs/${RUN_TAG}_check" --check-targets ) > "$CT_LOG" 2>&1 \
    || fail "check-targets rc != 0 (see $CT_LOG)"
fi
grep -q "check-targets OK" "$CT_LOG" || fail "check-targets not OK"
POOL="$(grep -o '^pool [0-9]*' "$CT_LOG" | head -1 | cut -d' ' -f2)"
log "check-targets OK, pool $POOL (steps = 16 x floor(pool/2) = $((16 * (POOL / 2))))"
ENVS+=(S6_EXPECT_TRAIN="$POOL")

# ---- 1. trigger, then PAUSE ----
if [ -n "$PAUSE_AFTER" ]; then
  log "waiting for $PAUSE_AFTER before PAUSE"; while [ ! -e "$PAUSE_AFTER" ]; do sleep 30; done
elif [ -n "$PAUSE_AT" ]; then
  t_at=$(date -d "$PAUSE_AT" +%s) || fail "bad --pause-at"
  log "waiting until $PAUSE_AT before PAUSE"; while [ "$(date +%s)" -lt "$t_at" ]; do sleep 30; done
fi
if [ -e "$Q/PAUSE" ]; then log "WARNING: $Q/PAUSE already exists (not ours): it is left in place at the end"
else touch "$Q/PAUSE" || fail "cannot create $Q/PAUSE"; CREATED_PAUSE=1; log "created $Q/PAUSE"; fi

# ---- 2. two free-GPU checks FREE_INTERVAL s apart ----
ftcount() { if [ -n "${S6_FTCOUNT_CMD:-}" ]; then $S6_FTCOUNT_CMD; else pgrep -fc '[f]leet_train.py'; fi; }
gpu_busy() {
  local apps; apps="$($NVSMI --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"
  local ft; ft="$(ftcount)"
  [ "$apps" -gt 0 ] || [ "$ft" -gt 0 ]
}
n_free=0
while [ "$n_free" -lt 2 ]; do
  if gpu_busy; then n_free=0; else n_free=$((n_free + 1)); log "GPU free check $n_free/2"; fi
  [ "$n_free" -lt 2 ] && sleep "$FREE_INTERVAL"
done

# ---- 3. 300-step smoke ----
SMOKE_LOG="$KIT/logs/${RUN_TAG}_smoke300.log"
log "smoke: 300 steps -> $SMOKE_LOG"
if [ "$DRY" = "1" ]; then
  "$PY" -c 'import json,os; print("S6_SMOKE " + json.dumps({"steps": 300, "nonfinite_losses": 0, "s_per_step_median": float(os.environ.get("S6_DRY_SPS", "1.3")), "peak_mem_reserved_gb": float(os.environ.get("S6_DRY_MEM", "11.0")), "scaler_skipped": 6}))' > "$SMOKE_LOG" 2>&1
  rc=$?
else
  rm -rf "$KIT/runs/${RUN_TAG}_smoke300"
  ( cd "$KIT" && env "${ENVS[@]}" S6_MAX_STEPS=300 S6_LOG_EVERY=50 "$PY" -u ct_s6v100.py "${ARGS[@]}" --out "$KIT/runs/${RUN_TAG}_smoke300" ) > "$SMOKE_LOG" 2>&1
  rc=$?
fi
[ "$rc" = "0" ] || fail "smoke rc $rc (see $SMOKE_LOG)"
SM="$(grep '^S6_SMOKE ' "$SMOKE_LOG" | tail -1 | cut -c10-)"
[ -n "$SM" ] || fail "no S6_SMOKE line"
GATE="$("$PY" -c '
import json, sys
r = json.loads(sys.argv[1]); sps, mem, skip = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
bad = []
if r.get("steps") != 300: bad.append("steps %s" % r.get("steps"))
if r.get("nonfinite_losses", 1) != 0: bad.append("non-finite losses %s" % r.get("nonfinite_losses"))
if not (r.get("s_per_step_median") or 99) <= sps: bad.append("s/step %s > %s" % (r.get("s_per_step_median"), sps))
if not (r.get("peak_mem_reserved_gb") or 99) < mem: bad.append("peak reserved %s GB >= %s" % (r.get("peak_mem_reserved_gb"), mem))
if (r.get("scaler_skipped") or 0) > skip: bad.append("scaler skipped %s > %s" % (r.get("scaler_skipped"), skip))
print("PASS" if not bad else "FAIL " + "; ".join(bad))' "$SM" "$MAX_SPS" "$MAX_MEM" "$MAX_SKIP")"
log "smoke gate: $GATE | $SM"
echo "$GATE $SM" > "$GATES/SMOKE_${RUN_TAG}.verdict"
case "$GATE" in PASS*) ;; *) fail "smoke gate: $GATE";; esac

# ---- 4. launch, wait until it holds the GPU, then rm PAUSE ----
RUN_LOG="$KIT/runs/${RUN_TAG}.log"
log "launch -> $OUT ($RUN_LOG)"
if [ "$DRY" = "1" ]; then
  setsid nohup bash -c "echo \$\$ > '$KIT/runs/${RUN_TAG}.pid'; exec sleep ${S6_DRY_RUN_S:-60}" > "$RUN_LOG" 2>&1 < /dev/null &
else
  ( cd "$KIT" && setsid nohup env "${ENVS[@]}" "$PY" -u ct_s6v100.py "${ARGS[@]}" --out "$OUT" --resume > "$RUN_LOG" 2>&1 < /dev/null & echo $! > "$KIT/runs/${RUN_TAG}.pid" )
fi
sleep 2
PID="$(cat "$KIT/runs/${RUN_TAG}.pid" 2>/dev/null)"
[ -n "$PID" ] || fail "no pid file"
t0=$(date +%s)
while true; do
  kill -0 "$PID" 2>/dev/null || fail "run process $PID exited before holding the GPU (see $RUN_LOG)"
  if $NVSMI --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$PID"; then break; fi
  [ $(( $(date +%s) - t0 )) -lt "$APPEAR_TIMEOUT" ] || fail "run process $PID not on the GPU after ${APPEAR_TIMEOUT}s"
  sleep 10
done
LAUNCHED=1
log "run pid $PID holds the GPU"
if [ "$CREATED_PAUSE" = "1" ]; then rm -f "$Q/PAUSE" && log "removed $Q/PAUSE (queue resumes when the S6 job ends)"; CREATED_PAUSE=0; fi
echo "PASS $(date '+%F %T') launched pid $PID out $OUT pool $POOL smoke $SM" > "$VERDICT"
log "DONE"
