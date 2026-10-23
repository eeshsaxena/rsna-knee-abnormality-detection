#!/bin/bash
# launch_when_free.sh v2 (10-15; v1 = launch_when_free.v1_a5db4932.sh). BOX_PLAN_1015 sec 3.5 PAUSE protocol + Amendment 7 pre-GPU
# gates for the external S6 member s43, with the 10-15 review fixes S6R-2 (pool pinned by A7), S6R-3 (PAUSE race), S6R-4 (time
# limits + watcher), S6R-5 (smoke gate on the loss scale).  ARMED BY MAIN ONLY (it creates and removes ~/queue/PAUSE and starts a
# GPU job):
#   setsid nohup bash ~/s6seed/launch_when_free.sh --px ~/s6seed/px320 --after-job <out dir name> [--after-job <alt name>] \
#       > ~/s6seed/logs/launch_when_free.log 2>&1 < /dev/null &
#   (--now instead of --after-job: PAUSE at once.  Stop everything later: touch ~/s6seed/gates/STOP_s6s43, then kill the pids.)
# Sequence:
#   0. pre-flight, CPU only, no PAUSE: kit shas; gates/PX_PARITY.verdict and gates/A7.verdict start with PASS; A7.verdict pins
#      'pool <N>' (and 'head_seed 42' iff main exports S6_HEAD_SEED_RUN=42, the soup option); ct_s6v100 --check-targets on CPU must give gold 58 and pool == N >= 4,328 (4,349 x (1 - .005), nartaa's
#      max_fail_frac); S6_EXPECT_TRAIN = N (from A7, never from the cache); px build errors listed in gates/PX_ERRORS_<tag>.txt
#   1. trigger: the queue runner's LAUNCH line for the named job (~/fleet_runs/queue_runner.log, '-> .../fleet_runs/<name> (')
#      and that job still running (no DONE / FAILED after it, a compute app on the GPU) at least S6_PAUSE_DELAY s (default 3,000 s,
#      past the runner's early-OOM fallback window) after its launch.  Then n0 = queue_runner.log line count, THEN an owner-tagged
#      PAUSE ('s6s43 launch_when_free pid ...'; a PAUSE that already exists is left alone and never removed by us).  The job
#      ahead keeps running; PAUSE only holds NEW launches, so it sits through the predecessor's run (harmless on one GPU).
#   2. two free-GPU checks FREE_INTERVAL apart, at most S6_MAX_FREE_WAIT_S (default 10 h); a queue LAUNCH line after n0 = the
#      runner beat our PAUSE: remove our PAUSE, FAIL
#   3. 300-step smoke under `timeout S6_SMOKE_TIMEOUT` (default 2,700 s): rc 0, 0 non-finite losses, s/step median <= 1.8, peak
#      reserved < 15.5 GB, scaler skipped <= 25, final loss scale >= 2**10, loss scale min >= 2**8
#   4. launch the real run through a wrapper (run pid + rc files), wait until it holds the GPU, re-check no LAUNCH after n0, then
#      remove OUR PAUSE (only if its content and mtime are still ours)
#   5. watch until the run ends: no log output for S6_STALL_S (default 1,800 s) -> gates/STALL_<tag> + a line in
#      ~/fleet_runs/IDLE_ALARM; a crash (no DONE, no 'S6V100 ABORT', rc not 143, no STOP file) -> re-PAUSE at once (the runner
#      needs 2 free checks 120 s apart; the watch poll is 10 s), free-GPU checks, relaunch with --resume (no smoke), at most
#      S6_AUTO_RESUME (default 1) times; the end -> gates/RUN_<tag>.verdict PASS / FAIL (+ an IDLE_ALARM line on FAIL)
#   Any failure before the run holds the GPU: remove OUR PAUSE, write gates/LAUNCH_<tag>.verdict FAIL <reason>, exit 1.  The EXIT
#   trap does the same if the script dies, so PAUSE is never left behind without a job on the GPU.
# Test hooks (tests/test_launch_when_free.sh): S6_QUEUE_DIR, S6_QLOG, S6_ALARM, S6_NVSMI, S6_FTCOUNT_CMD, S6_FREE_INTERVAL, S6_POLL,
# S6_WATCH_POLL, S6_PAUSE_DELAY, S6_DRYRUN=1 (stub check-targets / smoke / run; then the queue dir must NOT be ~/queue).  No em dashes.
set -u
KIT="${S6_KIT:-$HOME/s6seed}"
Q="${S6_QUEUE_DIR:-$HOME/queue}"
QLOG="${S6_QLOG:-$HOME/fleet_runs/queue_runner.log}"
ALARM="${S6_ALARM:-$HOME/fleet_runs/IDLE_ALARM}"
NVSMI="${S6_NVSMI:-nvidia-smi}"
PY="${S6_PY:-$HOME/miniconda3/envs/rsna/bin/python}"
FREE_INTERVAL="${S6_FREE_INTERVAL:-120}"
POLL="${S6_POLL:-30}"
WATCH_POLL="${S6_WATCH_POLL:-10}"
APPEAR_TIMEOUT="${S6_APPEAR_TIMEOUT:-900}"
PAUSE_DELAY="${S6_PAUSE_DELAY:-3000}"
MAX_TRIGGER_WAIT="${S6_MAX_TRIGGER_WAIT_S:-129600}"
MAX_FREE_WAIT="${S6_MAX_FREE_WAIT_S:-36000}"
SMOKE_TIMEOUT="${S6_SMOKE_TIMEOUT:-2700}"
STALL_S="${S6_STALL_S:-1800}"
AUTO_RESUME="${S6_AUTO_RESUME:-1}"
POOL_FLOOR="${S6_POOL_FLOOR:-4328}"
DRY="${S6_DRYRUN:-0}"
PX=""; RUN_TAG="s6s43"; NOW=0; AFTER=()
MAX_SPS="1.8"; MAX_MEM="15.5"; MAX_SKIP="25"; MIN_SCALE_END="1024"; MIN_SCALE_MIN="256"
while [ $# -gt 0 ]; do
  case "$1" in
    --px) PX="$2"; shift 2;;
    --after-job) AFTER+=("$2"); shift 2;;
    --now) NOW=1; shift;;
    --tag) RUN_TAG="$2"; shift 2;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
GATES="$KIT/gates"; mkdir -p "$GATES" "$KIT/logs" "$KIT/runs"
VERDICT="$GATES/LAUNCH_${RUN_TAG}.verdict"
RUNV="$GATES/RUN_${RUN_TAG}.verdict"
STOPF="$GATES/STOP_${RUN_TAG}"
log() { echo "$(date '+%F %T') $*"; }
alarm() { echo "$(date '+%F %T') ${RUN_TAG} $*" >> "$ALARM" 2>/dev/null; log "IDLE_ALARM line: $*"; }
CREATED_PAUSE=0; LAUNCHED=0; PAUSE_TAG=""; PAUSE_MT=""; N0=0; PHASE=pre
fail() {
  log "FAIL $*"
  if [ "$PHASE" = "watch" ]; then echo "FAIL $(date '+%F %T') $*" > "$RUNV"; alarm "run not resumed: $*"
  else echo "FAIL $(date '+%F %T') $*" > "$VERDICT"; fi
  cleanup_pause; exit 1
}
take_pause() {   # owner-tagged PAUSE; a PAUSE that already exists is not ours and is left alone
  if [ -e "$Q/PAUSE" ]; then log "WARNING: $Q/PAUSE already exists (not ours): left in place, never removed by us"; return 0; fi
  PAUSE_TAG="${RUN_TAG} launch_when_free pid $$ $(date '+%F %T')"
  echo "$PAUSE_TAG" > "$Q/PAUSE" || return 1
  PAUSE_MT="$(stat -c %Y "$Q/PAUSE")"; CREATED_PAUSE=1; log "created $Q/PAUSE ($PAUSE_TAG)"
}
release_pause() {   # remove PAUSE only if it is still exactly ours (content + mtime); someone else's touch / write keeps it
  [ "$CREATED_PAUSE" = "1" ] || return 0
  CREATED_PAUSE=0
  if [ -e "$Q/PAUSE" ] && [ "$(cat "$Q/PAUSE" 2>/dev/null)" = "$PAUSE_TAG" ] && [ "$(stat -c %Y "$Q/PAUSE")" = "$PAUSE_MT" ]; then
    rm -f "$Q/PAUSE" && log "removed $Q/PAUSE ($1)"
  else
    log "WARNING: $Q/PAUSE was changed or removed by someone else since we created it: left as it is ($1)"
  fi
}
cleanup_pause() { [ "$LAUNCHED" = "0" ] && release_pause "no S6 job on the GPU"; return 0; }
trap 'cleanup_pause' EXIT
trap 'log "signal: stopping"; exit 1' INT TERM HUP
qlines() { [ -f "$QLOG" ] && wc -l < "$QLOG" || echo 0; }
launch_after_n0() { [ -f "$QLOG" ] && tail -n +$((N0 + 1)) "$QLOG" | grep -q '\] LAUNCH '; }
gpu_apps() { $NVSMI --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'; }
ftcount() { if [ -n "${S6_FTCOUNT_CMD:-}" ]; then $S6_FTCOUNT_CMD; else pgrep -fc '[f]leet_train.py'; fi; }
gpu_busy() { [ "$(gpu_apps)" -gt 0 ] || [ "$(ftcount)" -gt 0 ]; }
wait_free() {   # $1 = what for; two consecutive free checks FREE_INTERVAL apart, a LAUNCH after n0 = FAIL, capped
  local n_free=0 t0; t0=$(date +%s)
  while [ "$n_free" -lt 2 ]; do
    launch_after_n0 && fail "the queue runner launched a job after our PAUSE ($1): $(tail -n +$((N0 + 1)) "$QLOG" | grep '\] LAUNCH ' | head -1 | cut -c1-160)"
    [ -e "$STOPF" ] && fail "STOP file $STOPF ($1)"
    if gpu_busy; then n_free=0; else n_free=$((n_free + 1)); log "GPU free check $n_free/2 ($1)"; fi
    if [ $(( $(date +%s) - t0 )) -ge "$MAX_FREE_WAIT" ]; then alarm "free-GPU wait > ${MAX_FREE_WAIT}s ($1): PAUSE released"; fail "free-GPU wait > ${MAX_FREE_WAIT}s ($1)"; fi
    [ "$n_free" -lt 2 ] && sleep "$FREE_INTERVAL"
  done
}

if [ "$DRY" = "1" ] && [ "$(readlink -f "$Q")" = "$(readlink -f "$HOME/queue")" ]; then echo "S6_DRYRUN=1 must not use the real queue dir"; exit 2; fi
[ -n "$PX" ] || fail "no --px"
[ "$NOW" = "1" ] || [ "${#AFTER[@]}" -gt 0 ] || fail "give --after-job <queue out dir name> (or --now)"
[ -f "$PX/index.json" ] && [ -f "$PX/px.u8.npy" ] && [ -f "$PX/meta.f32.npy" ] || fail "px cache incomplete: $PX"

# ---- recipe (nartaa E_prior_s42_fall run_config, mapped to ct_s6v100.py; see recipe_s6s43.json) ----
OUT="$KIT/runs/$RUN_TAG"
ARGS=(--px "$PX" --targets "$KIT/targets_s6seed.csv" --arms E_prior --backbone convnext_small
      --init "$KIT/pretrained/convnext_small_in22k_ft_in1k.safetensors" --seeds 1 --seed-base 43 --folds 5 --only-folds 0
      --epochs 16 --batch 2 --train-windows 64 --crop 288 --slices 3 --slice-jitter 0 --ema 0 --bn-recal-batches 0 --grad-ckpt
      --phys-mm 3.5 --lr 1e-4 --freeze-blocks 16 --llrd 1.0 --head-lr 1e-3 --wd 0.05 --workers 8 --eval-workers 3 --ckpt-every 1
      --eval-stride 3 --train-sampling uniform --cover-stride 3 --spatial-attn --aug crop --aug-res 0
      --i-have-user-authorisation-to-finetune)
ENVS=(S6_ALL_DATA=1 S6_AMP=fp16 S6_PX_GAP="${S6_PX_GAP_RUN:-16}" S6_LOG_EVERY=200 S6_MAX_SKIP_FRAC=0.02 S6_MIN_SCALE=256
      S6_LOADER_TIMEOUT=600 PYTHONUNBUFFERED=1)
[ -n "${S6_HEAD_SEED_RUN:-}" ] && ENVS+=(S6_HEAD_SEED="$S6_HEAD_SEED_RUN")    # P7 soup option: only if main sets it AND A7 says so
if [ "${S6_PRINT_RECIPE:-0}" = "1" ]; then printf '%s\n' "${ARGS[@]}"; echo "--ENV--"; printf '%s\n' "${ENVS[@]}"; exit 0; fi

# ---- 0. pre-flight (CPU only, before PAUSE) ----
sha() { sha256sum "$1" | cut -c1-64; }
want_ct="$(cut -c1-64 "$KIT/RECEIPT_ct_s6v100.sha256" 2>/dev/null)"
[ -n "$want_ct" ] && [ "$(sha "$KIT/ct_s6v100.py")" = "$want_ct" ] || fail "ct_s6v100.py sha differs from RECEIPT_ct_s6v100.sha256"
[ "$(sha "$KIT/pretrained/convnext_small_in22k_ft_in1k.safetensors")" = "2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9" ] || fail "init sha"
[ -f "$KIT/targets_s6seed.csv" ] || fail "no targets_s6seed.csv"
for g in PX_PARITY A7; do
  head -c 4 "$GATES/$g.verdict" 2>/dev/null | grep -q '^PASS' || fail "gate file $GATES/$g.verdict missing or not PASS (main writes it)"
done
A7POOL="$(grep -o 'pool [0-9]\+' "$GATES/A7.verdict" | head -1 | cut -d' ' -f2)"
[ -n "$A7POOL" ] || fail "A7.verdict pins no 'pool <N>' (write e.g. 'PASS A7 pasted <time> pool 4349 steps 34784')"
[ "$A7POOL" -ge "$POOL_FLOOR" ] || fail "A7 pool $A7POOL < floor $POOL_FLOOR (4,349 x (1 - .005))"
A7HS="$(grep -o 'head_seed [0-9]\+' "$GATES/A7.verdict" | head -1 | cut -d' ' -f2)"
[ "${A7HS:-43}" = "${S6_HEAD_SEED_RUN:-43}" ] || fail "head-init seed: A7.verdict says ${A7HS:-43 (default)}, S6_HEAD_SEED_RUN is ${S6_HEAD_SEED_RUN:-unset (43)}"
log "head-init seed ${S6_HEAD_SEED_RUN:-43} (= A7)"
log "pre-flight: shas OK, gates PX_PARITY + A7 PASS, A7 pool $A7POOL; check-targets on CPU"
CT_LOG="$KIT/logs/${RUN_TAG}_check_targets.log"
if [ "$DRY" = "1" ]; then printf 'pool %s gold %s folds [1,1,1,1,1]\ncheck-targets OK\n' "${S6_DRY_POOL:-4349}" "${S6_DRY_GOLD:-58}" > "$CT_LOG"
else
  ( cd "$KIT" && env CUDA_VISIBLE_DEVICES= "${ENVS[@]}" nice -n 10 "$PY" -u ct_s6v100.py "${ARGS[@]}" --out "$KIT/runs/${RUN_TAG}_check" --check-targets ) > "$CT_LOG" 2>&1 \
    || fail "check-targets rc != 0 (see $CT_LOG)"
fi
grep -q "check-targets OK" "$CT_LOG" || fail "check-targets not OK"
POOL="$(grep -o '^pool [0-9]*' "$CT_LOG" | head -1 | cut -d' ' -f2)"
GOLD="$(grep -o '^pool [0-9]* gold [0-9]*' "$CT_LOG" | head -1 | cut -d' ' -f4)"
[ "$GOLD" = "58" ] || fail "check-targets gold $GOLD != 58 (gold58 studies missing from the px cache)"
[ "$POOL" = "$A7POOL" ] || fail "check-targets pool $POOL != A7 pool $A7POOL (partial or different px cache)"
NERR="$("$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])).get("errors") or {}; open(sys.argv[2],"w").write("".join("%s %s\n" % (u, str(v)[:200]) for u, v in sorted(e.items()))); print(len(e))' "$PX/index.json" "$GATES/PX_ERRORS_${RUN_TAG}.txt" 2>/dev/null)"
log "check-targets OK: pool $POOL (= A7) gold $GOLD, px build errors ${NERR:-?} (gates/PX_ERRORS_${RUN_TAG}.txt); steps = 16 x floor(pool/2) = $((16 * (POOL / 2)))"
ENVS+=(S6_EXPECT_TRAIN="$A7POOL")

# ---- 1. trigger, then n0, then PAUSE ----
job_state() {   # $1 = out dir name -> "none" | "running <launch epoch s>" | "ended"
  local L lt ln ts
  [ -f "$QLOG" ] || { echo none; return; }
  L="$(grep -nF "/fleet_runs/$1 " "$QLOG" | grep '\] LAUNCH ' | tail -1)"
  [ -n "$L" ] || { echo none; return; }
  ln="${L%%:*}"
  if tail -n +$((ln + 1)) "$QLOG" | grep -F "/fleet_runs/$1 " | grep -qE '\] (DONE|FAILED) '; then echo ended; return; fi
  ts="$(echo "${L#*:}" | sed -E 's/^\[queue-runner ([0-9]{2})-([0-9]{2}) ([0-9:]{8})\].*/\1-\2 \3/')"
  lt="$(date -d "$(date +%Y)-$ts" +%s 2>/dev/null || echo 0)"
  echo "running $lt"
}
if [ "$NOW" = "1" ]; then
  log "--now: PAUSE at once"
else
  log "waiting for the queue job ${AFTER[*]} to run >= ${PAUSE_DELAY}s before PAUSE (queue_runner.log $QLOG)"
  t0=$(date +%s)
  while true; do
    [ -e "$STOPF" ] && fail "STOP file $STOPF (trigger wait)"
    hit=""
    for j in "${AFTER[@]}"; do
      s="$(job_state "$j")"
      case "$s" in
        ended) fail "the job ahead $j already ended (DONE / FAILED in the queue log) before our PAUSE: re-arm with the next job";;
        running*) lt="${s#running }"; if [ $(( $(date +%s) - lt )) -ge "$PAUSE_DELAY" ]; then hit="$j"; fi;;
      esac
    done
    if [ -n "$hit" ]; then
      [ "$(gpu_apps)" -gt 0 ] || fail "the job ahead $hit is in the queue log as running but no compute app holds the GPU"
      log "the job ahead $hit is running (>= ${PAUSE_DELAY}s since its LAUNCH line)"; break
    fi
    [ $(( $(date +%s) - t0 )) -lt "$MAX_TRIGGER_WAIT" ] || fail "the job ahead (${AFTER[*]}) never ran within ${MAX_TRIGGER_WAIT}s (skipped / re-ordered?)"
    sleep "$POLL"
  done
fi
N0="$(qlines)"
take_pause || fail "cannot create $Q/PAUSE"
sleep "${S6_PAUSE_SETTLE:-5}"
launch_after_n0 && fail "the queue runner launched a job while we took PAUSE"

# ---- 2. two free-GPU checks FREE_INTERVAL s apart (capped) ----
wait_free "job ahead"

# ---- 3. 300-step smoke (timeout) ----
SMOKE_LOG="$KIT/logs/${RUN_TAG}_smoke300.log"
log "smoke: 300 steps -> $SMOKE_LOG (timeout ${SMOKE_TIMEOUT}s)"
if [ "$DRY" = "1" ]; then
  timeout -k 5 "$SMOKE_TIMEOUT" bash -c 'sleep "${S6_DRY_SMOKE_SLEEP:-0}"; "$0" -c "import json,os; e=os.environ.get; print(\"S6_SMOKE \" + json.dumps({\"steps\": 300, \"nonfinite_losses\": 0, \"s_per_step_median\": float(e(\"S6_DRY_SPS\", \"1.3\")), \"peak_mem_reserved_gb\": float(e(\"S6_DRY_MEM\", \"11.0\")), \"scaler_skipped\": 6, \"loss_scale\": float(e(\"S6_DRY_SCALE\", \"32768\")), \"loss_scale_min\": float(e(\"S6_DRY_SCALE_MIN\", \"8192\"))}))"' "$PY" > "$SMOKE_LOG" 2>&1
  rc=$?
else
  rm -rf "$KIT/runs/${RUN_TAG}_smoke300"
  ( cd "$KIT" && timeout -k 60 "$SMOKE_TIMEOUT" env "${ENVS[@]}" S6_MAX_STEPS=300 S6_LOG_EVERY=50 "$PY" -u ct_s6v100.py "${ARGS[@]}" --out "$KIT/runs/${RUN_TAG}_smoke300" ) > "$SMOKE_LOG" 2>&1
  rc=$?
fi
[ "$rc" = "124" ] && fail "smoke timed out after ${SMOKE_TIMEOUT}s (see $SMOKE_LOG)"
[ "$rc" = "0" ] || fail "smoke rc $rc (see $SMOKE_LOG)"
SM="$(grep '^S6_SMOKE ' "$SMOKE_LOG" | tail -1 | cut -c10-)"
[ -n "$SM" ] || fail "no S6_SMOKE line"
GATE="$("$PY" -c '
import json, sys
r = json.loads(sys.argv[1]); sps, mem, skip, s_end, s_min = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
bad = []
if r.get("steps") != 300: bad.append("steps %s" % r.get("steps"))
if r.get("nonfinite_losses", 1) != 0: bad.append("non-finite losses %s" % r.get("nonfinite_losses"))
if not (r.get("s_per_step_median") or 99) <= sps: bad.append("s/step %s > %s" % (r.get("s_per_step_median"), sps))
if not (r.get("peak_mem_reserved_gb") or 99) < mem: bad.append("peak reserved %s GB >= %s" % (r.get("peak_mem_reserved_gb"), mem))
if (r.get("scaler_skipped") or 0) > skip: bad.append("scaler skipped %s > %s" % (r.get("scaler_skipped"), skip))
if not (r.get("loss_scale") or 0) >= s_end: bad.append("final loss scale %s < %s" % (r.get("loss_scale"), s_end))
if not (r.get("loss_scale_min") or 0) >= s_min: bad.append("loss scale min %s < %s" % (r.get("loss_scale_min"), s_min))
print("PASS" if not bad else "FAIL " + "; ".join(bad))' "$SM" "$MAX_SPS" "$MAX_MEM" "$MAX_SKIP" "$MIN_SCALE_END" "$MIN_SCALE_MIN")"
log "smoke gate: $GATE | $SM"
echo "$GATE $SM" > "$GATES/SMOKE_${RUN_TAG}.verdict"
case "$GATE" in PASS*) ;; *) fail "smoke gate: $GATE";; esac

# ---- 4. launch through a wrapper, wait until it holds the GPU, then release PAUSE ----
RUN_LOG="$KIT/runs/${RUN_TAG}.log"; PIDF="$KIT/runs/${RUN_TAG}.pid"; RCF="$KIT/runs/${RUN_TAG}.rc"
start_run() {   # $1 = attempt number; the wrapper records the trainer pid and its rc
  rm -f "$PIDF" "$RCF"
  log "launch attempt $1 -> $OUT ($RUN_LOG)"
  if [ "$DRY" = "1" ]; then
    setsid nohup bash -c "bash '$KIT/stub_run.sh' '$OUT' $1 >> '$RUN_LOG' 2>&1 & echo \$! > '$PIDF'; wait \$!; echo \$? > '$RCF'" > /dev/null 2>&1 < /dev/null &
  else
    ( cd "$KIT" && setsid nohup bash -c 'cd "$1"; shift; env "$@" >> "$RUN_LOG_W" 2>&1 & echo $! > "$PIDF_W"; wait $!; echo $? > "$RCF_W"' \
        _ "$KIT" "${ENVS[@]}" "$PY" -u ct_s6v100.py "${ARGS[@]}" --out "$OUT" --resume > /dev/null 2>&1 < /dev/null & )
  fi
  local i; for i in $(seq 1 50); do [ -s "$PIDF" ] && break; sleep 0.2; done
  PID="$(cat "$PIDF" 2>/dev/null)"
  [ -n "$PID" ] || fail "no pid file (attempt $1)"
  local t0; t0=$(date +%s)
  while true; do
    kill -0 "$PID" 2>/dev/null || fail "run process $PID exited before holding the GPU (attempt $1, see $RUN_LOG)"
    if $NVSMI --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$PID"; then break; fi
    [ $(( $(date +%s) - t0 )) -lt "$APPEAR_TIMEOUT" ] || fail "run process $PID not on the GPU after ${APPEAR_TIMEOUT}s (attempt $1)"
    sleep "$WATCH_POLL"
  done
  launch_after_n0 && log "WARNING: a queue LAUNCH line appeared after n0 while our run started (both on the GPU?)"
  LAUNCHED=1
  log "run pid $PID holds the GPU (attempt $1)"
  release_pause "the S6 run holds the GPU; the queue resumes when it ends"
}
export RUN_LOG_W="$RUN_LOG" PIDF_W="$PIDF" RCF_W="$RCF"
start_run 1
echo "PASS $(date '+%F %T') launched pid $PID out $OUT pool $POOL (A7) px_errors ${NERR:-?} smoke $SM" > "$VERDICT"

# ---- 5. watch: stall alarm, crash -> resume (bounded), final verdict ----
attempt=1; stalled=0; PHASE=watch
while true; do
  if kill -0 "$PID" 2>/dev/null; then
    age=$(( $(date +%s) - $(stat -c %Y "$RUN_LOG" 2>/dev/null || date +%s) ))
    if [ "$age" -ge "$STALL_S" ] && [ "$stalled" = "0" ]; then
      stalled=1; echo "STALL $(date '+%F %T') pid $PID: no output in $RUN_LOG for ${age}s" > "$GATES/STALL_${RUN_TAG}"
      alarm "STALL: run pid $PID wrote nothing for ${age}s (alive, GPU held); not killed"
    elif [ "$age" -lt "$STALL_S" ] && [ "$stalled" = "1" ]; then
      stalled=0; log "run output resumed after a stall"; rm -f "$GATES/STALL_${RUN_TAG}"
    fi
    sleep "$WATCH_POLL"; continue
  fi
  for i in $(seq 1 30); do [ -s "$RCF" ] && break; sleep 0.2; done
  rc="$(cat "$RCF" 2>/dev/null || echo '?')"
  if [ -f "$OUT/models/E_prior_s43_fall.DONE.json" ] && grep -q "ALL DONE" "$RUN_LOG" && [ "$rc" = "0" ]; then
    echo "PASS $(date '+%F %T') run finished rc 0, attempts $attempt, $(grep '^S6V100 DONE ' "$RUN_LOG" | tail -1 | cut -c13-)" > "$RUNV"
    log "run finished: $(cat "$RUNV")"; exit 0
  fi
  why="rc $rc"
  grep -q 'S6V100 ABORT' "$RUN_LOG" && why="$why, $(grep -o 'S6V100 ABORT.*' "$RUN_LOG" | tail -1 | cut -c1-200)"
  if grep -q 'S6V100 ABORT' "$RUN_LOG" || [ "$rc" = "143" ] || [ -e "$STOPF" ] || [ "$attempt" -gt "$AUTO_RESUME" ]; then
    echo "FAIL $(date '+%F %T') run ended without a model ($why), attempts $attempt; no resume (ABORT / SIGTERM / STOP file / limit)" > "$RUNV"
    alarm "run ended without a model ($why): GPU back to the queue"
    log "$(cat "$RUNV")"; exit 1
  fi
  attempt=$((attempt + 1)); LAUNCHED=0
  N0="$(qlines)"
  take_pause || fail "cannot re-create $Q/PAUSE for the resume"
  log "run crashed ($why): resume attempt $attempt from the last epoch checkpoint"
  sleep "${S6_PAUSE_SETTLE:-5}"
  launch_after_n0 && fail "crash ($why); the queue launched a job before our resume PAUSE"
  wait_free "resume $attempt"
  start_run "$attempt"
  echo "PASS $(date '+%F %T') resumed (attempt $attempt) pid $PID out $OUT" >> "$VERDICT"
done
