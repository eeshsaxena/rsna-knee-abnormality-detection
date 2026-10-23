#!/bin/bash
# Dry-run tests of launch_when_free.sh (v2 suite, 10-15 review S6R-2/3/4/5): a FAKE queue dir, a FAKE queue_runner.log, a fake
# nvidia-smi (pids in $T/gpu_pids + the live stub run of a case that has a 'follow' marker), stub check-targets / smoke / run
# (S6_DRYRUN=1).  Never touches ~/queue, ~/fleet_runs or the GPU.  Usage: test_launch_when_free.sh [LAUNCHER] [PX]
# Runs against v1 too (fail-before): v1 has no --after-job, so the trigger cases give it --pause-after <file> created at the
# moment the fake runner logs the job ahead's LAUNCH, and --now becomes no argument.  Prints LAUNCH_TESTS PASS n/n.
set -u
LWF="${1:-$HOME/s6seed/launch_when_free.sh}"
PX="${2:-$HOME/s6seed/tests/px_fixture}"
SRC="$HOME/s6seed"
T="$(mktemp -d /tmp/s6lwf2.XXXXXX)"
V2=0; grep -q -- '--after-job' "$LWF" && V2=1
PASS=0; N=0
ok() { N=$((N+1)); if [ "$1" = "1" ]; then PASS=$((PASS+1)); echo "PASS $2"; else echo "FAIL $2"; fi; }
J=fA_job_ahead
qts() { date -d "@$(( $(date +%s) - ${1:-0} ))" '+%m-%d %H:%M:%S'; }
qlaunch() { echo "[queue-runner $(qts "${3:-0}")] LAUNCH job $2 (test) -> /home/training/fleet_runs/$2 (grad_ckpt 0; gate: none) | trainer md5 x" >> "$1"; }
qdone() { echo "[queue-runner $(qts)] DONE job $2 (test) | /home/training/fleet_runs/$2 | rc 0 | {}" >> "$1"; }
mkkit() {   # $1 = case dir
  local k="$1/kit"
  mkdir -p "$k/gates" "$k/pretrained" "$k/logs" "$k/runs" "$1/queue"; : > "$1/qlog"; : > "$1/alarm"
  cp "$LWF" "$k/launch_when_free.sh"
  cp "$SRC/ct_s6v100.py" "$k/"; sha256sum "$k/ct_s6v100.py" | cut -c1-64 > "$k/RECEIPT_ct_s6v100.sha256"
  ln -sf "$SRC/pretrained/convnext_small_in22k_ft_in1k.safetensors" "$k/pretrained/"
  cp "$SRC/targets_s6seed.csv" "$k/"
  echo "PASS test" > "$k/gates/PX_PARITY.verdict"; echo "PASS A7 test pool 4349 steps 34784" > "$k/gates/A7.verdict"
  cat > "$k/stub_run.sh" <<'EOF'
#!/bin/bash
OUT="$1"; A="$2"; mkdir -p "$OUT/models"; MODE="${S6_DRY_RUN_MODE:-ok}"
echo "stub run attempt $A mode $MODE pid $$"
okend() { sleep "${S6_DRY_RUN_S:-3}"; echo '{"model": "E_prior_s43_fall.pt"}' > "$OUT/models/E_prior_s43_fall.DONE.json"; echo 'S6V100 DONE {"model": "E_prior_s43_fall.pt", "optimizer_steps": 34784}'; echo "ALL DONE"; exit 0; }
case "$MODE" in
  ok) okend;;
  crash_once) if [ "$A" = "1" ]; then sleep 2; echo "RuntimeError: DataLoader worker (pid(s) 1) exited unexpectedly / timed out after 600 seconds"; exit 1; fi; okend;;
  abort) sleep 2; echo "SystemExit: S6V100 ABORT: loss scale 128 < S6_MIN_SCALE 256 at step 9000"; exit 1;;
  stall) echo "S6V100 step 200: loss 0.5"; sleep "${S6_DRY_RUN_S:-60}"; exit 1;;
esac
EOF
}
cat > "$T/fake_nvsmi.sh" <<EOF
#!/bin/bash
cat "$T/gpu_pids" 2>/dev/null
for d in $T/*/; do [ -e "\$d/follow" ] || continue; p=\$(cat "\$d/kit/runs/s6s43.pid" 2>/dev/null); [ -n "\$p" ] && kill -0 "\$p" 2>/dev/null && echo "\$p"; done
EOF
chmod +x "$T/fake_nvsmi.sh"
: > "$T/gpu_pids"
start() {   # $1 = case, $2 = trigger kind (now | after), rest = env; background, pid in LP
  local c="$1" kind="$2"; shift 2
  local trig=()
  if [ "$kind" = "after" ]; then if [ "$V2" = 1 ]; then trig=(--after-job "$J"); else trig=(--pause-after "$T/$c/trigger"); fi
  elif [ "$V2" = 1 ]; then trig=(--now); fi
  env S6_FREE_INTERVAL=1 S6_POLL=1 S6_WATCH_POLL=1 S6_PAUSE_SETTLE=1 S6_PAUSE_DELAY=0 S6_APPEAR_TIMEOUT=8 "$@" \
      S6_KIT="$T/$c/kit" S6_QUEUE_DIR="$T/$c/queue" S6_QLOG="$T/$c/qlog" S6_ALARM="$T/$c/alarm" S6_NVSMI="$T/fake_nvsmi.sh" \
      S6_FTCOUNT_CMD="echo 0" S6_DRYRUN=1 bash "$T/$c/kit/launch_when_free.sh" --px "$PX" "${trig[@]}" > "$T/$c/lwf.log" 2>&1 &
  LP=$!
}
waitfor() { local i; for i in $(seq 1 $((${2:-30} * 5))); do eval "$1" && return 0; sleep 0.2; done; return 1; }
finish() { local i; for i in $(seq 1 $((${2:-40} * 5))); do kill -0 "$1" 2>/dev/null || { wait "$1"; return $?; }; sleep 0.2; done; kill "$1" 2>/dev/null; wait "$1"; echo "(killed: no exit in ${2:-40}s)" >> "$3"; return 99; }
killrun() { local p; p="$(cat "$T/$1/kit/runs/s6s43.pid" 2>/dev/null)"; [ -n "$p" ] && kill "$p" 2>/dev/null; true; }
V() { cat "$T/$1/kit/gates/$2" 2>/dev/null; }

# C1 happy path: the job ahead runs (LAUNCH 2 h ago, GPU busy) -> owner PAUSE -> it ends -> 2 free checks -> smoke -> launch
#    -> PAUSE released -> the watcher sees the run finish -> RUN verdict PASS
c=c1; mkkit "$T/$c"; qlaunch "$T/$c/qlog" "$J" 7200; touch "$T/$c/trigger"; echo 99999 > "$T/gpu_pids"
start $c after S6_DRY_RUN_S=3
waitfor "[ -e $T/$c/queue/PAUSE ]" 20; sleep 2
[ -e "$T/$c/queue/PAUSE" ] && ! grep -q 'GPU free check' "$T/$c/lwf.log" && pd=1 || pd=0
grep -q "launch_when_free pid" "$T/$c/queue/PAUSE" 2>/dev/null && own=1 || own=0
qdone "$T/$c/qlog" "$J"; : > "$T/gpu_pids"; touch "$T/$c/follow"
finish $LP 40 "$T/$c/lwf.log"; rc=$?
ok "$pd" "C1 PAUSE held while the job ahead runs"
ok "$own" "C1 PAUSE carries the owner tag (S6R-3)"
ok "$([ $rc = 0 ] && V $c LAUNCH_s6s43.verdict | grep -q '^PASS' && echo 1 || echo 0)" "C1 LAUNCH verdict PASS, rc $rc"
ok "$([ ! -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C1 PAUSE released after the run holds the GPU"
ok "$(V $c RUN_s6s43.verdict | grep -q '^PASS.*attempts 1' && echo 1 || echo 0)" "C1 watcher: RUN verdict PASS after the run finished (S6R-4)"
rm -f "$T/$c/follow"; : > "$T/gpu_pids"

# C2 smoke too slow (2.5 s/step > 1.8): FAIL, PAUSE removed, nothing launched
c=c2; mkkit "$T/$c"; start $c now S6_DRY_SPS=2.5; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 's/step 2.5 > 1.8' && [ ! -e "$T/$c/queue/PAUSE" ] && [ ! -e "$T/$c/kit/runs/s6s43.pid" ] && echo 1 || echo 0)" "C2 slow smoke -> FAIL, PAUSE removed, no launch"

# C3 smoke memory 15.6 GB: FAIL
c=c3; mkkit "$T/$c"; start $c now S6_DRY_MEM=15.6; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 'peak reserved 15.6' && [ ! -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C3 memory gate -> FAIL, PAUSE removed"

# C4 missing A7 gate file: FAIL before any PAUSE
c=c4; mkkit "$T/$c"; rm -f "$T/$c/kit/gates/A7.verdict"; start $c now; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 'A7.verdict missing' && [ ! -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C4 missing A7 gate -> FAIL, PAUSE never created"

# C5 a PAUSE that already existed is left alone (happy path otherwise)
c=c5; mkkit "$T/$c"; echo main > "$T/$c/queue/PAUSE"; touch "$T/$c/follow"; start $c now S6_DRY_RUN_S=2; finish $LP 40 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 0 ] && [ "$(cat "$T/$c/queue/PAUSE")" = main ] && grep -q 'already exists' "$T/$c/lwf.log" && echo 1 || echo 0)" "C5 foreign PAUSE kept, launch PASS"
rm -f "$T/$c/follow"

# C6 the run dies before it holds the GPU (no follow marker = never on the GPU; stub exits): FAIL, PAUSE removed
c=c6; mkkit "$T/$c"; start $c now S6_DRY_RUN_S=1 S6_DRY_RUN_MODE=crash_once; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -qE 'exited before holding the GPU|not on the GPU' && [ ! -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C6 run never on the GPU -> FAIL, PAUSE removed"

# C7 SIGTERM while waiting for a free GPU: the trap removes PAUSE
c=c7; mkkit "$T/$c"; echo 99999 > "$T/gpu_pids"; start $c now
waitfor "[ -e $T/$c/queue/PAUSE ]" 20; sleep 1; [ -e "$T/$c/queue/PAUSE" ] && p7=1 || p7=0
kill -TERM $LP; wait $LP
ok "$([ $p7 = 1 ] && [ ! -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C7 SIGTERM during the wait -> PAUSE removed by the trap"
: > "$T/gpu_pids"

# C8 S6R-3: the runner logs a LAUNCH after our PAUSE (it beat us): FAIL at once, our PAUSE removed, no smoke
c=c8; mkkit "$T/$c"; qlaunch "$T/$c/qlog" "$J" 7200; touch "$T/$c/trigger"; echo 99999 > "$T/gpu_pids"
start $c after
waitfor "[ -e $T/$c/queue/PAUSE ]" 20; sleep 1.5
qdone "$T/$c/qlog" "$J"; qlaunch "$T/$c/qlog" f0_s2011_other 0           # the runner started the next queue job anyway
sleep 4; : > "$T/gpu_pids"                                              # (v1 would wait for it to end, then launch)
finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 'launched a job after our PAUSE' && [ ! -e "$T/$c/queue/PAUSE" ] && [ ! -e "$T/$c/kit/logs/s6s43_smoke300.log" ] && echo 1 || echo 0)" "C8 queue LAUNCH after our PAUSE -> FAIL, PAUSE removed, no smoke (S6R-3)"

# C9 dry-run guard: S6_DRYRUN=1 refuses the real queue path (fake HOME)
mkdir -p "$T/fakehome/queue"
HOME="$T/fakehome" S6_DRYRUN=1 S6_KIT="$T/c1/kit" S6_QUEUE_DIR="$T/fakehome/queue" bash "$LWF" --px "$PX" $([ $V2 = 1 ] && echo --now) > "$T/c9.log" 2>&1; rc=$?
ok "$([ $rc = 2 ] && grep -q 'must not use the real queue dir' "$T/c9.log" && echo 1 || echo 0)" "C9 dry-run refuses \$HOME/queue"

# C10 S6R-3: no PAUSE before the job ahead's LAUNCH line (+ delay); PAUSE right after it
c=c10; mkkit "$T/$c"; echo 99999 > "$T/gpu_pids"; touch "$T/$c/follow"
start $c after S6_PAUSE_DELAY=2 S6_DRY_RUN_S=2
sleep 4; [ -e "$T/$c/queue/PAUSE" ] && early=1 || early=0
qlaunch "$T/$c/qlog" "$J" 0; touch "$T/$c/trigger"
waitfor "[ -e $T/$c/queue/PAUSE ]" 15 && got=1 || got=0
qdone "$T/$c/qlog" "$J"; : > "$T/gpu_pids"; finish $LP 40 "$T/$c/lwf.log"; rc=$?
ok "$([ $early = 0 ] && [ $got = 1 ] && [ $rc = 0 ] && echo 1 || echo 0)" "C10 PAUSE only after the job ahead's LAUNCH line, then PASS (early $early, got $got, rc $rc)"
rm -f "$T/$c/follow"

# C11 S6R-3: main touches PAUSE while ours exists -> we never remove it
c=c11; mkkit "$T/$c"; qlaunch "$T/$c/qlog" "$J" 7200; touch "$T/$c/trigger"; echo 99999 > "$T/gpu_pids"
start $c after S6_DRY_RUN_S=2
waitfor "[ -e $T/$c/queue/PAUSE ]" 20; sleep 1.2; touch "$T/$c/queue/PAUSE"     # main's own touch (new mtime)
qdone "$T/$c/qlog" "$J"; : > "$T/gpu_pids"; touch "$T/$c/follow"; finish $LP 40 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 0 ] && [ -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C11 PAUSE touched by main is left in place (rc $rc)"
rm -f "$T/$c/follow" "$T/$c/queue/PAUSE"

# C12 S6R-3: the job ahead already ended before arming (late trigger) -> FAIL, no PAUSE, no launch
c=c12; mkkit "$T/$c"; qlaunch "$T/$c/qlog" "$J" 7200; qdone "$T/$c/qlog" "$J"; touch "$T/$c/trigger"
start $c after; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && [ ! -e "$T/$c/kit/runs/s6s43.pid" ] && [ ! -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C12 job ahead already ended -> FAIL, no launch (S6R-3)"
killrun $c

# C13 S6R-2: partial cache: check-targets pool 2,942 != A7 pool 4,349 -> FAIL before PAUSE
c=c13; mkkit "$T/$c"; start $c now S6_DRY_POOL=2942; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 'pool 2942 != A7 pool 4349' && [ ! -e "$T/$c/queue/PAUSE" ] && [ ! -e "$T/$c/kit/runs/s6s43.pid" ] && echo 1 || echo 0)" "C13 pool != A7 pool -> FAIL before PAUSE (S6R-2)"
killrun $c

# C14 S6R-2: A7 pins a pool below the floor 4,328 -> FAIL
c=c14; mkkit "$T/$c"; echo "PASS A7 test pool 4300 steps 34400" > "$T/$c/kit/gates/A7.verdict"; start $c now S6_DRY_POOL=4300; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 'pool 4300 < floor 4328' && [ ! -e "$T/$c/kit/runs/s6s43.pid" ] && echo 1 || echo 0)" "C14 pool below the 4,328 floor -> FAIL (S6R-2)"
killrun $c

# C15 S6R-2: gold 57 -> FAIL
c=c15; mkkit "$T/$c"; start $c now S6_DRY_GOLD=57; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 'gold 57 != 58' && [ ! -e "$T/$c/kit/runs/s6s43.pid" ] && echo 1 || echo 0)" "C15 gold 57 -> FAIL (S6R-2)"
killrun $c

# C16 S6R-4: the smoke hangs -> timeout -> FAIL, PAUSE removed
c=c16; mkkit "$T/$c"; start $c now S6_DRY_SMOKE_SLEEP=30 S6_SMOKE_TIMEOUT=3; finish $LP 40 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 'smoke timed out' && [ ! -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C16 hanging smoke -> timeout FAIL, PAUSE removed (S6R-4)"
killrun $c

# C17 S6R-4: the GPU never frees -> capped wait -> FAIL, PAUSE removed, IDLE_ALARM line
c=c17; mkkit "$T/$c"; echo 99999 > "$T/gpu_pids"; start $c now S6_MAX_FREE_WAIT_S=4; finish $LP 25 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 'free-GPU wait' && [ ! -e "$T/$c/queue/PAUSE" ] && grep -q 'free-GPU wait' "$T/$c/alarm" && echo 1 || echo 0)" "C17 free-GPU wait capped -> FAIL + alarm, PAUSE removed (S6R-4)"
: > "$T/gpu_pids"

# C18 S6R-4: the run stalls (alive, no output) -> STALL file + IDLE_ALARM line; STOP file + kill -> RUN FAIL, no resume
c=c18; mkkit "$T/$c"; touch "$T/$c/follow"; start $c now S6_DRY_RUN_MODE=stall S6_DRY_RUN_S=60 S6_STALL_S=3
waitfor "[ -e $T/$c/kit/gates/STALL_s6s43 ]" 30 && st=1 || st=0
grep -q 'STALL' "$T/$c/alarm" && al=1 || al=0
touch "$T/$c/kit/gates/STOP_s6s43"; killrun $c; finish $LP 20 "$T/$c/lwf.log"; rc=$?
ok "$([ $st = 1 ] && [ $al = 1 ] && V $c RUN_s6s43.verdict | grep -q '^FAIL' && [ "$(grep -c 'launch attempt' "$T/$c/lwf.log")" = 1 ] && echo 1 || echo 0)" "C18 stall -> STALL + alarm; STOP -> RUN FAIL without resume (stall $st alarm $al rc $rc) (S6R-4)"
rm -f "$T/$c/follow"

# C19 S6R-4: the run crashes once (loader timeout) -> re-PAUSE, resume with --resume, RUN PASS after 2 attempts, PAUSE released
c=c19; mkkit "$T/$c"; touch "$T/$c/follow"; start $c now S6_DRY_RUN_MODE=crash_once S6_DRY_RUN_S=2 S6_AUTO_RESUME=1
finish $LP 60 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 0 ] && V $c RUN_s6s43.verdict | grep -q '^PASS.*attempts 2' && [ "$(grep -c 'created .*PAUSE' "$T/$c/lwf.log")" = 2 ] && [ ! -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C19 crash -> re-PAUSE + resume -> RUN PASS (2 attempts), PAUSE released (rc $rc) (S6R-4)"
rm -f "$T/$c/follow"

# C20 S6R-5: an ABORT (loss-scale floor / overflow) is never resumed: RUN FAIL + alarm, one attempt
c=c20; mkkit "$T/$c"; touch "$T/$c/follow"; start $c now S6_DRY_RUN_MODE=abort S6_AUTO_RESUME=1; finish $LP 40 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c RUN_s6s43.verdict | grep -q 'S6V100 ABORT' && [ "$(grep -c 'launch attempt' "$T/$c/lwf.log")" = 1 ] && grep -q 'without a model' "$T/$c/alarm" && echo 1 || echo 0)" "C20 ABORT -> RUN FAIL + alarm, no resume (rc $rc)"
rm -f "$T/$c/follow"

# C21 S6R-5: the smoke's final loss scale 512 < 2**10 -> FAIL
c=c21; mkkit "$T/$c"; start $c now S6_DRY_SCALE=512; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 'final loss scale 512' && [ ! -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C21 smoke loss scale 512 < 1024 -> FAIL (S6R-5)"
killrun $c

# C22 (v2 only) REAL-mode command lines (S6_DRYRUN=0) with a stub python: check-targets on CPU, the smoke (S6_MAX_STEPS=300, no
#     --resume), the run through the wrapper (--resume, --out runs/s6s43, S6_EXPECT_TRAIN = the A7 pool, v2 env), rc file, watcher
c=c22; mkkit "$T/$c"; touch "$T/$c/follow"
cat > "$T/$c/stubpy" <<STUBEOF
#!/bin/bash
[ "\$1" = "-c" ] && exec $HOME/miniconda3/envs/rsna/bin/python "\$@"
echo "STUBPY argv: \$*"; env | grep -E '^(S6_|CUDA_VISIBLE)' | sort | sed 's/^/STUBPY env: /'
case " \$* " in *" --check-targets "*) echo "pool 4349 gold 58 folds [1,1,1,1,1]"; echo "check-targets OK"; exit 0;; esac
if [ -n "\${S6_MAX_STEPS:-}" ]; then echo 'S6_SMOKE {"steps": 300, "nonfinite_losses": 0, "s_per_step_median": 1.2, "peak_mem_reserved_gb": 9.0, "scaler_skipped": 3, "loss_scale": 32768.0, "loss_scale_min": 16384.0}'; exit 0; fi
o=""; prev=""; for x in "\$@"; do [ "\$prev" = "--out" ] && o="\$x"; prev="\$x"; done
mkdir -p "\$o/models"; sleep 3; echo '{}' > "\$o/models/E_prior_s43_fall.DONE.json"; echo 'S6V100 DONE {"model": "E_prior_s43_fall.pt"}'; echo "ALL DONE"; exit 0
STUBEOF
chmod +x "$T/$c/stubpy"
if [ "$V2" = 1 ]; then
  env S6_FREE_INTERVAL=1 S6_POLL=1 S6_WATCH_POLL=1 S6_PAUSE_SETTLE=1 S6_APPEAR_TIMEOUT=8 S6_KIT="$T/$c/kit" S6_QUEUE_DIR="$T/$c/queue" \
      S6_QLOG="$T/$c/qlog" S6_ALARM="$T/$c/alarm" S6_NVSMI="$T/fake_nvsmi.sh" S6_FTCOUNT_CMD="echo 0" S6_DRYRUN=0 S6_PY="$T/$c/stubpy" \
      bash "$T/$c/kit/launch_when_free.sh" --px "$PX" --now > "$T/$c/lwf.log" 2>&1 &
  LP=$!; finish $LP 40 "$T/$c/lwf.log"; rc=$?
  K="$T/$c/kit"
  ctl=$(grep -c 'STUBPY argv: -u ct_s6v100.py .*--check-targets' "$K/logs/s6s43_check_targets.log")
  sm_ok=$(grep 'STUBPY argv' "$K/logs/s6s43_smoke300.log" | grep -v -- '--resume' | grep -c -- "--out $K/runs/s6s43_smoke300")
  sm_env=$(grep -c 'STUBPY env: S6_MAX_STEPS=300' "$K/logs/s6s43_smoke300.log")
  run_argv=$(grep 'STUBPY argv' "$K/runs/s6s43.log" | grep -- '--resume' | grep -c -- "--out $K/runs/s6s43 ")
  run_env=$(grep -cE 'STUBPY env: (S6_EXPECT_TRAIN=4349|S6_ALL_DATA=1|S6_AMP=fp16|S6_MIN_SCALE=256|S6_LOADER_TIMEOUT=600)$' "$K/runs/s6s43.log")
  ok "$([ $rc = 0 ] && [ "$ctl" = 1 ] && [ "$sm_ok" = 1 ] && [ "$sm_env" = 1 ] && [ "$run_argv" = 1 ] && [ "$run_env" = 5 ] && [ "$(cat "$K/runs/s6s43.rc")" = 0 ] && V $c RUN_s6s43.verdict | grep -q '^PASS' && echo 1 || echo 0)" \
     "C22 real-mode command lines: check-targets $ctl, smoke argv $sm_ok env $sm_env, run argv $run_argv env $run_env/5, rc file, RUN PASS (rc $rc)"
else
  ok 0 "C22 real-mode command lines (v2 interface only)"
fi
rm -f "$T/$c/follow"

# C23 soup option bookkeeping: A7 pins head_seed 42 but S6_HEAD_SEED_RUN is unset -> FAIL before PAUSE; both 42 -> pre-flight OK
c=c23; mkkit "$T/$c"; echo "PASS A7 test pool 4349 steps 34784 head_seed 42" > "$T/$c/kit/gates/A7.verdict"; start $c now; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 1 ] && V $c LAUNCH_s6s43.verdict | grep -q 'head-init seed' && [ ! -e "$T/$c/queue/PAUSE" ] && echo 1 || echo 0)" "C23 A7 head_seed 42 without S6_HEAD_SEED_RUN -> FAIL before PAUSE (rc $rc)"
c=c24; mkkit "$T/$c"; echo "PASS A7 test pool 4349 steps 34784 head_seed 42" > "$T/$c/kit/gates/A7.verdict"; touch "$T/$c/follow"; start $c now S6_HEAD_SEED_RUN=42 S6_DRY_RUN_S=1; finish $LP 30 "$T/$c/lwf.log"; rc=$?
ok "$([ $rc = 0 ] && grep -q 'head-init seed 42 (= A7)' "$T/$c/lwf.log" && echo 1 || echo 0)" "C24 A7 head_seed 42 + S6_HEAD_SEED_RUN=42 -> launch PASS (rc $rc)"
rm -f "$T/$c/follow"

echo "LAUNCH_TESTS $([ $PASS = $N ] && echo PASS || echo FAIL) $PASS/$N  (launcher $(sha256sum "$LWF" | cut -c1-12), v2 interface $V2, logs in $T)"
