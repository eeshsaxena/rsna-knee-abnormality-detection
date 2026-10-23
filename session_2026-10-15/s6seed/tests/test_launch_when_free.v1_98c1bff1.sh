#!/bin/bash
# Dry-run tests of launch_when_free.sh: a FAKE queue dir, a fake nvidia-smi (prints the pids in $T/gpu_pids), stub smoke / stub run
# (S6_DRYRUN=1).  Never touches ~/queue or the GPU.  Prints one line per check and LAUNCH_TESTS PASS n/n.
set -u
SRC="${1:-$HOME/s6seed}"
PX="${2:-$HOME/s6seed/tests/px_fixture}"
T="$(mktemp -d /tmp/s6lwf.XXXXXX)"
PASS=0; N=0
ok() { N=$((N+1)); if [ "$1" = "1" ]; then PASS=$((PASS+1)); echo "PASS $2"; else echo "FAIL $2"; fi; }
mkkit() {   # $1 = kit dir
  mkdir -p "$1/gates" "$1/pretrained" "$1/logs" "$1/runs"
  cp "$SRC/ct_s6v100.py" "$1/"; sha256sum "$1/ct_s6v100.py" | cut -c1-64 > "$1/RECEIPT_ct_s6v100.sha256"
  ln -sf "$SRC/pretrained/convnext_small_in22k_ft_in1k.safetensors" "$1/pretrained/"
  cp "$SRC/targets_s6seed.csv" "$1/"
  echo "PASS test" > "$1/gates/PX_PARITY.verdict"; echo "PASS test" > "$1/gates/A7.verdict"
}
cat > "$T/fake_nvsmi.sh" <<EOF
#!/bin/bash
cat "$T/gpu_pids" 2>/dev/null
EOF
chmod +x "$T/fake_nvsmi.sh"
run() {   # $1 = case name, rest = extra env; runs the script in the background, returns its pid in LWF_PID
  local name="$1"; shift
  mkdir -p "$T/$name/queue"; mkkit "$T/$name/kit"
  env "$@" S6_KIT="$T/$name/kit" S6_QUEUE_DIR="$T/$name/queue" S6_NVSMI="$T/fake_nvsmi.sh" S6_FTCOUNT_CMD="echo 0" S6_FREE_INTERVAL=1 S6_DRYRUN=1 \
      S6_APPEAR_TIMEOUT=8 bash "$SRC/launch_when_free.sh" --px "$PX" > "$T/$name/lwf.log" 2>&1 &
  LWF_PID=$!
}
gpu_follow() {   # simulate: the launched stub takes the GPU (its pid appears in gpu_pids) once its pid file exists
  local kit="$1"; for i in $(seq 1 60); do [ -s "$kit/runs/s6s43.pid" ] && { cat "$kit/runs/s6s43.pid" > "$T/gpu_pids"; return; }; sleep 0.5; done
}

# C1 happy path: GPU busy for ~4 s (job ahead), then free; smoke PASS; launch; PAUSE removed
echo 99999 > "$T/gpu_pids"
run c1 S6_DRY_RUN_S=30
for i in $(seq 1 120); do [ -e "$T/c1/queue/PAUSE" ] && break; sleep 0.5; done
sleep 3; [ -e "$T/c1/queue/PAUSE" ] && ! grep -q 'GPU free check' "$T/c1/lwf.log" && p_during=1 || p_during=0
: > "$T/gpu_pids"
gpu_follow "$T/c1/kit"
wait $LWF_PID; rc=$?
ok "$p_during" "C1 PAUSE present while the job ahead holds the GPU"
ok "$([ $rc = 0 ] && grep -q '^PASS' "$T/c1/kit/gates/LAUNCH_s6s43.verdict" && echo 1 || echo 0)" "C1 verdict PASS, rc $rc"
ok "$([ ! -e "$T/c1/queue/PAUSE" ] && echo 1 || echo 0)" "C1 PAUSE removed after the run holds the GPU"
ok "$(grep -q 'GPU free check 2/2' "$T/c1/lwf.log" && grep -q 'smoke gate: PASS' "$T/c1/lwf.log" && echo 1 || echo 0)" "C1 two free checks + smoke gate PASS logged"
kill "$(cat "$T/c1/kit/runs/s6s43.pid")" 2>/dev/null; : > "$T/gpu_pids"

# C2 smoke too slow (2.5 s/step > 1.8): FAIL, PAUSE removed, nothing launched
run c2 S6_DRY_SPS=2.5
wait $LWF_PID; rc=$?
ok "$([ $rc = 1 ] && grep -q 's/step 2.5 > 1.8' "$T/c2/kit/gates/LAUNCH_s6s43.verdict" && echo 1 || echo 0)" "C2 slow smoke -> FAIL verdict"
ok "$([ ! -e "$T/c2/queue/PAUSE" ] && [ ! -e "$T/c2/kit/runs/s6s43.pid" ] && echo 1 || echo 0)" "C2 PAUSE removed, no launch"

# C3 smoke memory 15.6 GB: FAIL
run c3 S6_DRY_MEM=15.6
wait $LWF_PID; rc=$?
ok "$([ $rc = 1 ] && grep -q 'peak reserved 15.6' "$T/c3/kit/gates/LAUNCH_s6s43.verdict" && [ ! -e "$T/c3/queue/PAUSE" ] && echo 1 || echo 0)" "C3 memory gate -> FAIL, PAUSE removed"

# C4 missing A7 gate file: FAIL before any PAUSE
mkdir -p "$T/c4b/queue"; mkkit "$T/c4b/kit"; rm -f "$T/c4b/kit/gates/A7.verdict"
S6_KIT="$T/c4b/kit" S6_QUEUE_DIR="$T/c4b/queue" S6_NVSMI="$T/fake_nvsmi.sh" S6_FTCOUNT_CMD="echo 0" S6_FREE_INTERVAL=1 S6_DRYRUN=1 bash "$SRC/launch_when_free.sh" --px "$PX" > "$T/c4b/lwf.log" 2>&1; rc=$?
ok "$([ $rc = 1 ] && grep -q 'A7.verdict missing' "$T/c4b/kit/gates/LAUNCH_s6s43.verdict" && [ ! -e "$T/c4b/queue/PAUSE" ] && echo 1 || echo 0)" "C4 missing A7 gate -> FAIL, PAUSE never created"

# C5 a PAUSE that already existed is left alone (happy path otherwise)
mkdir -p "$T/c5/queue"; echo main > "$T/c5/queue/PAUSE"
run c5 S6_DRY_RUN_S=30
gpu_follow "$T/c5/kit"; wait $LWF_PID; rc=$?
ok "$([ $rc = 0 ] && [ "$(cat "$T/c5/queue/PAUSE")" = main ] && grep -q 'already exists' "$T/c5/lwf.log" && echo 1 || echo 0)" "C5 foreign PAUSE kept, launch PASS"
kill "$(cat "$T/c5/kit/runs/s6s43.pid")" 2>/dev/null; : > "$T/gpu_pids"

# C6 the run dies before it holds the GPU: FAIL, PAUSE removed
run c6 S6_DRY_RUN_S=1
wait $LWF_PID; rc=$?
ok "$([ $rc = 1 ] && grep -q 'exited before holding the GPU' "$T/c6/kit/gates/LAUNCH_s6s43.verdict" && [ ! -e "$T/c6/queue/PAUSE" ] && echo 1 || echo 0)" "C6 run died early -> FAIL, PAUSE removed"

# C7 SIGTERM while waiting for a free GPU: the trap removes PAUSE
echo 99999 > "$T/gpu_pids"
run c7
for i in $(seq 1 120); do [ -e "$T/c7/queue/PAUSE" ] && break; sleep 0.5; done
sleep 2; [ -e "$T/c7/queue/PAUSE" ] && p7=1 || p7=0
kill -TERM $LWF_PID; wait $LWF_PID
ok "$([ $p7 = 1 ] && [ ! -e "$T/c7/queue/PAUSE" ] && echo 1 || echo 0)" "C7 SIGTERM during the wait -> PAUSE removed by the trap"
: > "$T/gpu_pids"

# C8 --pause-after: no PAUSE until the trigger file exists
mkdir -p "$T/c8b/queue"; mkkit "$T/c8b/kit"
S6_KIT="$T/c8b/kit" S6_QUEUE_DIR="$T/c8b/queue" S6_NVSMI="$T/fake_nvsmi.sh" S6_FTCOUNT_CMD="echo 0" S6_FREE_INTERVAL=1 S6_DRYRUN=1 S6_DRY_RUN_S=30 \
  bash "$SRC/launch_when_free.sh" --px "$PX" --pause-after "$T/c8b/trigger" > "$T/c8b/lwf.log" 2>&1 &
P8=$!
sleep 4; [ -e "$T/c8b/queue/PAUSE" ] && early=1 || early=0
touch "$T/c8b/trigger"; gpu_follow "$T/c8b/kit"; wait $P8; rc=$?
ok "$([ $early = 0 ] && [ $rc = 0 ] && grep -q 'created' "$T/c8b/lwf.log" && echo 1 || echo 0)" "C8 --pause-after waits for the trigger, then PASS"
kill "$(cat "$T/c8b/kit/runs/s6s43.pid")" 2>/dev/null; : > "$T/gpu_pids"

# C9 dry-run guard: S6_DRYRUN=1 refuses the real queue path (fake HOME, so nothing real is resolved)
mkdir -p "$T/fakehome/queue"
HOME="$T/fakehome" S6_DRYRUN=1 S6_KIT="$T/c1/kit" S6_QUEUE_DIR="$T/fakehome/queue" bash "$SRC/launch_when_free.sh" --px "$PX" > "$T/c9.log" 2>&1; rc=$?
ok "$([ $rc = 2 ] && grep -q 'must not use the real queue dir' "$T/c9.log" && echo 1 || echo 0)" "C9 dry-run refuses \$HOME/queue"

echo "LAUNCH_TESTS $([ $PASS = $N ] && echo PASS || echo FAIL) $PASS/$N  (logs in $T)"
