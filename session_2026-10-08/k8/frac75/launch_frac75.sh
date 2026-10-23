#!/bin/bash
# launch_frac75.sh (k8/frac75, 2026-10-07). NOT ARMED, NOT A WAITING CHAIN. Default = print the exact commands (DRY).
#   bash ~/novel/frac75/launch_frac75.sh probe            # print the fold-0 probe command
#   bash ~/novel/frac75/launch_frac75.sh folds            # print the folds 1-4 commands
#   LAUNCH=1 bash ~/novel/frac75/launch_frac75.sh probe   # run it NOW, foreground (wrap in setsid nohup ... &)
# With LAUNCH=1 it refuses (exit 4, never waits, never kills) when PAUSE_CHAIN exists, any python holds the GPU
# (nvidia-smi error = busy), the TTR288 chain bash (auto_after_ttr288.sh) is alive, any `python ... fleet_train.py`
# process is alive (a trainer that has not opened CUDA yet), ~/queue/queue_runner.sh is alive (then the queue is THE
# launcher: its queue.txt Q1/Q2 lines carry this exact argv), or the training freeze (2026-10-20 12:00 IST) has passed.
# It skips a fold whose out dir OR out .log exists (never truncates another launcher's log).
# "folds" also refuses unless ~/novel/frac75/out/gate_f0_s1043_ttr_frac75.json says the REAL probe dir PASSED, not dry
# (FORCE=1 overrides, operator only). A queue runner may instead source argv_frac75.sh and call frac75_argv F itself.
# grad_ckpt 0 folds: one early-failure fallback (< 45 min, e.g. CUDA OOM) to --grad_ckpt 1, out dir moved aside first.
# (10-07 verify pass: added the fleet_train / queue_runner / freeze guards, the .log skip and the gate json "run" check.)
source $HOME/novel/frac75/argv_frac75.sh
PY=~/miniconda3/envs/rsna/bin/python; LOG=$HOME/novel/frac75/launch_frac75.log
say(){ echo "[frac75 $(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
gpu_python(){ local o; o=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null) || { echo 99; return; }; echo "$o" | grep -c python; }
TRAIN_PROC='[^ ]*python[0-9.]*( -[^ ]+)* ([^ ]*/)?fleet_train[.]py( .*)?'   # == ~/queue/queue_runner.sh DEF_TRAIN_PROC
alive_x(){ pgrep -x -f "$1" > /dev/null 2>&1; [ $? -ne 1 ]; }   # whole-cmdline match; a pgrep error counts as alive
alive_f(){ pgrep -f "$1" > /dev/null 2>&1; [ $? -ne 1 ]; }      # substring match; a pgrep error counts as alive
FREEZE=$(date -d 2026-10-20T12:00:00+05:30 +%s)
busy_reason(){
  [ -e $HOME/fleet_runs/PAUSE_CHAIN ] && { echo "PAUSE_CHAIN present"; return; }
  local g; g=$(gpu_python); [ "$g" -ne 0 ] && { echo "GPU busy (python compute apps: $g; 99 = nvidia-smi error)"; return; }
  alive_f "bash .*auto_after_ttr288\.sh" && { echo "TTR288 chain auto_after_ttr288.sh alive"; return; }
  alive_x "$TRAIN_PROC" && { echo "a python fleet_train.py process is alive"; return; }
  alive_f "bash .*queue_runner\.sh" && { echo "~/queue/queue_runner.sh is alive: it is the launcher (queue.txt)"; return; }
  [ "$(date +%s)" -ge "$FREEZE" ] && { echo "past the training freeze 2026-10-20 12:00 IST"; return; }
  echo free
}
MODE=${1:-probe}
case $MODE in probe) FOLDS="0";; folds) FOLDS="1 2 3 4";; *) echo "usage: $0 probe|folds"; exit 2;; esac
if [ "$MODE" = folds ] && [ "${LAUNCH:-0}" = 1 ] && [ "${FORCE:-0}" != 1 ]; then
  G=$HOME/novel/frac75/out/gate_f0_s1043_ttr_frac75.json
  { grep -q '"pass_": true' $G && grep -q '"dry": false' $G && grep -qF "\"run\": \"$HOME/fleet_runs/f0_s1043_ttr_frac75\"" $G; } 2>/dev/null \
    || { say "folds refused: the probe gate has not PASSED (run score_frac75.py first)"; exit 4; }
fi
cd ~/fast_build/code2 || exit 1
for F in $FOLDS; do
  frac75_argv $F; OUT=$FRAC75_OUT
  CMD="$PY fleet_train.py ${FRAC75_ARGV[*]}"
  if [ "${LAUNCH:-0}" != 1 ]; then echo "cd ~/fast_build/code2 && $CMD > $OUT.log 2>&1"; continue; fi
  R=$(busy_reason); [ "$R" != free ] && { say "f$F refused: $R"; exit 4; }
  { [ -e $OUT ] || [ -e $OUT.log ]; } && { say "f$F: $OUT or its .log exists, skipping (never overwritten)"; continue; }
  say "f$F: launching -> $OUT (trainer md5 $(md5sum fleet_train.py | cut -c1-12), labels md5 $(md5sum $FRAC75_LABELS | cut -c1-12))"
  T0=$(date +%s); $PY fleet_train.py "${FRAC75_ARGV[@]}" > $OUT.log 2>&1; RC=$?; DT=$(( ($(date +%s) - T0) / 60 ))
  say "f$F: exited rc=$RC after ${DT} min ($(tail -n 1 $OUT.log | cut -c1-120))"
  if [ $RC -ne 0 ] && [ $DT -lt 45 ] && [ "$F" != 0 ]; then
    AS=${OUT}_fastfail_$(date +%m%d_%H%M); [ -e $OUT ] && mv $OUT $AS; mv $OUT.log $AS.log
    R=$(busy_reason); [ "$R" != free ] && { say "f$F: early failure moved aside to $AS; fallback refused ($R)"; exit 4; }
    say "f$F: early failure moved aside to $AS; re-running with --grad_ckpt 1"
    $PY fleet_train.py "${FRAC75_ARGV[@]}" --grad_ckpt 1 > $OUT.log 2>&1; say "f$F: fallback exited rc=$? ($(tail -n 1 $OUT.log | cut -c1-120))"
  fi
done
