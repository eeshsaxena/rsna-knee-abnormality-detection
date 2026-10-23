#!/bin/bash
# auto_f2probe.sh (2026-10-05): family-2 (F2) decorrelation probe, JUMP_PLAN_10-05 sec 2 action 4 + sec 3, HANDOFF TASK 4.
# NOT started by its author. The operator starts it (any time before fold 2 ends):
#   setsid nohup bash ~/auto_f2probe.sh > /dev/null 2>&1 < /dev/null &
# Status only (launches nothing, writes nothing):   bash ~/auto_f2probe.sh --check
# Print the exact trainer argv it would use:        bash ~/auto_f2probe.sh --print-args ev2s|r50
# What it does:
#  1) waits (sleep F2P_POLL = 120 s) until the fast chain is finished (its log has a "chain done" line after its last
#     "chain start" line, OR no auto_folds_fast.sh bash process is alive), the old chain auto_folds.sh is not alive,
#     ~/fleet_runs/PAUSE_CHAIN does not exist, and nvidia-smi lists no python compute app (an nvidia-smi error counts as
#     busy), on two checks 120 s apart; then takes ~/jump2/gpu.lock without blocking (the lock auto_round2.sh uses) and
#     holds it until the probe exits;
#  2) runs the G-1 preflight ~/fleet2_code/f2_preflight.py (a few GPU minutes, writes nothing; its output goes to
#     ~/fleet_runs/f2probe_preflight_<MMDD_HHMMSS>.log) and reads its last "PREFLIGHT <backbone>=<VERDICT> ..." line;
#  3) chooses EV2S (tf_efficientnetv2_s.in21k_ft_in1k, seed 142, lr 2e-4) if its case is PASS, else R50
#     (resnet50.a1_in1k, seed 242, lr 3e-4) if its case is PASS, else logs ABORT and launches nothing;
#  4) launches the fold-0 probe ALONE, in the foreground, with the family-2 path of ~/bigjump_build/code/fleet_train.py
#     (--f2 1, F1 recipe defaults, pure steven_v4 labels, 10 epochs) -> ~/fleet_runs/f0_s<S>_f2v4 (+ .log); refuses if
#     that out dir or its log already exists; re-checks pause / GPU / deadline right before the launch;
#  5) if the EV2S run dies with the F2-T5 fp16 watchdog (rc 2 and a "!! F2-T5" line) and R50 passed the same preflight,
#     launches the R50 probe once (same guards). F2P_WATCHDOG_FALLBACK=0 turns this off.
# Never kills or signals any process (it only reads ps / pgrep / nvidia-smi). Gives up without launching anything once
# F2P_DEADLINE (default 2026-10-06 12:00 IST) is reached. Every decision goes to ~/fleet_runs/auto_f2probe.log.
# Recipe (fleet2/SCHEDULE.md 4c, FAMILY2.md, reviewed): --lr is the backbone LR of the reviewed recipe (EV2S 2e-4,
# R50 3e-4; the trainer default 1e-4 is the ConvNeXt one, so it must be passed); everything else is the trainer default
# (head_lr 1e-3, wd 0.05, ema 0.999, bs 2 x accum 4, drop_path 0.1, dropout 0.2, aug light, model f1, grad ckpt on).
# --freeze_bn 1 / --channels_last 1 are the defaults, passed explicitly. No --init_ckpt (timm ImageNet init). R50 runs
# alone here, so bs 2 (the --bs 1 --accum 8 rule is only for sharing the GPU).
# Env overrides exist for the CPU dry test only (F2P_*); the operator passes none.
set -u
PY=${F2P_PY:-$HOME/miniconda3/envs/rsna/bin/python}
FR=${F2P_FR:-$HOME/fleet_runs}
LOG=${F2P_LOG:-$FR/auto_f2probe.log}
CHAIN_LOG=${F2P_CHAIN_LOG:-$FR/auto_folds_fast.log}
DEF_CHAIN_RE='^(/usr)?(/bin/)?(ba)?sh ([^ ]*/)?auto_folds_fast\.sh$'
DEF_OLD_RE='^(/usr)?(/bin/)?(ba)?sh ([^ ]*/)?auto_folds\.sh$'
CHAIN_RE=${F2P_CHAIN_RE:-$DEF_CHAIN_RE}
OLD_RE=${F2P_OLD_RE:-$DEF_OLD_RE}
PAUSE=${F2P_PAUSE:-$FR/PAUSE_CHAIN}
LOCK=${F2P_LOCK:-$HOME/jump2/gpu.lock}
INST=${F2P_INSTANCE:-$HOME/.auto_f2probe.instance}
CODE=${F2P_CODE:-$HOME/bigjump_build/code}
PFDIR=${F2P_PFDIR:-$HOME/fleet2_code}
STORE=${F2P_STORE:-$HOME/fleet_store}
RT=${F2P_RT:-$HOME/rsna_train}
POLL=${F2P_POLL:-120}
PF_POLL=${F2P_PF_POLL:-5}
PF_LIMIT=${F2P_PF_LIMIT:-2700}          # preflight wall limit (s); past it: give up, launch nothing, kill nothing
HEARTBEAT=${F2P_HEARTBEAT:-30}         # unchanged-wait reminder every N polls (30 x 120 s = 1 h)
DEF_DEADLINE='2026-10-06 12:00 +0530'  # 10-06 12:00 IST
DEADLINE=${F2P_DEADLINE:-$DEF_DEADLINE}
FALLBACK=${F2P_WATCHDOG_FALLBACK:-1}

EV2S_BB=tf_efficientnetv2_s.in21k_ft_in1k; EV2S_SEED=142; EV2S_LR=2e-4
R50_BB=resnet50.a1_in1k;                    R50_SEED=242;  R50_LR=3e-4
CNX_CASE=convnext_tiny.fb_in22k_ft_in1k:1e-4    # ConvNeXt-T reference case (G0 ratio), not a candidate
N_TRAIN=3466                                    # "train 3466" of the fold-0 v4 runs (f0_s1042_ttr.log line 1)

cfg(){ case "${1:-}" in
         ev2s) BB=$EV2S_BB; SEED=$EV2S_SEED; LR=$EV2S_LR;;
         r50)  BB=$R50_BB;  SEED=$R50_SEED;  LR=$R50_LR;;
         *) return 1;;
       esac; OUT=$FR/f0_s${SEED}_f2v4; }
targs(){ cfg "$1" || return 1
  echo "--f2 1 --backbone $BB --lr $LR --seed $SEED --freeze_bn 1 --channels_last 1 --store $STORE" \
       "--labels $RT/llm_labels_v4_blend.csv --folds $RT/folds5.csv --fold 0 --exclude $RT/gold74_eval_set.csv" \
       "--eval_labels $RT/llm_labels_v4_blend.csv --gold $RT/gold74_eval_set.csv --epochs 10 --workers 10 --out $OUT"; }
pfargs(){ echo "f2_preflight.py --store $STORE --n_train $N_TRAIN --cases $CNX_CASE $EV2S_BB:$EV2S_LR $R50_BB:$R50_LR"; }

MODE=${1:-run}
case "$MODE" in
  --print-args) targs "${2:-}" || { echo "usage: $0 --print-args ev2s|r50" >&2; exit 2; }; exit 0;;
  --check|run) ;;
  *) echo "usage: $0 [--check | --print-args ev2s|r50]" >&2; exit 2;;
esac
DL_EPOCH=$(date -d "$DEADLINE" +%s 2>/dev/null) || { echo "bad deadline '$DEADLINE'" >&2; exit 2; }

say(){ local m; m="[auto-f2probe $(date '+%m-%d %H:%M')] $*"; echo "$m" >> "$LOG"; echo "$m"; }
gpu_python(){ local o; o=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null) || { echo 99; return; }
              echo "$o" | grep -c python; }        # same rule as auto_folds_fast.sh: nvidia-smi error = busy (99)
gpu_mem(){ local o; o=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -n 1); echo "${o:-?}"; }
chain_pids(){ pgrep -f "$CHAIN_RE" | tr '\n' ' ' | sed 's/ $//'; }
old_pids(){ pgrep -f "$OLD_RE" | tr '\n' ' ' | sed 's/ $//'; }
chain_alive(){ [ -n "$(chain_pids)" ]; }
chain_done_logged(){ [ -f "$CHAIN_LOG" ] && awk '/\] chain start/{d=0} /\] chain done[[:space:]]*$/{d=1} END{exit !d}' "$CHAIN_LOG"; }
paused(){ [ -e "$PAUSE" ]; }
past_deadline(){ [ "$(date +%s)" -ge "$DL_EPOCH" ]; }
sha16(){ sha256sum "$1" 2>/dev/null | cut -c1-16; }
running(){ local st; st=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$st" ] && [ "${st:0:1}" != Z ]; }   # a reaped-later zombie counts as exited
wait_reason(){                                    # empty = all clear
  local r="" g p
  if ! chain_done_logged && chain_alive; then r="$r fast chain running (pid $(chain_pids)), no 'chain done' yet;"; fi
  p=$(old_pids); [ -n "$p" ] && r="$r old chain auto_folds.sh alive (pid $p);"
  paused && r="$r $PAUSE present;"
  g=$(gpu_python)
  if [ "$g" = 99 ]; then r="$r nvidia-smi error (counted as busy);"; elif [ "$g" != 0 ]; then r="$r python on GPU: $g;"; fi
  echo "$r"; }

# ------------------------------------------------ status mode ------------------------------------------------
if [ "$MODE" = --check ]; then
  c(){ echo "[auto-f2probe check] $*"; }
  c "fast chain log $CHAIN_LOG: 'chain done' after its last 'chain start': $(chain_done_logged && echo yes || echo no)"
  c "fast chain bash: $(chain_alive && echo "alive (pid $(chain_pids))" || echo gone) | old chain: $([ -n "$(old_pids)" ] && echo "alive (pid $(old_pids))" || echo gone)"
  c "$PAUSE: $(paused && echo present || echo absent) | python apps on GPU: $(gpu_python) | GPU mem used/total: $(gpu_mem)"
  r=$(wait_reason); c "a run would wait now: ${r:-no (all clear; it confirms once more ${POLL}s later, then takes the lock)}"
  if [ -e "$LOCK" ]; then c "GPU lock $LOCK: $(flock -n "$LOCK" true 2>/dev/null && echo free || echo HELD)"; else c "GPU lock $LOCK: absent"; fi
  if [ -e "$INST" ]; then c "launcher instance: $(flock -n "$INST" true 2>/dev/null && echo 'none running' || echo 'one is RUNNING')"; else c "launcher instance: none started yet"; fi
  for k in ev2s r50; do cfg $k; c "$k -> $OUT: $({ [ -e "$OUT" ] || [ -e "$OUT.log" ]; } && echo 'EXISTS (would be refused)' || echo 'free')"; done
  c "deadline $DEADLINE: $(past_deadline && echo PASSED || echo 'not reached')"
  c "preflight: cd $PFDIR && $PY $(pfargs)"
  c "ev2s: cd $CODE && $PY fleet_train.py $(targs ev2s)"
  c "r50 : cd $CODE && $PY fleet_train.py $(targs r50)"
  exit 0
fi

# ------------------------------------------------ run mode ------------------------------------------------
[ -d "$FR" ] || { echo "no $FR" >&2; exit 2; }
exec 8>>"$INST" || { say "ABORT: cannot open instance lock $INST"; exit 2; }
flock -n 8 || { say "another auto_f2probe.sh is already running (instance lock $INST held); this one exits, nothing launched"; exit 1; }
for f in "$CODE/fleet_train.py" "$CODE/fleet_model_f2.py" "$PFDIR/f2_preflight.py"; do
  [ -f "$f" ] || { say "ABORT: missing $f; nothing launched"; exit 2; }
done
[ -x "$PY" ] || { say "ABORT: python $PY not executable; nothing launched"; exit 2; }
{ [ -e "$LOCK" ] || [ -w "$(dirname "$LOCK")" ]; } || { say "ABORT: cannot create GPU lock $LOCK; nothing launched"; exit 2; }
exec 9>>"$LOCK" || { say "ABORT: cannot open GPU lock $LOCK; nothing launched"; exit 2; }
say "start pid $$ | deadline $DEADLINE | poll ${POLL}s | GPU lock $LOCK | trainer $CODE/fleet_train.py sha $(sha16 "$CODE/fleet_train.py") | preflight $PFDIR/f2_preflight.py sha $(sha16 "$PFDIR/f2_preflight.py") | watchdog fallback $FALLBACK"
say "plan A (EV2S PASS): cd $CODE && $PY fleet_train.py $(targs ev2s)"
say "plan B (else R50 PASS): cd $CODE && $PY fleet_train.py $(targs r50)"
cmp -s "$CODE/fleet_model_f2.py" "$PFDIR/fleet_model_f2.py" || say "WARNING: $CODE/fleet_model_f2.py differs from $PFDIR/fleet_model_f2.py (the preflight builds the model from the latter)"
nex=0
for k in ev2s r50; do cfg $k
  if [ -e "$OUT" ] || [ -e "$OUT.log" ]; then nex=$((nex + 1)); say "note: $OUT (or its .log) already exists, so choice $k would be refused"; fi
done
[ $nex -eq 2 ] && { say "REFUSE: both probe out dirs already exist; nothing to do, nothing launched"; exit 3; }

CHOICE=""; EV=""; RV=""; LSTATE=""; RC=""
run_preflight(){                                 # sets CHOICE/EV/RV, or exits (ABORT)
  local pid t0 rc pfl
  PFLOG=$FR/f2probe_preflight_$(date +%m%d_%H%M%S).log
  say "preflight: start, GPU mem used/total $(gpu_mem) | cd $PFDIR && $PY $(pfargs) > $PFLOG"
  ( cd "$PFDIR" && export PYTHONDONTWRITEBYTECODE=1 && exec "$PY" $(pfargs) ) > "$PFLOG" 2>&1 < /dev/null 8>&- &
  pid=$!; t0=$(date +%s)
  while running "$pid"; do
    if [ $(( $(date +%s) - t0 )) -ge "$PF_LIMIT" ]; then
      say "ABORT: preflight pid $pid still running after ${PF_LIMIT} s; launching nothing and NOT killing it (it holds the GPU lock until it exits); exiting"
      exit 6
    fi
    sleep "$PF_POLL"
  done
  wait "$pid"; rc=$?
  say "preflight: exited rc=$rc after $(( $(date +%s) - t0 )) s"
  grep -a '^GPU ' "$PFLOG" | head -n 1 | tr -d '\r' | while IFS= read -r l; do say "preflight: $l"; done
  grep -aE '^\[(PASS|FAIL|OOM|ERROR)\] \{' "$PFLOG" | tr -d '\r' | while IFS= read -r l; do
    say "preflight: $("$PY" -c '
import sys, json
v, _, j = sys.argv[1].partition(" ")
r = json.loads(j)
ks = ("backbone", "fp32_amax", "fp32_amax_at", "fp16_inf_modules", "fp16_logits_finite", "skipped_after_first",
      "bn_stats_unchanged", "peak_reserved_gb", "s_per_step", "proj_train_min_per_epoch", "error")
print(v + " " + " ".join(f"{k}={r[k]}" for k in ks if k in r))' "$l" 2>/dev/null || printf '%s' "${l:0:300}")"
  done
  pfl=$(grep -a '^PREFLIGHT ' "$PFLOG" | tail -n 1 | tr -d '\r')
  verdict(){ printf '%s\n' "$pfl" | awk -v k="$1" '{for (i = 2; i <= NF; i++) { j = index($i, "="); if (j && substr($i, 1, j - 1) == k) { print substr($i, j + 1); exit } } }'; }
  EV=$(verdict "$EV2S_BB"); RV=$(verdict "$R50_BB")
  say "preflight: verdicts EV2S=${EV:-missing} R50=${RV:-missing} | ${pfl:-no PREFLIGHT line in $PFLOG}"
  [ $rc -ne 0 ] && [ -n "$pfl" ] && say "preflight: WARNING rc $rc although a PREFLIGHT line was printed; going by the line"
  if [ "$EV" = PASS ]; then CHOICE=ev2s; say "choose: EV2S passed -> $EV2S_BB seed $EV2S_SEED lr $EV2S_LR"
  elif [ "$RV" = PASS ]; then CHOICE=r50; say "choose: EV2S ${EV:-missing}, R50 passed -> $R50_BB seed $R50_SEED lr $R50_LR"
  else say "ABORT: no family-2 backbone passed the preflight (EV2S=${EV:-missing} R50=${RV:-missing}, rc $rc); nothing launched. Read $PFLOG"; exit 4
  fi
}
launch(){                                        # sets LSTATE (ran|backoff|refused|deadline) and RC
  local r a t0
  cfg "$1"; LSTATE=""; RC=""
  if past_deadline; then say "GIVE UP: deadline $DEADLINE reached before launching $1; nothing launched"; LSTATE=deadline; return; fi
  if [ -e "$OUT" ] || [ -e "$OUT.log" ]; then say "REFUSE: $OUT (or $OUT.log) already exists; not launching $1, never overwriting a run"; LSTATE=refused; return; fi
  r=$(wait_reason)
  if [ -n "$r" ]; then say "launch of $1 deferred:$r"; LSTATE=backoff; return; fi
  a=$(targs "$1")
  say "LAUNCH $1: $BB seed $SEED lr $LR fold 0, alone on the GPU (mem used/total $(gpu_mem)) | cd $CODE && $PY fleet_train.py $a > $OUT.log 2>&1"
  t0=$(date +%s)
  ( cd "$CODE" && export PYTHONDONTWRITEBYTECODE=1 && exec "$PY" fleet_train.py $a ) > "$OUT.log" 2>&1 < /dev/null 8>&-
  RC=$?
  say "probe $1 exited rc=$RC after $(( ($(date +%s) - t0) / 60 )) min | last line: $(tail -n 1 "$OUT.log" 2>/dev/null | cut -c1-160)"
  LSTATE=ran
}

clear=0; last=""; n=0
while true; do
  if past_deadline; then say "GIVE UP: deadline $DEADLINE reached and nothing was launched; exiting"; exit 5; fi
  r=$(wait_reason)
  if [ -n "$r" ]; then
    { [ "$r" != "$last" ] || [ $(( n % HEARTBEAT )) -eq 0 ]; } && say "waiting:$r"
    [ "$r" != "$last" ] && n=0
    last=$r; n=$((n + 1)); clear=0; sleep "$POLL"; continue
  fi
  if [ $clear -eq 0 ]; then
    clear=1; last=""; n=0
    say "all clear once (chain finished: done-line $(chain_done_logged && echo yes || echo no), bash $(chain_alive && echo alive || echo gone); no pause; no python on GPU); confirming in ${POLL}s"
    sleep "$POLL"; continue
  fi
  if ! flock -n 9; then
    [ "$last" != LOCK ] && say "waiting: GPU lock $LOCK is held by another launcher (e.g. auto_round2.sh)"
    last=LOCK; clear=0; sleep "$POLL"; continue
  fi
  say "GPU lock taken; all clear on two checks ${POLL}s apart"
  if [ -z "$CHOICE" ]; then
    r=$(wait_reason)
    if [ -n "$r" ]; then say "backing off inside the lock:$r"; flock -u 9; clear=0; last=""; sleep "$POLL"; continue; fi
    run_preflight
  fi
  launch "$CHOICE"
  case "$LSTATE" in
    backoff) flock -u 9; say "GPU lock released; waiting again (choice $CHOICE kept, no new preflight)"; clear=0; last=""; sleep "$POLL"; continue;;
    refused) exit 3;;
    deadline) exit 5;;
  esac
  break
done

if [ "$RC" -eq 0 ]; then
  say "done: $CHOICE probe finished OK ($OUT). Next (operator): gate F2-G, JUMP_PLAN_10-05 sec 3. Nothing else launched."
  exit 0
fi
if [ "$CHOICE" = ev2s ] && [ "$RC" -eq 2 ] && grep -aq '^!! F2-T5 fp16 watchdog' "$OUT.log"; then
  if [ "$FALLBACK" != 1 ] || [ "$RV" != PASS ]; then
    say "EV2S stopped by the F2-T5 fp16 watchdog; no fallback (F2P_WATCHDOG_FALLBACK=$FALLBACK, R50 preflight ${RV:-missing}); nothing else launched"
    exit 2
  fi
  say "EV2S stopped by the F2-T5 fp16 watchdog (= an fp16 FAIL the preflight missed); fallback: R50 passed the same preflight, launching it once (same guards; GPU lock kept)"
  CHOICE=r50
  while true; do
    launch r50
    case "$LSTATE" in
      backoff) sleep "$POLL"; continue;;
      refused) exit 3;;
      deadline) exit 5;;
    esac
    break
  done
  if [ "$RC" -eq 0 ]; then say "done: r50 fallback probe finished OK ($OUT). Next (operator): gate F2-G, JUMP_PLAN_10-05 sec 3. Nothing else launched."; exit 0; fi
fi
say "FAILED: $CHOICE probe exited rc=$RC; nothing else launched. Operator: read $OUT.log"
exit 1
