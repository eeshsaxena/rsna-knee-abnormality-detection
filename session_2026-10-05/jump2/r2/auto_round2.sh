#!/bin/bash
# auto_round2.sh (jump2 r2, written 2026-10-05; NOT started by its author). Own-OOF pseudo-label round 2: gate run only.
# Start (operator):  setsid nohup bash ~/auto_round2.sh > /dev/null 2>&1 < /dev/null &
# Status only (no launch, no writes):  bash ~/auto_round2.sh --check
# 1) waits until all 5 fold runs of the OOF set (default ~/fleet_runs/f{0..4}_s1043_ttr_mix50) have a closing JSON line,
#    the fold chain (auto_folds.sh) has exited, no PAUSE file exists (~/fleet_runs/PAUSE_CHAIN, ~/jump2/r2/PAUSE_R2) and
#    no python process holds the GPU on two checks 2 min apart;
# 2) builds ~/jump2/r2/labels_r2_<VARIANT>.csv with build_r2_targets.py in FULL mode (aborts if any training row lacks OOF);
# 3) under 'flock ~/jump2/gpu.lock' (GPU idle re-checked inside the lock) launches round-2 TTR fold 0 ALONE: seed 1543,
#    J1 recipe, --labels the new csv, --eval_labels steven_v4, out ~/fleet_runs/f0_s1543_ttr_r2<VARIANT>, through
#    r2_capped.py (CUDA memory cap 0.75); fast path (code2 + --grad_ckpt 0) only if ~/fast_build/SWITCH.md has a line
#    'STATUS: VERIFIED' and code2's model sources are byte-identical to bigjump's;
# 4) after the run exits, evaluates the pre-registered gate (r2_gate.py, CPU) and logs the verdict. Launches nothing else.
# Never kills or signals any process. Log ~/fleet_runs/auto_round2.log. Gives up (no launch) after R2_DEADLINE.
# Env overrides: R2_VARIANT (a50) R2_SEED (1543) R2_SETS (space-separated globs) R2_FAST (auto|0) R2_HOLD_LOCK (1)
#                R2_MEMFRAC (0.75) R2_DEADLINE ('2026-10-07 12:00'); test only: R2_RUNS R2_DIR R2_CODE_DIR R2_LOCK R2_TEST.
# verify fix 10-05: seed default was 2043, which fleet_train.py BJ-5 refuses for --model ttr (seeds 1000-1999 only), so
#   the gate run would have died at launch. R2_SEED must be 1000-1999 and must not reuse an existing f0_s<seed>_* run.
#   R2_SETS: each pattern needs exactly one 'f?_' and no '*' or '[' (the bash closed-check cannot expand them); a
#   leading '~' is expanded. The chain check also covers ~/fast_build/auto_folds_fast.sh.
set -u
SELF=$(readlink -f "$0")
PY=$HOME/miniconda3/envs/rsna/bin/python
RUNS=${R2_RUNS:-$HOME/fleet_runs}
R2=${R2_DIR:-$HOME/jump2/r2}
LOCK=${R2_LOCK:-$HOME/jump2/gpu.lock}
LOG=$RUNS/auto_round2.log
VARIANT=${R2_VARIANT:-a50}
SEED=${R2_SEED:-1543}
SETS=${R2_SETS:-$RUNS/f?_s1043_ttr_mix50}
HOLD_LOCK=${R2_HOLD_LOCK:-1}
MEMFRAC=${R2_MEMFRAC:-0.75}
DEADLINE=${R2_DEADLINE:-2026-10-07 12:00}
TEST=${R2_TEST:-0}
LABELS=$R2/labels_r2_${VARIANT}.csv
OUT=$RUNS/f0_s${SEED}_ttr_r2${VARIANT}
SLOW=120; [ "$TEST" = 1 ] && SLOW=1
read -r -a SETARR <<< "$SETS"          # read does not glob: patterns reach python untouched
COMMON="--store $HOME/fleet_store --folds $HOME/rsna_train/folds5.csv --exclude $HOME/rsna_train/gold74_eval_set.csv --eval_labels $HOME/rsna_train/llm_labels_v4_blend.csv --gold $HOME/rsna_train/gold74_eval_set.csv"
RECIPE="--model ttr --aug heavy --train_slices 16 --drop_path 0.2 --dropout 0.3 --ema 0.9995 --epochs 24 --seed $SEED --save_epochs 6-24 --workers 10"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
say(){ if [ "$CHECK" = 1 ]; then echo "[auto-r2 check] $*"; else echo "[auto-r2 $(date '+%m-%d %H:%M')] $*" >> "$LOG"; fi; }

[[ "$VARIANT" =~ ^a[0-9]+(s[0-9]+)?$ ]] || { say "bad R2_VARIANT '$VARIANT'"; exit 2; }
# fleet_train.py BJ-5: --model ttr refuses seeds outside 1000-1999 (AssertionError at launch)
[[ "$SEED" =~ ^1[0-9]{3}$ ]] || { say "bad R2_SEED '$SEED': fleet_train.py --model ttr accepts only seeds 1000-1999 (BJ-5)"; exit 2; }
for d in "$RUNS"/f0_s${SEED}_*; do [ -e "$d" ] && [ "$d" != "$OUT" ] && [ "$d" != "$OUT.log" ] && { say "bad R2_SEED '$SEED': $d already uses member key f0_s$SEED"; exit 2; }; done
SETARR=("${SETARR[@]/#\~/$HOME}")
for pat in "${SETARR[@]}"; do
  case "$pat" in *'*'*|*'['*) say "bad R2_SETS pattern '$pat': no '*' or '[' (use one 'f?_' per pattern)"; exit 2;; esac
  [[ "$pat" == *'f?_'* ]] || { say "bad R2_SETS pattern '$pat': must contain 'f?_'"; exit 2; }
done

closed(){ tail -n 1 "$1.log" 2>/dev/null | grep -qE '^\{"raw": [-0-9.eE+]+, "ema": [-0-9.eE+]+\}$'; }
n_closed(){ local n=0 pat F; for pat in "${SETARR[@]}"; do for F in 0 1 2 3 4; do closed "${pat/f\?_/f${F}_}" && n=$((n+1)); done; done; echo $n; }
NEED=$(( 5 * ${#SETARR[@]} ))
gpu_apps(){ [ "$TEST" = 1 ] && { echo 0; return; }; local o; o=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null) || { echo 99; return; }; echo "$o" | grep -c python; }
chain_running(){ [ "$TEST" = 1 ] && return 1; pgrep -f 'auto_folds(_fast)?\.sh' > /dev/null; }
paused(){ [ -e "$RUNS/PAUSE_CHAIN" ] || [ -e "$R2/PAUSE_R2" ]; }
past_deadline(){ [ "$(date +%s)" -ge "$(date -d "$DEADLINE" +%s)" ]; }

CODE=$HOME/bigjump_build/code; FASTFLAG=""; FASTWHY=""
fast_ok(){
  local S=$HOME/fast_build/SWITCH.md C2=$HOME/fast_build/code2 B=$HOME/bigjump_build/code
  [ "${R2_FAST:-auto}" = 0 ] && { FASTWHY="R2_FAST=0"; return 1; }
  [ -f "$S" ] || { FASTWHY="no $S"; return 1; }
  grep -qiE '^[[:space:]#*-]*(STATUS|VERDICT)[*[:space:]]*[:=][*[:space:]]*VERIFIED([^A-Za-z]|$)' "$S" || { FASTWHY="SWITCH.md has no 'STATUS: VERIFIED' line"; return 1; }
  grep -qiE 'NOT[[:space:]]+VERIFIED|UNVERIFIED' "$S" && { FASTWHY="SWITCH.md mentions NOT VERIFIED/UNVERIFIED"; return 1; }
  grep -q -- "--grad_ckpt" "$C2/fleet_train.py" 2>/dev/null || { FASTWHY="code2/fleet_train.py has no --grad_ckpt"; return 1; }
  cmp -s "$C2/fleet_model_ttr.py" "$B/fleet_model_ttr.py" && cmp -s "$C2/fleet_model.py" "$B/fleet_model.py" || { FASTWHY="code2 model sources differ from bigjump (model_sha would change)"; return 1; }
  FASTWHY="SWITCH.md VERIFIED + model sources identical"; return 0
}
pick_code(){ if [ -n "${R2_CODE_DIR:-}" ]; then CODE=$R2_CODE_DIR; FASTFLAG=""; FASTWHY="R2_CODE_DIR override";
             elif fast_ok; then CODE=$HOME/fast_build/code2; FASTFLAG="--grad_ckpt 0";
             else CODE=$HOME/bigjump_build/code; FASTFLAG=""; fi; }

# ---------------- inner: runs while holding the GPU lock (or right after a lock barrier) ----------------
if [ "${1:-}" = "--inner" ]; then
  [ "$(gpu_apps)" -ne 0 ] && { say "inner: a python process holds the GPU inside the lock window; backing off"; exit 75; }
  paused && { say "inner: pause file present; backing off"; exit 75; }
  [ -e "$OUT" ] && { say "inner: $OUT exists, not launching"; exit 76; }
  pick_code
  cd "$CODE" || { say "inner: cannot cd $CODE"; exit 77; }
  say "inner: LAUNCH fold 0 seed $SEED labels $LABELS -> $OUT | code $CODE ($FASTWHY) $FASTFLAG | memfrac $MEMFRAC | hold_lock $HOLD_LOCK"
  R2_MEMFRAC=$MEMFRAC $PY "$R2/r2_capped.py" fleet_train.py $RECIPE $FASTFLAG $COMMON --labels "$LABELS" --fold 0 --out "$OUT" > "$OUT.log" 2>&1 < /dev/null
  rc=$?
  say "inner: trainer exited rc=$rc ($(tail -n 1 "$OUT.log" | cut -c1-140))"
  exit $rc
fi

# ---------------- check mode ----------------
if [ "$CHECK" = 1 ]; then
  pick_code
  say "OOF set(s): ${SETARR[*]} | closed fold runs $(n_closed)/$NEED"
  for pat in "${SETARR[@]}"; do for F in 0 1 2 3 4; do d=${pat/f\?_/f${F}_}; if closed "$d"; then say "  $d: CLOSED $(tail -n 1 "$d.log")"; else say "  $d: open"; fi; done; done
  chain_running && say "fold chain auto_folds.sh: RUNNING" || say "fold chain auto_folds.sh: not running"
  say "python apps on GPU: $(gpu_apps)"; paused && say "PAUSE file present" || say "no pause file"
  say "code path at launch time would be: $CODE ($FASTWHY) flag '$FASTFLAG'"
  say "variant $VARIANT -> labels $LABELS | out $OUT (exists: $([ -e "$OUT" ] && echo yes || echo no)) | lock $LOCK hold $HOLD_LOCK | deadline $DEADLINE (past: $(past_deadline && echo yes || echo no))"
  exit 0
fi

# ---------------- main ----------------
mkdir -p "$R2"
exec 8> "$R2/.auto_round2.instance"
flock -n 8 || { echo "auto_round2.sh already running"; exit 1; }
say "start pid $$ | variant $VARIANT seed $SEED | sets ${SETARR[*]} | labels $LABELS | out $OUT | deadline $DEADLINE"
built=0; i=0
while true; do
  past_deadline && { say "deadline $DEADLINE passed without a launch; exiting, nothing launched"; exit 5; }
  nc=$(n_closed); ga=$(gpu_apps)
  if [ "$nc" -lt "$NEED" ] || chain_running || paused || [ "$ga" -ne 0 ]; then
    [ $((i % 15)) -eq 0 ] && say "waiting: closed $nc/$NEED | chain $(chain_running && echo running || echo idle) | python on GPU $ga | $(paused && echo PAUSED || echo no-pause)"
    i=$((i + 1)); sleep $SLOW; continue
  fi
  sleep $SLOW                                   # idle must hold on two checks
  [ "$(gpu_apps)" -ne 0 ] && continue
  chain_running && continue
  if [ "$built" = 0 ]; then
    say "all $NEED fold runs closed, chain idle, GPU idle: building $LABELS"
    nice -n 19 $PY "$R2/build_r2_targets.py" --sets "${SETARR[@]}" --variants "$VARIANT" --out_dir "$R2" > "$R2/build_${VARIANT}.log" 2>&1
    brc=$?
    if [ $brc -ne 0 ] || [ ! -s "$LABELS" ]; then say "ABORT: label build failed rc=$brc (see $R2/build_${VARIANT}.log); nothing launched"; exit 4; fi
    grep -E '^\[r2\] (mode|set[0-9]|MACRO|PARENT)|vs raptor' "$R2/build_${VARIANT}.log" | while read -r l; do say "build: $l"; done
    say "labels ready: $(wc -l < "$LABELS") lines sha256 $(sha256sum "$LABELS" | cut -c1-16)"
    built=1
  fi
  if [ "$HOLD_LOCK" = 1 ]; then flock "$LOCK" bash "$SELF" --inner; rc=$?
  else flock "$LOCK" true; bash "$SELF" --inner; rc=$?; fi
  [ $rc -eq 75 ] && { say "launch backed off (GPU or pause); waiting again"; sleep $SLOW; continue; }
  break
done
if closed "$OUT"; then
  say "gate: evaluating (r2_gate.py, CPU)"
  nice -n 19 $PY "$R2/r2_gate.py" --cand "$OUT" --out "$R2/GATE_RESULT_${VARIANT}" > "$R2/gate_${VARIANT}.log" 2>&1
  grc=$?
  while read -r l; do say "gate: $l"; done < "$R2/gate_${VARIANT}.log"
  say "gate rc=$grc (0 = PASS). Nothing else launched; the operator decides all-data seeds from GATE_RESULT_${VARIANT}.txt"
else
  say "run $OUT has no closing line (rc=$rc); gate not evaluated. Nothing else launched."
fi
say "done"
