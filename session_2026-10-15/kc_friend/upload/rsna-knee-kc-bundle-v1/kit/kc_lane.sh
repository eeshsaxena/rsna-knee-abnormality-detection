#!/bin/bash
# kc_lane.sh LANE_ID GPU KEYS STAGE (rental_kc instance, setsid via start_lanes.sh; also the box mock with RSNA_ROOT + MOCK_*).
# Runs the members KEYS (comma list, one after another) on GPU (CUDA_VISIBLE_DEVICES=GPU), each with the exact box queue
# argv (kit/members.tsv template, @R@ -> $R), from $R/fast_build/code2 as `python fleet_train.py <argv>` (= the box queue
# runner: the shim dispatches --kc). No em dashes.
#   STAGE smoke: --out (and --gate_out) redirected to $R/smoke/ (a speed / crash probe the box driver stops after minutes;
#                never a member, never fetched); status/smoke_<KEY>.{running,ended}
#   STAGE real : admission first: the member's estimate (plan.env EST_<kind>_S x 1.05) must end before plan.env
#                DEADLINE_EPOCH (the spend cap / guard line), else status/<KEY>.skipped (the member goes back to the box).
#                Out $R/fleet_runs/<out>, log <out>.log. Early CUDA OOM (< 45 min) with --grad_ckpt 0 -> moved aside
#                (<out>_fastfail_<ts>) and re-run ONCE with --grad_ckpt 1 (= the box runner's fallback). On rc 0 + a closing
#                line: <out>.receipt.json + <out>.sha256 (every file of the member dir + log + receipt [+ the KC2 gate verdict
#                copies]) and status/<KEY>.done; else status/<KEY>.failed. status/<KEY>.running = "<epoch> <gpu>".
set -u
R=${RSNA_ROOT:-/workspace/rsna}; K=$R/kit; S=$R/status; FR=$R/fleet_runs
. $K/kit.env; . $K/plan.env      # plan: PYI NG MEMBERS DEADLINE_EPOCH EST_cnxs_S EST_ev2s_S
LID=$1; G=$2; KEYS=$(echo "$3" | tr ',' ' '); STAGE=${4:-real}
mkdir -p $S $R/logs $FR $R/smoke $R/kc/gate
echo $$ > $S/lane_$LID.pid; echo "$G:$3:$STAGE" > $S/lane_$LID.spec
say(){ echo "[lane$LID $(date '+%m-%d %H:%M:%S')] $*" >> $R/logs/lane_$LID.log; }
closing(){ tail -n 1 "$1" 2>/dev/null | tr -d '\r' | grep -Eq '^\{"raw": [^,]+, "ema": [^}]+\}$'; }
gpu_name(){ [ "${MOCK_CPU:-0}" = 1 ] && { echo "mock-cpu"; return; }; nvidia-smi --query-gpu=name --format=csv,noheader -i "$1" 2>/dev/null | head -n 1; }
if [ "${MOCK_CPU:-0}" = 1 ]; then CS=$(echo ${MOCK_CPUSETS:-0-1} | cut -d' ' -f$(( G + 1 ))); PRE="taskset -c $CS nice -n 19 ionice -c3"; CVD=""
else PRE=""; CVD=$G; fi
say "start pid $$ | GPU $G | members [$KEYS] | $STAGE"
run_member(){
  local KEY=$1 REG N T ARGV OUT LOGF PFX FB=0 t0 t1 dt rc tp EXTRA GO TAB A x est kind
  REG=$(awk -F'\t' -v k=$KEY '$1==k' $K/members.tsv)
  [ -n "$REG" ] || { say "no member $KEY in members.tsv"; echo "no member $KEY in members.tsv" > $S/$KEY.failed; return; }
  N=$(printf '%s' "$REG" | cut -f2); T=$(printf '%s' "$REG" | cut -f7)
  ARGV=$(printf '%s' "$T" | sed -e "s#@R@#$R#g" | tr -d '\r\n')
  if [ "$STAGE" = smoke ]; then
    OUT=$R/smoke/$N; LOGF=$OUT.log; PFX=smoke_$KEY
    ARGV=$(printf '%s' "$ARGV" | sed -e "s#--out $FR/$N#--out $OUT#" -e "s#--gate_out [^ ]*#--gate_out $R/smoke/$N.verdict#")
    rm -rf $OUT $LOGF
  else OUT=$FR/$N; LOGF=$OUT.log; PFX=$KEY; fi
  case " $ARGV " in *" --out $OUT "*) ;; *) say "argv --out is not $OUT"; echo "argv --out mismatch" > $S/$PFX.failed; return;; esac
  if [ "$STAGE" = real ]; then
    if closing $LOGF && [ -s $FR/$N.sha256 ]; then say "$KEY already complete"; touch $S/$KEY.done; return; fi
    . $K/plan.env; kind=cnxs; [ "$KEY" = s2011 ] && kind=ev2s; eval est=\${EST_${kind}_S:-0}; est=$(( est * 105 / 100 ))
    if [ $(( $(date +%s) + est )) -gt "${DEADLINE_EPOCH:-0}" ]; then
      say "SKIP $KEY: estimated $(( est / 60 )) min would end after the spend / guard line $(date -d @${DEADLINE_EPOCH:-0} '+%m-%d %H:%M'); it goes back to the box"
      echo "estimated $(( est / 60 )) min > time left $(( (${DEADLINE_EPOCH:-0} - $(date +%s)) / 60 )) min" > $S/$KEY.skipped; return
    fi
  fi
  while :; do
    if [ -e $OUT ] || [ -e $LOGF ]; then
      if [ "$STAGE" = smoke ]; then rm -rf $OUT $LOGF; else A=${OUT}_partial_$(date +%m%d_%H%M%S); mv -n $OUT $A 2>/dev/null; mv -n $LOGF $A.log 2>/dev/null; say "$KEY partial run moved aside to $A"; fi
    fi
    rm -f $S/$PFX.failed $S/$PFX.done $S/$PFX.ended; echo "$(date +%s) $G" > $S/$PFX.running
    say "LAUNCH $KEY ($STAGE) on GPU $G: $PYI fleet_train.py $ARGV"
    t0=$(date +%s)
    ( cd $R/fast_build/code2 && exec env CUDA_VISIBLE_DEVICES=$CVD HF_HOME=$R/hf HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
        OMP_NUM_THREADS=${OMP_T:-4} $PRE $PYI fleet_train.py $ARGV ) > $LOGF 2>&1 < /dev/null &
    tp=$!; echo $tp > $S/$PFX.tpid; wait $tp; rc=$?; rm -f $S/$PFX.tpid; t1=$(date +%s); dt=$(( t1 - t0 ))
    rm -f $S/$PFX.running
    if [ "$STAGE" = smoke ]; then say "smoke $KEY ended rc $rc after $(( dt / 60 )) min"; echo "rc $rc after $dt s" > $S/$PFX.ended; return; fi
    if [ $rc -eq 0 ] && closing $LOGF; then
      EXTRA=""
      if printf '%s' "$ARGV" | grep -q -- '--gate_out '; then      # KC2 f0: keep its G1 verdict next to the member
        GO=$(printf '%s' "$ARGV" | sed -n 's/.*--gate_out \([^ ]*\).*/\1/p')
        for x in "" .json; do [ -f "$GO$x" ] && cp -p "$GO$x" "$FR/$N.gate_verdict$x" && EXTRA="$EXTRA $N.gate_verdict$x"; done
      fi
      TAB=$(printf '%s' "$ARGV" | sed -n 's/.*--labels \([^ ]*\).*/\1/p')
      RCPT=$FR/$N.receipt.json RN=$N RKEY=$KEY RG=$G RGN="$(gpu_name $G)" RLID=$LID RT0=$t0 RT1=$t1 RRC=$rc RARGV="$ARGV" RFB=$FB \
      RCLOSE="$(tail -n 1 $LOGF | tr -d '\r')" RTAB=$TAB RCODE=$R/fast_build/code2 ${PYR:-$PYI} - <<'PYR' || say "receipt write failed for $KEY"
import json, os, hashlib, glob
e = os.environ
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump(dict(run=e['RN'], key=e['RKEY'], gpu_index=e['RG'], gpu_name=e['RGN'], lane=e['RLID'], started=int(e['RT0']),
               ended=int(e['RT1']), wall_s=int(e['RT1']) - int(e['RT0']), rc=int(e['RRC']), argv=e['RARGV'].strip(),
               grad_ckpt_fallback=int(e['RFB']), closing=e['RCLOSE'], labels=os.path.basename(e['RTAB']), table_sha256=sha(e['RTAB']),
               code_sha256={os.path.basename(p): sha(p) for p in sorted(glob.glob(e['RCODE'] + '/*.py'))},
               container_id=e.get('CONTAINER_ID', '')), open(e['RCPT'], 'w'), indent=1)
PYR
      ( cd $FR && { find $N -type f; echo $N.log; echo $N.receipt.json; for x in $EXTRA; do echo $x; done; } | LC_ALL=C sort | xargs -d '\n' sha256sum > $N.sha256.tmp && mv -f $N.sha256.tmp $N.sha256 ) \
        || say "sha256 list failed for $KEY"
      if [ -s $FR/$N.sha256 ] && [ -s $FR/$N.receipt.json ]; then touch $S/$KEY.done
      else echo "trained (rc 0) but the receipt / sha256 list could not be written: outputs kept in $OUT" > $S/$KEY.failed; fi
      say "DONE $KEY rc 0 in $(( dt / 60 )) min: $(tail -n 1 $LOGF | cut -c1-120) | files $(wc -l < $FR/$N.sha256 2>/dev/null)"; return
    fi
    say "FAILED $KEY rc $rc after $(( dt / 60 )) min: $(tail -n 2 $LOGF | tr '\n' ' ' | cut -c1-300)"
    if [ $FB = 0 ] && [ $dt -lt 2700 ] && [ $rc -ne 0 ] && case " $ARGV " in *" --grad_ckpt 0 "*) true;; *) false;; esac \
       && tail -n 300 $LOGF | grep -Eqi 'out of memory|OutOfMemoryError|CUDNN_STATUS_NOT_ENOUGH_MEMORY|CUBLAS_STATUS_ALLOC_FAILED'; then
      A=${OUT}_fastfail_$(date +%m%d_%H%M); mv -n $OUT $A 2>/dev/null; mv -n $LOGF $A.log 2>/dev/null
      ARGV=$(printf '%s' "$ARGV" | sed 's/ --grad_ckpt 0\( \|$\)/ --grad_ckpt 1\1/'); FB=1
      say "$KEY early CUDA OOM: moved aside to $A, re-running ONCE with --grad_ckpt 1 (box runner fallback)"; continue
    fi
    echo "rc $rc after $(( dt / 60 )) min: $(grep -m1 -E 'Error|error' $LOGF | cut -c1-200)" > $S/$KEY.failed; return
  done
}
for KEY in $KEYS; do run_member $KEY; done
say "lane exit"; rm -f $S/lane_$LID.pid
