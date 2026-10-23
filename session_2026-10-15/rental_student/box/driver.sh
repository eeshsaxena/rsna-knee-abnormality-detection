#!/bin/bash
# driver.sh (BOX, setsid nohup; launch.sh starts it right after the rental is up and the box can ssh to it). From that
# moment the laptop / Claude session is NOT needed. Reads ~/rstud/run.env. Steps (markers in ~/rstud/state_<IID>/, re-run
# resumes; any fatal step -> abort(): logs pulled to ~/rstud/abort_<ts>/, status/ABORT on the instance (its guard wipes +
# self-destroys within 60 s), placeholders released, ~/rstud/ABORTED (the laptop watcher destroys too)):
#   D1 box -> instance ssh (dedicated key)
#   D2 push the bundle: small tree in 1 tar stream + the store in 8 parallel tar streams over ssh (RSNA-only bundle built
#      and OAI-asserted by pack.sh; BUNDLE.json mode must be REAL unless RSTUD_ALLOW_DRY=1 (tests))
#   D3 instance setup.sh (manifest sha256 of every file + no extras, OAI assertion again, pins == box, GPU checks, parity probe)
#   D4 parity: instance parity_inst.json == the bundle's parity_box.json (contract, decode probe, model / backbone sha, counts)
#   D5 EMPTY placeholders ~/fleet_runs/f<F>_s1047_ttr_r7tb (a fold the box already owns leaves the rental plan)
#   D6 lanes in the smoke layout (4 GPUs: f0-f3 one per GPU + f4 paired on GPU 0) = the first minutes of the REAL run
#   D7 smoke window SMOKE_MIN (10) .. SMOKE_MAX (20) min: decide.py on the pulled logs: FAIL -> abort; projected total > cap
#      -> abort; else keep the pair or move f4 to the first free GPU; refresh plan.env (measured fold time), tighten the
#      guard deadline, status/RUN_STARTED
#   D8 exec fetch_loop.sh (pull + verify + place each fold, FINAL_FETCH_OK)
set -u
D=${RSTUD_HOME:-$HOME/rstud}; SRC=${RSTUD_SRC:-$HOME/rstud/src}; . $SRC/remote/kit.env; ENVF=${RSTUD_ENV:-$D/run.env}; . $ENVF
R=${I_ROOT:-/workspace/rsna}; B=${BUNDLE:-$D/bundle}; ST=$D/state_${INSTANCE_ID}; mkdir -p $ST $D/smoke
L=$D/driver.log; PY=$HOME/miniconda3/envs/rsna/bin/python; OPS="bash $SRC/box/ops.sh"
SSH=${RSTUD_SSH:-ssh -i $HOME/.ssh/vast_rsna_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_rstud -o LogLevel=ERROR -p $I_PORT root@$I_HOST}
IENV="RSNA_ROOT=$R ${MOCK_ENV:-}"        # MOCK_ENV (tests only): MOCK_CPU=1 MOCK_CPUSETS=.. EXTRA_ARGS=.. SETUP_MOCK=1 ...
say(){ echo "[driver $(date '+%m-%d %H:%M:%S')] $*" >> $L; }
done_(){ [ -e $ST/$1.ok ]; }; mark(){ date '+%F %T' > $ST/$1.ok; }
abort(){
  say "ABORT: $*"; local A=$D/abort_$(date +%m%d_%H%M%S); mkdir -p $A
  $SSH "cd $R && tar cf - setup.log guard.log logs status fleet_runs/*.log 2>/dev/null" 2>/dev/null | tar xf - -C $A 2>/dev/null
  $SSH "echo '$(echo "$*" | tr -d "'" | head -c 300)' > $R/status/ABORT" && say "status/ABORT written on the instance (guard wipes + self-destroys within ~60 s)"
  for F in $FOLDS; do $OPS ph-release $F >> $L 2>&1; done
  echo "$(date '+%F %T') $*" > $D/ABORTED; say "logs kept in $A; placeholders released"; exit 1
}
if [ -s $D/driver.pid ] && kill -0 "$(cat $D/driver.pid)" 2>/dev/null && [ "$(cat $D/driver.pid)" != $$ ]; then echo "driver already running"; exit 0; fi
echo $$ > $D/driver.pid; rm -f $D/ABORTED
say "== start pid $$ | instance $INSTANCE_ID ($NG x $GPU_NAME, \$$DPH/h) $I_HOST:$I_PORT | folds [$FOLDS] | guard $(date -d @$GUARD_DEADLINE '+%m-%d %H:%M') | boot fuse $(date -d @$BOOT_DEADLINE '+%m-%d %H:%M') | spent before \$$SPENT_BEFORE of \$$CAP"

# ---- D1 ----
ok=""; for i in $(seq 1 20); do $SSH 'echo box-ok' 2>/dev/null | grep -q box-ok && { ok=1; break; }; sleep 30; done
[ -n "$ok" ] || abort "D1: the box cannot ssh to the instance"
say "D1 box -> instance ssh OK"

# ---- D2 push ----
if done_ push; then say "D2 push: already done"; else
  [ -e $B/BUNDLE_OK ] || abort "D2: no finished bundle at $B (run pack.sh)"
  MODE=$(sed -n 's/.*"mode": "\([A-Z]*\)".*/\1/p' $B/BUNDLE.json)
  [ "$MODE" = REAL ] || [ "${RSTUD_ALLOW_DRY:-0}" = 1 ] || abort "D2: bundle mode $MODE (REAL needed)"
  cp -p $B/BUNDLE.json $ST/BUNDLE.json || abort "D2: BUNDLE.json snapshot"   # review 10-14: fetch_loop verifies against the PUSHED bundle
  t0=$(date +%s); BY=$(sed -n 's/.*"bytes": \([0-9]*\).*/\1/p' $B/BUNDLE.json)
  $SSH "mkdir -p $R && rm -rf $R/fleet_store $R/fast_build $R/r7t_labels $R/rsna_train $R/hf $R/kit" || abort "D2: mkdir on the instance"
  ( cd $B && tar cf - fast_build r7t_labels rsna_train hf kit MANIFEST.sha256 BUNDLE.json ) | $SSH "cd $R && tar xf -" || abort "D2: small-tree stream"
  ls $B/fleet_store | sed 's#^#fleet_store/#' > $ST/store.lst; rm -f $ST/store_shard_*; split -n r/8 -d $ST/store.lst $ST/store_shard_
  pids=(); for s in $ST/store_shard_*; do ( cd $B && tar cf - -T $s ) | $SSH "cd $R && tar xf -" & pids+=($!); done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; [ $rc = 0 ] || abort "D2: a store stream failed"
  dt=$(( $(date +%s) - t0 )); [ $dt -lt 1 ] && dt=1
  n=$($SSH "ls $R/fleet_store | wc -l"); [ "$n" = "$(wc -l < $ST/store.lst)" ] || abort "D2: instance store holds $n files, want $(wc -l < $ST/store.lst)"
  mark push; say "D2 pushed $((BY / 1000000)) MB in ${dt}s ($((BY / dt / 1000000)) MB/s), store $n files"
fi

# ---- D3 setup ----
if done_ setup; then say "D3 setup: already done"; else
  printf 'FOLDS="%s"\nDEADLINE_EPOCH=%s\nFOLD_EST_S=%s\nPYI=%s\nNG=%s\n' "$FOLDS" "$LANE_DEADLINE" "$FOLD_EST_S" "${PYI:-/opt/conda/bin/python}" "$NG" \
    | $SSH "cat > $R/kit/plan.env" || abort "D3: plan.env"
  $SSH "cd $R && $IENV setsid -f bash $R/kit/setup.sh > /dev/null 2>&1 < /dev/null" || abort "D3: could not start setup.sh"
  t0=$(date +%s)
  while :; do
    s=$($SSH "ls $R/status/" 2>/dev/null)
    case "$s" in *setup.ok*) break;; *setup.failed*) abort "D3 setup FAILED: $($SSH "cat $R/status/setup.failed; tail -n 4 $R/setup.log" | tr '\n' ' ' | cut -c1-400)";; esac
    [ $(( $(date +%s) - t0 )) -gt 2700 ] && abort "D3: setup not done after 45 min"
    sleep 30
  done
  mark setup; say "D3 SETUP OK: $($SSH "grep -E 'MANIFEST OK|OAI_ASSERT|packages|GPU OK' $R/setup.log" | sed 's/^\[setup [^]]*\] //' | tr '\n' ' ' | cut -c1-600)"
fi

# ---- D4 parity ----
if done_ parity; then say "D4 parity: already done"; else
  $SSH "cat $R/status/parity_inst.json" > $ST/parity_inst.json || abort "D4: no parity_inst.json"
  P=$($PY - $B/kit/parity_box.json $ST/parity_inst.json "$FOLDS" <<'EOF'
import json, sys
b, i = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); bad = []
# 10-14 20:07: the instance probes only the rental FOLDS (a 4-fold plan leaves f4 to the box): compare those folds' counts
# with the box's, and require exactly the planned folds (5-fold plan = the old full comparison)
want = sorted(sys.argv[3].split()); fi = i.get('fold_counts') or {}
if sorted(fi) != want or any(b.get('fold_counts', {}).get(F) != fi.get(F) for F in want): bad.append('fold_counts')
for k in ('contract', 'select_sha', 'jpeg_fp', 'model_sha', 'n', 'decode_probe', 'backbone_sha', 'state_keys', 'pretrained_cfg'):
    if b.get(k) != i.get(k): bad.append(k)
print(('PARITY_OK ' if not bad else 'PARITY_MISMATCH ' + ','.join(bad) + ' ') + 'model_sha %s n %s folds %s | box %s | inst %s' % (b.get('model_sha'), b.get('n'), b.get('fold_counts'), b.get('versions'), i.get('versions')))
EOF
)
  say "D4 $P"; case $P in PARITY_OK*) mark parity;; *) abort "D4 $P";; esac
fi

# ---- D5 placeholders ----
if done_ ph; then say "D5 placeholders: already done"; FOLDS=$(cat $ST/folds); else
  keep=""; for F in $FOLDS; do r=$($OPS ph-create $F); say "D5 $r"; case $r in OK*) keep="$keep $F";; esac; done
  keep=$(echo $keep); [ -n "$keep" ] || abort "D5: no fold could be reserved (the box owns them all)"
  [ "$keep" = "$FOLDS" ] || { say "D5 plan reduced to [$keep]"; FOLDS=$keep; }
  echo "$FOLDS" > $ST/folds; $SSH "sed -i 's/^FOLDS=.*/FOLDS=\"$FOLDS\"/' $R/kit/plan.env"; mark ph
fi

# ---- D6 lanes (smoke layout) ----
set -- $FOLDS; NF=$#
layout(){   # NG folds... -> lane spec
  local ng=$1; shift; local f=("$@") n=$# out="" i
  if [ $ng -ge $n ]; then for i in $(seq 0 $((n - 1))); do out="$out $i:${f[$i]}"; done
  elif [ $ng -eq 4 ] && [ $n -eq 5 ]; then out="0:${f[0]} 1:${f[1]} 2:${f[2]} 3:${f[3]} 0:${f[4]}"
  elif [ $ng -ge 2 ]; then for i in $(seq 0 $((n - 1))); do if [ $i -lt $((2 * ng)) ]; then out="$out $((i % ng)):${f[$i]}"; else out="$out any:${f[$i]}"; fi; done
  else out="0:$(echo ${f[*]} | tr ' ' ',')"; fi
  echo $out; }
LAYOUT=${LAYOUT_OVERRIDE:-$(layout $NG $FOLDS)}
if done_ lanes; then say "D6 lanes: already started"; LANES_EPOCH=$(cat $ST/lanes_epoch); else
  r=$($SSH "$IENV bash $R/kit/start_lanes.sh '$LAYOUT'") || abort "D6: start_lanes failed: $r"
  LANES_EPOCH=$(date +%s); echo $LANES_EPOCH > $ST/lanes_epoch; mark lanes
  say "D6 lanes started, layout [$LAYOUT]: $(echo $r | tr '\n' ' ')"
fi

# ---- D7 smoke window + decision ----
if done_ run; then say "D7 run: already started"; else
  SMIN=${SMOKE_MIN:-10}; SMAX=${SMOKE_MAX:-20}; SP=${SMOKE_POLL_S:-60}
  while :; do
    sleep $SP; el=$(( ($(date +%s) - LANES_EPOCH) / 60 ))
    rm -rf $D/smoke/cur; mkdir -p $D/smoke/cur
    $SSH "cd $R && tar cf - fleet_runs/*.log logs/gpu_mon.csv 2>/dev/null" 2>/dev/null | tar xf - -C $D/smoke/cur 2>/dev/null
    mv $D/smoke/cur/logs/gpu_mon.csv $D/smoke/cur/fleet_runs/ 2>/dev/null
    fin=0; [ $el -ge $SMAX ] && fin=1
    J=$($PY $SRC/box/decide.py --logs $D/smoke/cur/fleet_runs --layout "$LAYOUT" --ng $NG --seed $SEED --runtag $RUNTAG --epochs ${RUN_EPOCHS:-$EPOCHS} \
         --dph $DPH --spent_before $SPENT_BEFORE --cap $CAP --din ${DIN:-0} --launch_epoch $LAUNCH_EPOCH --lanes_epoch $LANES_EPOCH \
         --guard_deadline $GUARD_DEADLINE --final $fin --fetch_h ${FETCH_H:-0.3} --min_it ${DECIDE_MIN_IT:-200}) || abort "D7: decide.py crashed"
    V=$(printf '%s' "$J" | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d["verdict"], d.get("action"))')
    say "D7 smoke t+${el} min: $V | $(printf '%s' "$J" | cut -c1-700)"
    case $V in FAIL*) cp -r $D/smoke/cur $D/smoke/fail_$(date +%H%M%S); abort "D7 SMOKE FAIL: $(printf '%s' "$J" | $PY -c 'import json,sys; print("; ".join(json.load(sys.stdin)["reasons"]))')";; esac
    if [ $el -ge $SMIN ] && [ "${V%% *}" = PASS ]; then break; fi
  done
  cp -r $D/smoke/cur $D/smoke/pass_$(date +%H%M%S); echo "$J" > $ST/decision.json
  ACT=${V#* }; EST=$(printf '%s' "$J" | $PY -c 'import json,sys; print(json.load(sys.stdin)["fold_est_s"])')
  GT=$(printf '%s' "$J" | $PY -c 'import json,sys; print(json.load(sys.stdin)["guard_tighten"])')
  if [ "$ACT" = seq ] && [ $NF -le $NG ]; then say "D7 seq ignored: [$FOLDS] on $NG GPUs has no paired fold"   # review 10-14
  elif [ "$ACT" = seq ]; then
    if [ $NG -ge 4 ]; then sf=${FOLDS##* }; pf=${FOLDS%% *}
      r=$($SSH "$IENV bash $R/kit/stop_fold.sh $sf"); say "D7 seq: f$sf leaves the paired GPU: $(echo $r | tr '\n' ' ')"
      # review 10-14: an OOM on the paired GPU 0 can kill f$pf (its partner) instead of f$sf: restart it alone on GPU 0
      if [ "$pf" != "$sf" ] && $SSH "test -e $R/status/f$pf.failed && grep -qE 'out of memory|OutOfMemoryError' $R/fleet_runs/f${pf}_s${SEED}_${RUNTAG}.log"; then
        r=$($SSH "$IENV bash $R/kit/stop_fold.sh $pf; $IENV bash $R/kit/start_lanes.sh '0:$pf'"); say "D7 seq: f$pf ran out of memory on the paired GPU 0, restarted alone there: $(echo $r | tr '\n' ' ')"; fi
      r=$($SSH "$IENV bash $R/kit/start_lanes.sh 'any:$sf'"); say "D7 seq: f$sf re-runs on the first free GPU: $(echo $r | tr '\n' ' ')"
    else paired=$(echo $FOLDS | cut -d' ' -f$((NG + 1))-); for F in $paired; do $SSH "$IENV bash $R/kit/stop_fold.sh $F" >> $L; done
      set -- $paired; r=$($SSH "$IENV bash $R/kit/start_lanes.sh 'any:$(echo $paired | tr ' ' ',')'"); say "D7 seq (OOM): [$paired] re-run one at a time on free GPUs: $(echo $r | tr '\n' ' ')"; fi
  else say "D7 keep: layout [$LAYOUT] continues"; fi
  $SSH "sed -i 's/^FOLD_EST_S=.*/FOLD_EST_S=$EST/' $R/kit/plan.env; cur=\$(cat $R/status/guard_deadline 2>/dev/null || cat $R/status/guard_deadline0 2>/dev/null);[ $GT -lt \${cur:-9999999999} ] && echo $GT > $R/status/guard_deadline; date +%s > $R/status/RUN_STARTED" \
    || abort "D7: could not write RUN_STARTED"
  if [ "$GT" -lt "$GUARD_DEADLINE" ]; then   # review 10-14: the fetch loop and the lanes' admission follow the TIGHTENED guard
    sed -i "s/^GUARD_DEADLINE=.*/GUARD_DEADLINE=$GT/" $ENVF; GUARD_DEADLINE=$GT
    $SSH "sed -i 's/^DEADLINE_EPOCH=.*/DEADLINE_EPOCH=$((GT - 2700))/' $R/kit/plan.env" && say "D7 run.env GUARD_DEADLINE and the lane deadline (guard - 45 min) follow the tightened guard"
  fi
  mark run; say "D7 RUN_STARTED: measured fold $((EST / 60)) min, guard deadline tightened to $(date -d @$GT '+%m-%d %H:%M') (if earlier), projected total \$$(printf '%s' "$J" | $PY -c 'import json,sys; print(json.load(sys.stdin)["proj_total"])') of \$$CAP"
fi

# ---- D8 fetch loop ----
say "D8 handing over to fetch_loop.sh"; rm -f $D/driver.pid
sed -i "s/^FOLDS=.*/FOLDS=\"$FOLDS\"/" $ENVF
exec bash $SRC/box/fetch_loop.sh
