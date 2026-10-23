#!/bin/bash
# driver.sh (rental_kc, BOX, setsid; launch.sh starts it once the rental is up and the box can ssh to it). From then on the
# laptop / Claude session is NOT needed. Reads ~/rkc/run.env. Markers in ~/rkc/state_<IID>/ (a re-run resumes). Any fatal
# step -> abort(): logs pulled to ~/rkc/abort_<ts>/, status/ABORT on the instance (its guard wipes + self-destroys within
# ~60 s), rental placeholders released, ~/rkc/ABORTED (the laptop watcher destroys too). No em dashes.
#   D1 box -> instance ssh (dedicated key)
#   D2 push the RSNA-only bundle (small tree in 1 tar stream + the KC store in 8 parallel streams) with a PUSH-RATE GATE:
#      instance-side bytes / elapsed < PUSH_MIN_MBS (6 MB/s) after 4 min -> abort (10-14 lesson: the Taiwan route ran 0.2 MB/s)
#   D3 instance setup.sh (manifest sha256 + no extras, OAI assertion again, pins == box, GPU checks, parity probe)
#   D4 parity: instance parity_inst.json == the bundle's parity_box.json, EVERY field (10-14 D4 lesson: compare the same
#      quantities on both sides; this probe has no per-plan fields)
#   D5 smoke (never a member; outputs under /workspace/rsna/smoke): phase 1 = 4 trainers, 2 per GPU (the paired speed of each
#      candidate), phase 2 = 1 per GPU (ConvNeXt-S and EffNetV2-S alone); decide.py --smoke -> speeds.json; a crash, no
#      progress or an OOM on an unpaired GPU -> abort
#   D6 WAIT for the box f0 early G1 (~/kc/gate/kc_f0_ep3_ablation.txt): 'EARLY_G1 ablation_pass=True' -> GO;
#      'ablation_pass=False', a FAIL line in ~/kc/gate/kc_f0.verdict, ~/rkc/STOP, or GATE_LATEST passed -> abort (destroy)
#   D7 GO: S read -> the s2002 variant; members in queue priority whose box line is still ACTIVE and argv-identical; code2
#      md5s == the pushed bundle; decide.py --go (measured speeds, spend so far, cap, guard deadline) -> the largest plan that
#      fits (4 = 2 per GPU, 3, 2 = 1 per GPU) else abort; EMPTY placeholders; lanes; RUN_STARTED; guard deadline tightened
#   D8 exec fetch_loop.sh (pull + verify + place each member, FINAL_FETCH_OK)
set -u
D=${RKC_HOME:-$HOME/rkc}; SRC=${RKC_SRC:-$HOME/rkc/src}; . $SRC/remote/kit.env; ENVF=${RKC_ENV:-$D/run.env}; . $ENVF
R=${I_ROOT:-/workspace/rsna}; B=${BUNDLE:-$D/bundle}; ST=$D/state_${INSTANCE_ID}; mkdir -p $ST $D/smoke
L=$D/driver.log; PY=$HOME/miniconda3/envs/rsna/bin/python; OPS="bash $SRC/box/ops.sh"; KG=${KC_GATE_DIR:-$HOME/kc/gate}
SSH=${RKC_SSH:-ssh -i $HOME/.ssh/vast_rsna_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_rkc -o LogLevel=ERROR -p $I_PORT root@$I_HOST}
IENV="RSNA_ROOT=$R ${MOCK_ENV:-}"
say(){ echo "[driver $(date '+%m-%d %H:%M:%S')] $*" >> $L; }
done_(){ [ -e $ST/$1.ok ]; }; mark(){ date '+%F %T' > $ST/$1.ok; }
abort(){
  say "ABORT: $*"; local A=$D/abort_$(date +%m%d_%H%M%S); mkdir -p $A
  $SSH "cd $R && tar cf - setup.log guard.log logs status smoke/*.log fleet_runs/*.log 2>/dev/null" 2>/dev/null | tar xf - -C $A 2>/dev/null
  $SSH "echo '$(echo "$*" | tr -d "'" | head -c 300)' > $R/status/ABORT" && say "status/ABORT written on the instance (guard wipes + self-destroys within ~60 s)"
  [ -s $ST/members ] && for k in $(cat $ST/members); do $OPS ph-release $k >> $L 2>&1; done
  echo "$(date '+%F %T') $*" > $D/ABORTED; say "logs kept in $A; placeholders released"; rm -f $D/driver.pid; exit 1
}
if [ -s $D/driver.pid ] && kill -0 "$(cat $D/driver.pid)" 2>/dev/null && [ "$(cat $D/driver.pid)" != $$ ]; then echo "driver already running"; exit 0; fi
echo $$ > $D/driver.pid; rm -f $D/ABORTED
say "== start pid $$ | instance $INSTANCE_ID ($NG x $GPU_NAME, \$$DPH/h) $I_HOST:$I_PORT | guard $(date -d @$GUARD_DEADLINE '+%m-%d %H:%M') | boot fuse $(date -d @$BOOT_DEADLINE '+%m-%d %H:%M') | gate latest $(date -d @$GATE_LATEST '+%m-%d %H:%M') | spent before \$$SPENT_BEFORE of \$$CAP"

# ---- D1 ----
ok=""; for i in $(seq 1 20); do $SSH 'echo box-ok' 2>/dev/null | grep -q box-ok && { ok=1; break; }; sleep 30; done
[ -n "$ok" ] || abort "D1: the box cannot ssh to the instance"
say "D1 box -> instance ssh OK"

# ---- D2 push (+ push-rate gate) ----
if done_ push; then say "D2 push: already done"; else
  [ -e $B/BUNDLE_OK ] || abort "D2: no finished bundle at $B (run pack.sh)"
  cp -p $B/BUNDLE.json $ST/BUNDLE.json && cp -p $B/kit/members.tsv $ST/members.tsv && cp -p $B/kit/parity_box.json $ST/parity_box.json || abort "D2: bundle snapshot"
  t0=$(date +%s); BY=$(sed -n 's/^ *"bytes": \([0-9]*\).*/\1/p' $B/BUNDLE.json)
  $SSH "mkdir -p $R && rm -rf $R/kc_store $R/fast_build $R/r7t_labels $R/kc $R/rsna_train $R/r7port $R/hf $R/kit" || abort "D2: mkdir on the instance"
  ( cd $B && tar cf - fast_build r7t_labels kc rsna_train r7port hf kit MANIFEST.sha256 BUNDLE.json ) | $SSH "cd $R && tar xf -" || abort "D2: small-tree stream"
  ( cd $B && find kc_store -maxdepth 2 -name '*.npz' | LC_ALL=C sort ) > $ST/store.lst; rm -f $ST/store_shard_*; split -n r/8 -d $ST/store.lst $ST/store_shard_
  pids=(); for s in $ST/store_shard_*; do ( cd $B && tar cf - -T $s ) | $SSH "cd $R && tar xf -" & pids+=($!); done
  gate_ok=0; while :; do
    sleep ${PUSH_POLL_S:-30}; alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=1; done; [ $alive = 0 ] && break
    el=$(( $(date +%s) - t0 ))
    if [ $gate_ok = 0 ] && [ $el -ge ${PUSH_GATE_S:-240} ]; then
      b=$($SSH "du -sb $R/kc_store 2>/dev/null | cut -f1" 2>/dev/null); b=${b:-0}; mbs=$(( b / el / 1000000 ))
      say "D2 push-rate gate at t+${el}s: $((b / 1000000)) MB on the instance = ${mbs} MB/s (need >= ${PUSH_MIN_MBS:-6})"
      if [ $mbs -lt ${PUSH_MIN_MBS:-6} ]; then for p in "${pids[@]}"; do kill $p 2>/dev/null; done; abort "D2 PUSH-RATE GATE: ${mbs} MB/s < ${PUSH_MIN_MBS:-6} MB/s"; fi
      gate_ok=1
    fi
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; [ $rc = 0 ] || abort "D2: a store stream failed"
  dt=$(( $(date +%s) - t0 )); [ $dt -lt 1 ] && dt=1
  n=$($SSH "ls $R/kc_store/store | wc -l"); [ "$n" = "$(wc -l < $ST/store.lst)" ] || abort "D2: instance store holds $n files, want $(wc -l < $ST/store.lst)"
  mark push; say "D2 pushed $((BY / 1000000)) MB in ${dt}s ($((BY / dt / 1000000)) MB/s), store $n files"
fi

# ---- D3 setup ----
if done_ setup; then say "D3 setup: already done"; else
  printf 'MEMBERS=""\nPYI=%s\nNG=%s\n' "${PYI:-/opt/conda/bin/python}" "$NG" | $SSH "cat > $R/kit/plan.env" || abort "D3: plan.env"
  $SSH "cd $R && $IENV setsid -f bash $R/kit/setup.sh > /dev/null 2>&1 < /dev/null" || abort "D3: could not start setup.sh"
  t0=$(date +%s)
  while :; do
    s=$($SSH "ls $R/status/" 2>/dev/null)
    case "$s" in *setup.ok*) break;; *setup.failed*) abort "D3 setup FAILED: $($SSH "cat $R/status/setup.failed; tail -n 4 $R/setup.log" | tr '\n' ' ' | cut -c1-400)";; esac
    [ $(( $(date +%s) - t0 )) -gt 2700 ] && abort "D3: setup not done after 45 min"
    sleep 30
  done
  mark setup; say "D3 SETUP OK: $($SSH "grep -E 'MANIFEST OK|OAI_ASSERT|packages|GPU OK' $R/setup.log" | sed 's/^\[setup [^]]*\] //' | tr '\n' ' ' | cut -c1-700)"
fi

# ---- D4 parity (every field) ----
if done_ parity; then say "D4 parity: already done"; else
  $SSH "cat $R/status/parity_inst.json" > $ST/parity_inst.json || abort "D4: no parity_inst.json"
  P=$($PY - $ST/parity_box.json $ST/parity_inst.json <<'EOF'
import json, sys
b, i = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
vb, vi = dict(b.get('versions') or {}), dict(i.get('versions') or {})
py = (vb.pop('python', ''), vi.pop('python', ''))          # the image python patch level may differ (3.11.x); libraries must not
bad = sorted(k for k in (set(b) | set(i)) - {'versions'} if b.get(k) != i.get(k))
bad += ['versions:' + k for k in sorted(set(vb) | set(vi)) if vb.get(k) != vi.get(k)]
if py[0].split('.')[:2] != py[1].split('.')[:2]: bad.append('versions:python(major.minor)')
print(('PARITY_OK ' if not bad else 'PARITY_MISMATCH ' + ','.join(bad) + ' ') + 'contract %s model_sha ttr %s f2 %s store %s cnxs %s ev2s %s | box %s | inst %s' % (
    b.get('contract_hash'), b.get('model_sha_ttr'), b.get('model_sha_f2'), b.get('n_store'), b.get('backbone_sha_cnxs'), b.get('backbone_sha_ev2s'),
    b.get('versions'), i.get('versions') if bad else 'same'))
EOF
)
  say "D4 $P"; case $P in PARITY_OK*) mark parity;; *) abort "D4 $P";; esac
fi

# ---- D5 smoke (2 phases, never a member) ----
SV=$( [ -s $KG/s_read.verdict ] && grep -Em1 '^[[:space:]]*(PASS|FAIL)' $KG/s_read.verdict | awk '{print $1}' )
S2=s2002tb; [ "$SV" = FAIL ] && S2=s2002d2
CAND=${CANDIDATES:-s2003 $S2 s2004 s2011}
if done_ smoke; then say "D5 smoke: already done"; else
  set -- $CAND; P1="0:${1} 0:${3:-} 1:${2:-} 1:${4:-}"; P1=$(echo $P1 | tr ' ' '\n' | grep -v ':$' | tr '\n' ' ')
  P2="0:$1"; for k in $CAND; do [ "$k" = s2011 ] && P2="$P2 1:s2011"; done; [ "$P2" = "0:$1" ] && [ -n "${2:-}" ] && P2="$P2 1:$2"
  for ph in 1 2; do
    eval LAY=\$P$ph; MIN=$( [ $ph = 1 ] && echo ${SMOKE1_MIN:-8} || echo ${SMOKE2_MIN:-6} )
    r=$($SSH "$IENV bash $R/kit/start_lanes.sh smoke '$LAY'") || abort "D5: start_lanes smoke phase $ph failed: $r"
    say "D5 smoke phase $ph started [$LAY] for $MIN min: $(echo $r | tr '\n' ' ')"
    sleep $(( MIN * 60 ))
    rm -rf $D/smoke/p$ph; mkdir -p $D/smoke/p$ph
    $SSH "cd $R && tar cf - smoke/*.log logs/gpu_mon.csv 2>/dev/null" 2>/dev/null | tar xf - -C $D/smoke/p$ph 2>/dev/null
    for it in $LAY; do k=${it#*:}; r=$($SSH "$IENV bash $R/kit/stop_member.sh $k smoke" 2>&1); say "D5 phase $ph stop $k: $(echo $r | tr '\n' ' ' | cut -c1-200)"; done
    sleep 20
  done
  J=$($PY $SRC/box/decide.py --smoke --reg $ST/members.tsv --p1 "$P1" --p2 "$P2" --dir $D/smoke --epochs $EPOCHS) || abort "D5: decide.py --smoke crashed"
  echo "$J" > $ST/speeds.json; V=$(printf '%s' "$J" | $PY -c 'import json,sys; print(json.load(sys.stdin)["verdict"])')
  say "D5 smoke: $(printf '%s' "$J" | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d["verdict"], "| kinds", json.dumps(d.get("kinds")), "| pairing_disabled", d.get("pairing_disabled"), "| gpu mem peak", d.get("gpu_mem_peak_frac"), "| reasons", "; ".join(d["reasons"])[:300])')"
  [ "$V" = PASS ] || abort "D5 SMOKE FAIL: $(printf '%s' "$J" | $PY -c 'import json,sys; print("; ".join(json.load(sys.stdin)["reasons"]))')"
  mark smoke
fi

# ---- D6 wait for the box f0 early G1 ----
if done_ gate; then say "D6 gate: already PASS"; else
  say "D6 waiting for the early G1 on the box f0 ($KG/kc_f0_ep3_ablation.txt) until $(date -d @$GATE_LATEST '+%m-%d %H:%M')"
  while :; do
    [ -e $D/STOP ] && abort "D6: ~/rkc/STOP present (main / Claude stop)"
    [ -f $KG/kc_f0.verdict ] && grep -Eq '^[[:space:]]*FAIL' $KG/kc_f0.verdict && abort "D6: kc_f0.verdict FAIL: $(grep -Em1 '^[[:space:]]*FAIL' $KG/kc_f0.verdict | cut -c1-160)"
    if [ -f $KG/kc_f0_ep3_ablation.txt ]; then
      g=$(grep -m1 '^EARLY_G1' $KG/kc_f0_ep3_ablation.txt)
      case "$g" in *"ablation_pass=True"*) say "D6 EARLY G1 PASS: $(echo $g | cut -c1-300)"; echo "$g" > $ST/early_g1.txt; mark gate; break;;
                   *"ablation_pass=False"*) abort "D6 EARLY G1 FAIL (destroy): $(echo $g | cut -c1-300)";; esac
    fi
    [ $(date +%s) -ge $GATE_LATEST ] && abort "D6: no early G1 result by $(date -d @$GATE_LATEST '+%H:%M') (waiting costs \$$DPH/h)"
    $SSH "RSNA_ROOT=$R bash $R/kit/status.sh touch_seen" 2>/dev/null
    sleep ${GATE_POLL_S:-60}
  done
fi

# ---- D7 GO: members, plan, placeholders, lanes ----
if done_ run; then say "D7 run: already started"; else
  SV=$( [ -s $KG/s_read.verdict ] && grep -Em1 '^[[:space:]]*(PASS|FAIL)' $KG/s_read.verdict | awk '{print $1}' )
  case $SV in PASS) S2=s2002tb;; FAIL) S2=s2002d2;; *) S2="";; esac
  say "D7 S read verdict: ${SV:-none yet} -> s2002 variant ${S2:-none (s2002 stays with the box)}"
  CM=$($OPS code-check $ST/BUNDLE.json)
  say "D7 box code2 vs the pushed bundle: $CM"; case $CM in CODE_SAME) ;; *) abort "D7 $CM (the box trains with other code than the rental would)";; esac
  avail=""
  for k in s2003 $S2 s2004 s2011; do
    case " $CAND " in *" $k "*) ;; *) [ "$k" = "$S2" ] || continue;; esac
    ls=$($OPS line-state $k); ac=$($OPS argv-check $ST/members.tsv $k); say "D7 $k: queue line $ls | $(echo $ac | cut -c1-160)"
    [ "$ls" = ACTIVE ] && case $ac in *IDENTICAL*) true;; *) false;; esac && avail="$avail $k"
  done
  avail=$(echo $avail); [ -n "$avail" ] || abort "D7: no member available (box lines not ACTIVE / argv differs)"
  J=$($PY $SRC/box/decide.py --go --speeds $ST/speeds.json --avail "$avail" --ng $NG --dph $DPH --spent_before $SPENT_BEFORE --cap $CAP \
       --din ${DIN:-0} --launch_epoch $LAUNCH_EPOCH --guard_deadline $GUARD_DEADLINE --fetch_h ${FETCH_H:-0.3} --margin ${PLAN_MARGIN:-1.10}) || abort "D7: decide.py --go crashed"
  echo "$J" > $ST/decision.json; say "D7 decision: $(printf '%s' "$J" | cut -c1-900)"
  V=$(printf '%s' "$J" | $PY -c 'import json,sys; print(json.load(sys.stdin)["verdict"])')
  [ "$V" = PASS ] || abort "D7 PLAN FAIL: $(printf '%s' "$J" | $PY -c 'import json,sys; print("; ".join(json.load(sys.stdin)["reasons"]))')"
  LAYOUT=$(printf '%s' "$J" | $PY -c 'import json,sys; print(json.load(sys.stdin)["layout"])')
  keep=""; lay=""
  for it in $LAYOUT; do g=${it%%:*}; ks=""
    for k in $(echo ${it#*:} | tr ',' ' '); do r=$($OPS ph-create $k); say "D7 $r"; case $r in OK*) keep="$keep $k"; ks="$ks,$k"; echo "$keep" > $ST/members;; esac; done
    [ -n "$ks" ] && lay="$lay $g:${ks#,}"
  done
  keep=$(echo $keep); lay=$(echo $lay); echo "$keep" > $ST/members
  [ -n "$keep" ] || abort "D7: no placeholder could be created (the box owns every member)"
  PE=$(printf '%s' "$J" | $PY -c 'import json,sys; d=json.load(sys.stdin); print("DEADLINE_EPOCH=%d EST_cnxs_S=%d EST_ev2s_S=%d" % (d["deadline_epoch"], d["est_s"].get("cnxs", 0), d["est_s"].get("ev2s", 0)))')
  $SSH "sed -i -e '/^MEMBERS=/d' -e '/^DEADLINE_EPOCH=/d' -e '/^EST_/d' $R/kit/plan.env && printf 'MEMBERS=\"%s\"\n' '$keep' >> $R/kit/plan.env && printf '%s\n' $PE >> $R/kit/plan.env" \
    || abort "D7: plan.env MEMBERS / DEADLINE_EPOCH"
  r=$($SSH "$IENV bash $R/kit/start_lanes.sh real '$lay'") || abort "D7: start_lanes real failed: $r"
  say "D7 lanes started [$lay]: $(echo $r | tr '\n' ' ')"
  GT=$(printf '%s' "$J" | $PY -c 'import json,sys; print(json.load(sys.stdin)["guard_tighten"])')
  $SSH "cur=\$(cat $R/status/guard_deadline 2>/dev/null || cat $R/status/guard_deadline0 2>/dev/null); [ $GT -lt \${cur:-9999999999} ] && echo $GT > $R/status/guard_deadline; date +%s > $R/status/RUN_STARTED" \
    || abort "D7: could not write RUN_STARTED"
  if [ "$GT" -lt "$GUARD_DEADLINE" ]; then sed -i "s/^GUARD_DEADLINE=.*/GUARD_DEADLINE=$GT/" $ENVF; GUARD_DEADLINE=$GT; fi
  sed -i "s/^MEMBERS=.*//" $ENVF; echo "MEMBERS=\"$keep\"" >> $ENVF
  mark run; say "D7 RUN_STARTED: members [$keep] layout [$lay] | wall $(printf '%s' "$J" | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d["wall_h"], "h | projected total $%s of $%s" % (d["proj_total"], d["cap"]))') | guard deadline $(date -d @$GUARD_DEADLINE '+%m-%d %H:%M')"
fi

# ---- D8 fetch loop ----
say "D8 handing over to fetch_loop.sh"; rm -f $D/driver.pid
exec bash $SRC/box/fetch_loop.sh
