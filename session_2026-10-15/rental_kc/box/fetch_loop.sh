#!/bin/bash
# fetch_loop.sh (rental_kc, BOX, setsid; started by driver.sh after RUN_STARTED; survives laptop / Claude session loss).
# Reads ~/rkc/run.env (I_HOST I_PORT GUARD_DEADLINE MEMBERS ...). PULL ONLY: its only writes on the instance are status
# markers (box_seen, <key>.fetched_ok, FINAL_FETCH_OK). Every POLL_S (300 s):
#   done member    -> tar pull of <out>/ + <out>.log + <out>.receipt.json + <out>.sha256 (+ <out>.gate_verdict*) into
#                     ~/rkc/fetch/tmp_<out>, verify_member.py (sha256 of every file, no extras, closing line, expected files,
#                     ckpt args vs the box f0 recipe + the queue argv, receipt), then PLACE into the EMPTY placeholder
#                     ~/fleet_runs/<out> (+ <out>.log) and write ~/fleet_runs/rkc_<out>_receipt.json; KC2: its G1 verdict
#                     is copied to ~/kc/gate/kc2_f0.verdict(.json) only if that file does not exist yet;
#                     3 failed verifies -> kept aside as ~/rkc/fetch/bad_<out>_<ts>, placeholder released
#   failed / stopped / skipped member -> placeholder released (the box line can train it if it is still ACTIVE), resolved
#   from GUARD_DEADLINE - 40 min: running members pulled as ~/fleet_runs/<out>_rentpartial_<ts> (never a member),
#                     pending ones resolved; placeholders released
# When every member is resolved: FINAL_FETCH_OK on the instance (its guard self-destroys 15 min later) and ~/rkc/FINAL_FETCH_OK.
# A released placeholder whose queue line the box already REFUSED needs a re-queue by main: kc_fallback.sh writes the exact
# qctl commands to ~/rkc/REQUEUE_CMDS.sh. Tests: RKC_SSH, RKC_FR, RKC_POLL_S, RKC_MAX_POLLS, VERIFY_SKIP. No em dashes.
set -u
D=${RKC_HOME:-$HOME/rkc}; SRC=${RKC_SRC:-$HOME/rkc/src}; . $SRC/remote/kit.env; . ${RKC_ENV:-$D/run.env}
FR=${RKC_FR:-$HOME/fleet_runs}; FD=$D/fetch; mkdir -p $FD; L=$FD/fetch.log; POLL=${RKC_POLL_S:-300}
PY=$HOME/miniconda3/envs/rsna/bin/python; R=${I_ROOT:-/workspace/rsna}; ST=$D/state_${INSTANCE_ID:-none}; KG=${KC_GATE_DIR:-$HOME/kc/gate}
SSH=${RKC_SSH:-ssh -i $HOME/.ssh/vast_rsna_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_rkc -o LogLevel=ERROR -p $I_PORT root@$I_HOST}
OPS="bash $SRC/box/ops.sh"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L; }
if [ -s $FD/fetch.pid ] && kill -0 "$(cat $FD/fetch.pid)" 2>/dev/null && [ "$(cat $FD/fetch.pid)" != $$ ]; then echo "fetch loop already running (pid $(cat $FD/fetch.pid))"; exit 0; fi
echo $$ > $FD/fetch.pid
MEMBERS=${MEMBERS:-$(cat $ST/members 2>/dev/null)}
say "start pid $$ | instance ${INSTANCE_ID:-?} $I_HOST:$I_PORT | members [$MEMBERS] | guard deadline $(date -d @$GUARD_DEADLINE '+%m-%d %H:%M') | FR $FR"
REG=$ST/members.tsv; BJ=$ST/BUNDLE.json; PB=$ST/parity_box.json
REF=$(ls -tr $FR/$F0_OUT/epochs/f0_s2001_raw_*_ep*_best.pt 2>/dev/null | head -n 1)   # the OLDEST epoch ckpt (fully written)
say "verify against the PUSHED bundle $BJ | registry $REG | recipe reference ${REF:-none (box f0 epoch ckpt missing)}"
resolve(){ echo "$2" > $FD/resolved_$1; say "$1 RESOLVED: $2"; }
place(){   # KEY OUT TMP -> 0 placed
  local k=$1 N=$2 T=$3 DD=$FR/$2
  if pgrep -f -- "fleet_train[.]py .*--out [^ ]*/$N( |\$)" > /dev/null; then say "$k: a box trainer runs $N: not placing"; return 1; fi
  if [ -e $DD.log ]; then say "$k: $DD.log already exists: not placing"; return 1; fi
  if [ -d $DD ] && [ -n "$(ls -A $DD)" ]; then say "$k: $DD is not empty: not placing"; return 1; fi
  [ -d $DD ] || { $OPS ph-create $k >> $L 2>&1 || return 1; }
  mv $T/$N/* $DD/ && mv $T/$N.log $DD.log || { say "$k: move into the placeholder failed"; return 1; }
  return 0
}
miss=0; polls=0; declare -A vfail=()
while :; do
  now=$(date +%s); polls=$((polls + 1)); all=1
  st=$($SSH "RSNA_ROOT=$R bash $R/kit/status.sh touch_seen; RSNA_ROOT=$R bash $R/kit/status.sh" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ] || [ -z "$st" ]; then
    miss=$((miss + 1)); say "instance unreachable ($miss)"
    if [ $now -gt $((GUARD_DEADLINE + 3600)) ]; then
      say "past the guard deadline + 1 h: giving up; unresolved members return to the box"
      for k in $MEMBERS; do [ -e $FD/resolved_$k ] || { $OPS ph-release $k >> $L 2>&1; resolve $k unreachable; }; done
      date > $D/FETCH_GAVE_UP; rm -f $FD/fetch.pid; exit 1
    fi
    sleep $POLL; continue
  fi
  miss=0; [ $((polls % 3)) = 1 ] && say "poll: $(printf '%s\n' "$st" | grep -E '^MEMBER' | cut -c1-110 | tr '\n' '|')"
  near=0; [ $now -ge $((GUARD_DEADLINE - 2400)) ] && near=1
  pp=()
  for k in $MEMBERS; do
    [ -e $FD/resolved_$k ] && continue; N=$($OPS out $k); T=$FD/tmp_$N
    s=$(printf '%s\n' "$st" | sed -n "s/^MEMBER $k state=\([a-z]*\).*/\1/p")
    [ "$s" = done ] && [ ! -e $T/.pulled ] || continue
    rm -rf $T; mkdir -p $T
    ( $SSH "cd $R/fleet_runs && tar cf - $N $N.log $N.receipt.json $N.sha256 \$(ls $N.gate_verdict* 2>/dev/null)" | tar xf - -C $T 2>> $L && touch $T/.pulled ) & pp+=($!)
  done
  t0=$(date +%s); for p in "${pp[@]}"; do wait $p; done; [ ${#pp[@]} -gt 0 ] && say "pulled ${#pp[@]} member(s) in parallel in $(( $(date +%s) - t0 ))s"
  for k in $MEMBERS; do
    [ -e $FD/resolved_$k ] && continue
    N=$($OPS out $k); s=$(printf '%s\n' "$st" | sed -n "s/^MEMBER $k state=\([a-z]*\).*/\1/p")
    case $s in
      done)
        T=$FD/tmp_$N
        if [ -e $T/.pulled ] && rm -f $T/.pulled; then
          V=$(nice -n 19 $PY $SRC/box/verify_member.py $T $k --reg $REG --bundle $BJ --parity $PB --ref "${REF:-}" --root $R --skip "${VERIFY_SKIP:-}" 2>> $L); vr=$?
          echo "$V" >> $L
          if [ $vr -eq 0 ] && place $k $N $T; then
            mkdir -p $FD/$N && cp -p $T/$N.receipt.json $T/$N.sha256 $FD/$N/ && echo "$V" > $FD/$N/verify.json
            for x in "" .json; do
              if [ -f $T/$N.gate_verdict$x ]; then cp -p $T/$N.gate_verdict$x $FR/$N.gate_verdict$x
                if [ ! -e $KG/kc2_f0.verdict$x ]; then cp -p $T/$N.gate_verdict$x $KG/kc2_f0.verdict$x && say "$k: G1 verdict copied to $KG/kc2_f0.verdict$x"
                else say "$k: $KG/kc2_f0.verdict$x exists: kept; the rental copy is $FR/$N.gate_verdict$x"; fi
              fi
            done
            VJ="$V" $PY - "$FR/rkc_${N}_receipt.json" "$T/$N.receipt.json" <<PYR
import json, sys, os
r = json.load(open(sys.argv[2])); v = json.loads(os.environ['VJ'])
json.dump(dict(run="$N", member="$k", source="Vast instance ${INSTANCE_ID:-?} (${GPU_NAME:-?}, offer ${OFFER:-?}), RSNA-only bundle (rental_kc)", placed_into="$FR/$N",
               instance_receipt=r, box_verify=v, verified_at="$(date '+%F %T %Z')"), open(sys.argv[1], 'w'), indent=1)
PYR
            rm -rf $T; $SSH "touch $R/status/$k.fetched_ok" 2>/dev/null
            resolve $k ok; say "$k FETCHED + VERIFIED + PLACED -> $FR/$N ($(du -sh $FR/$N | cut -f1); $(printf '%s' "$V" | sed -n 's/.*"wall_h": \([0-9.]*\).*/\1/p') h on the rental)"
          else
            vfail[$k]=$(( ${vfail[$k]:-0} + 1 )); say "$k verify/place failed (attempt ${vfail[$k]})"; all=0
            if [ ${vfail[$k]} -ge 3 ]; then mv $T $FD/bad_${N}_$(date +%m%d_%H%M%S); $OPS ph-release $k >> $L 2>&1; resolve $k verify_failed; else rm -rf $T; fi
          fi
        else say "$k pull failed (retry next poll)"; rm -rf $T; all=0; fi;;
      failed|stopped|skipped)
        $OPS ph-release $k >> $L 2>&1; resolve $k "$s on the instance: $($SSH "cat $R/status/$k.$s" 2>/dev/null | head -c 160)";;
      running)
        if [ $near = 1 ]; then
          P=${N}_rentpartial_$(date +%m%d_%H%M); T=$FD/tmp_$P; rm -rf $T; mkdir -p $T
          if $SSH "cd $R/fleet_runs && tar cf - $N $N.log" | tar xf - -C $T; then
            mkdir -p $FR/$P && mv $T/$N/* $FR/$P/ 2>/dev/null; mv $T/$N.log $FR/$P.log; rm -rf $T
            $OPS ph-release $k >> $L 2>&1; resolve $k "partial (near the guard deadline) -> $FR/$P (not a member)"
          else say "$k partial pull failed"; all=0; fi
        else all=0; fi;;
      pending)
        if [ $near = 1 ]; then $OPS ph-release $k >> $L 2>&1; resolve $k "never started before the deadline"; else all=0; fi;;
      *) all=0;;
    esac
    [ -e $FD/resolved_$k ] || all=0
  done
  if [ $all = 1 ]; then
    $SSH "date > $R/status/FINAL_FETCH_OK" && say "FINAL_FETCH_OK written on the instance (its guard destroys it 15 min later)"
    date > $D/FINAL_FETCH_OK
    say "ALL RESOLVED: $(for k in $MEMBERS; do echo -n "$k=$(head -c 60 $FD/resolved_$k) | "; done)"; rm -f $FD/fetch.pid; exit 0
  fi
  [ -n "${RKC_MAX_POLLS:-}" ] && [ $polls -ge $RKC_MAX_POLLS ] && { say "test: max polls reached"; rm -f $FD/fetch.pid; exit 2; }
  sleep $POLL
done
