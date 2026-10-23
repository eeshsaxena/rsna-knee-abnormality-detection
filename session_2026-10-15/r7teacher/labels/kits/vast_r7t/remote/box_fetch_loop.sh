#!/bin/bash
# box_fetch_loop.sh HOST PORT GUARD_DEADLINE "FOLDS" IID GPU OFFER (BOX side, setsid; survives laptop / session loss).
# vastR7T (2026-10-14, pattern = session_2026-10-13/raptor/vast/box_fetch.sh, lesson 10-10: a laptop-side watcher dying
# lost a whole rental). PULL ONLY: the box writes nothing to the instance except status markers (f<F>.fetched_ok,
# FINAL_FETCH_OK). Every 10 min: for each planned fold whose trainer log has the closing line on the instance, create the
# box placeholder if needed (box_ops.sh ph-create), pull it (box_ops.sh pull) and verify it (box_ops.sh verify: md5 of every
# file vs the instance manifest, closing line, ckpt args); on VERIFY_OK mark it fetched on both sides. From GUARD_DEADLINE
# - 45 min, folds still running are pulled as *_vastpartial_<time> (never counted as students). When every fold is fetched,
# skipped, failed or partial: touch status/FINAL_FETCH_OK on the instance (its guard self-destroys 15 min later).
# Log ~/vastR7T/fetch_loop.log, pid ~/vastR7T/fetch_loop.pid. Idempotent: re-running resumes (fetched_f<F> markers).
set -u
H=$1; P=$2; DL=$3; FOLDS=$4; IID=${5:-?}; GPU=${6:-?}; OFFER=${7:-?}; R=/workspace/rsna; D=$HOME/vastR7T
. $D/kit.env
L=$D/fetch_loop.log; OPS="bash $D/box_ops.sh"
SSH="ssh -i $HOME/.ssh/vast_rsna_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_vastR7T -o LogLevel=ERROR -p $P root@$H"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L; }
if [ -s $D/fetch_loop.pid ] && kill -0 "$(cat $D/fetch_loop.pid)" 2>/dev/null; then echo "fetch loop already running (pid $(cat $D/fetch_loop.pid))"; exit 0; fi
echo $$ > $D/fetch_loop.pid
say "start pid $$ | instance $H:$P ($IID, $GPU, offer $OFFER) | folds [$FOLDS] | guard deadline $(date -d @$DL '+%m-%d %H:%M')"
miss=0
while :; do
  now=$(date +%s); all=1
  st=$($SSH "bash $R/kit/status.sh" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ] || [ -z "$st" ]; then
    miss=$((miss + 1)); say "instance unreachable ($miss)"
    [ $now -gt $((DL + 3600)) ] && { say "past the guard deadline + 1 h: giving up"; rm -f $D/fetch_loop.pid; exit 1; }
    sleep 600; continue
  fi
  miss=0
  for F in $FOLDS; do
    [ -e $D/fetched_f$F ] && continue
    N=f${F}_s${SEED}_${RUNTAG}
    s=$(printf '%s\n' "$st" | sed -n "s/^FOLD f$F state=\([^ ]*\).*/\1/p")
    case $s in
      done*)
        $OPS ph-create $F >> $L 2>&1
        MAN=$($SSH "EPOCHPT=0 bash $R/kit/status.sh manifest $F") || { say "f$F manifest failed"; all=0; continue; }
        if $OPS pull "$N" "$N" "$H" "$P" "$R" 0 >> $L 2>&1 && printf '%s\n' "$MAN" | $OPS verify "$N" "$N" canonical "$IID" "$GPU" "$OFFER" >> $L 2>&1; then
          $SSH "touch $R/status/f$F.fetched_ok" && touch $D/fetched_f$F && say "f$F FETCHED + VERIFIED -> ~/fleet_runs/$N"
        else say "f$F pull/verify failed (retry next poll; see above)"; all=0; fi;;
      skipped*|failed*)
        echo "$s" > $D/fetched_f$F; say "f$F ended as '$s' on the instance: nothing to fetch (the fold stays untrained)";;
      running*|pending*)
        [ "$s" = "running" ] && $OPS ph-create $F >> $L 2>&1
        if [ $now -ge $((DL - 2700)) ] && $SSH "test -e $R/fleet_runs/$N.log"; then
          T=${N}_vastpartial_$(date +%m%d_%H%M)
          MAN=$($SSH "EPOCHPT=0 bash $R/kit/status.sh manifest $F") && $OPS pull "$N" "$T" "$H" "$P" "$R" 0 >> $L 2>&1 \
            && printf '%s\n' "$MAN" | $OPS verify "$N" "$T" partial "$IID" "$GPU" "$OFFER" >> $L 2>&1 \
            && { echo partial > $D/fetched_f$F; say "f$F near the deadline: partial output kept as ~/fleet_runs/$T (not a student)"; } || all=0
        else all=0; fi;;
      *) all=0;;
    esac
  done
  if [ $all = 1 ]; then
    $SSH "date > $R/status/FINAL_FETCH_OK" && say "FINAL_FETCH_OK written on the instance (its guard destroys it 15 min later)"
    date > $D/FINAL_FETCH_OK
    say "done: $(for F in $FOLDS; do echo -n "f$F=$(cat $D/fetched_f$F 2>/dev/null || echo ok) "; done)"; rm -f $D/fetch_loop.pid; exit 0
  fi
  sleep 600
done
