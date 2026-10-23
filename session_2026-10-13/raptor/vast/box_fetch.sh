#!/bin/bash
# box_fetch.sh HOST PORT DEADLINE_EPOCH "43 44 45" (BOX side, setsid nohup; survives laptop disconnects).
# PULL ONLY (the box never sends data to the rental; its only write there is `touch status/FINAL_FETCH_OK`).
# Every 10 min: each seed whose status/s<seed>.rc exists on the instance is pulled (tar over ssh) into
# ~/raptor_store/rented/s<seed>/ and verified with the instance's SHA256SUMS (sha256sum -c on the box). From
# DEADLINE - 45 min, seeds still running are pulled as s<seed>_partial. A finished seed is pulled once sv_watch.sh scored it
# (status/sv_s<seed>.done; its runs/sv_s<seed>/ SV gold58 predictions come along, sha-checked) (epoch ckpts + last.pt + logs, sha-checked).
# When every seed is fetched (or skipped / partial at the deadline): touch status/FINAL_FETCH_OK on the instance (its guard
# self-destroys 15 min later) and write ~/raptor_store/rented/FINAL_FETCH_OK. Log: ~/raptor_store/rented/fetch.log.
set -u
H=$1; P=$2; DL=$3; SEEDS=$4; R=/workspace/rsna; D=$HOME/raptor_store/rented; mkdir -p $D; L=$D/fetch.log
SSH="ssh -i $HOME/.ssh/vast_rsna_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_vast -o LogLevel=ERROR -p $P root@$H"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L; }
pull(){   # seed tag(final|partial)
  local s=$1 tag=$2 dst=$D/s$1; [ $tag = partial ] && dst=$D/s$1_partial
  rm -rf $dst.tmp; mkdir -p $dst.tmp
  if [ $tag = partial ]; then $SSH "cd $R/runs/s$s && sha256sum train.log last.pt ep*.pt gold58_ep*.npy 2>/dev/null > SHA256SUMS.partial; true" || return 1; fi
  $SSH "cd $R/runs && tar cf - \$(ls -d s$s s$s.out sv_s$s sv_s$s.log 2>/dev/null)" | tar xf - -C $dst.tmp || return 1
  local sums=SHA256SUMS; [ $tag = partial ] && sums=SHA256SUMS.partial
  ( cd $dst.tmp/s$s && sha256sum -c --quiet $sums ) >> $L 2>&1 || { say "s$s $tag: sha256 check FAILED"; return 1; }
  if [ -d $dst.tmp/sv_s$s ]; then ( cd $dst.tmp/sv_s$s && sha256sum -c --quiet SHA256SUMS ) >> $L 2>&1 || { say "s$s sv: sha256 check FAILED"; return 1; }; fi
  rm -rf $dst; mv $dst.tmp/s$s $dst; mv $dst.tmp/s$s.out $dst/run.out 2>/dev/null
  [ -d $dst.tmp/sv_s$s ] && mv $dst.tmp/sv_s$s $dst/sv; mv $dst.tmp/sv_s$s.log $dst/sv.log 2>/dev/null; rm -rf $dst.tmp
  say "s$s $tag fetched + verified -> $dst ($(du -sh $dst | cut -f1); $(grep -E '^=== SWA' $dst/train.log | tail -n 1 | cut -c1-120))"
}
say "start pid $$ | instance $H:$P | seeds [$SEEDS] | deadline $(date -d @$DL '+%m-%d %H:%M')"
miss=0
while :; do
  now=$(date +%s); done_all=1
  st=$($SSH "cd $R/status 2>/dev/null && for f in s*.rc s*.skipped s*.running sv_s*.done; do [ -e \$f ] && echo \$f:\$(cat \$f); done; true" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ]; then miss=$((miss + 1)); say "instance unreachable ($miss)"; [ $now -gt $((DL + 3600)) ] && { say "past deadline + 1 h, giving up"; exit 1; }; sleep 600; continue; fi
  miss=0
  for s in $SEEDS; do
    [ -e $D/s$s/FETCHED ] && continue
    if printf '%s\n' "$st" | grep -q "^s$s.rc:" && { printf '%s\n' "$st" | grep -q "^sv_s$s.done:" || [ $now -ge $((DL - 2700)) ]; }; then
      r=$(printf '%s\n' "$st" | sed -n "s/^s$s.rc://p"); pull $s final && echo "rc $r" > $D/s$s/FETCHED || done_all=0
    elif printf '%s\n' "$st" | grep -q "^s$s.skipped:"; then mkdir -p $D/s$s; echo skipped > $D/s$s/FETCHED; say "s$s skipped on the instance"
    elif [ $now -ge $((DL - 2700)) ]; then
      pull $s partial && { mkdir -p $D/s$s; echo partial > $D/s$s/FETCHED; } || done_all=0
    else done_all=0; fi
  done
  if [ $done_all = 1 ]; then
    $SSH "touch $R/status/FINAL_FETCH_OK" && say "FINAL_FETCH_OK written on the instance (guard destroys it in 15 min)"
    date > $D/FINAL_FETCH_OK; say "all seeds fetched: $(for s in $SEEDS; do echo -n "s$s=$(cat $D/s$s/FETCHED) "; done)"; exit 0
  fi
  sleep 600
done
