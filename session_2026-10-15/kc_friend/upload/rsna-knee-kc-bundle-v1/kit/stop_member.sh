#!/bin/bash
# stop_member.sh KEY STAGE (rental_kc instance; box driver: end of a smoke phase, or shedding a member). Kills ONLY verified
# PIDs (never pkill): the lane whose spec <G>:<keys>:STAGE holds KEY (cmdline must contain kc_lane.sh), then the trainer from the lane's
# pid file (cmdline must contain fleet_train.py and the member's out dir), then its DataLoader children recorded before the
# kill. smoke: the probe dir is removed (its log kept as <out>.log in $R/smoke). real: the partial run is moved aside
# (never deleted) and the member's status files cleared. Prints what it did.
set -u
R=${RSNA_ROOT:-/workspace/rsna}; S=$R/status; FR=$R/fleet_runs; K=$R/kit
KEY=$1; STAGE=${2:-real}; PFX=$KEY; [ "$STAGE" = smoke ] && PFX=smoke_$KEY
N=$(awk -F'\t' -v k=$KEY '$1==k {print $2}' $K/members.tsv); [ -n "$N" ] || { echo "no member $KEY"; exit 1; }
cmdl(){ tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null; }
for sp in $S/lane_*.spec; do [ -e "$sp" ] || continue
  spec=$(cat "$sp"); [ "${spec##*:}" = "$STAGE" ] || continue
  case ",$(echo "$spec" | cut -d: -f2)," in *",$KEY,"*) id=$(basename "$sp" .spec); id=${id#lane_}
    p=$(cat $S/lane_$id.pid 2>/dev/null)
    if [ -n "$p" ] && cmdl $p | grep -q 'kc_lane[.]sh'; then kill "$p" 2>/dev/null && echo "killed lane $id (pid $p, spec $(cat "$sp"))"; fi
    rm -f $S/lane_$id.pid; mv -f "$sp" "$sp.stopped";;
  esac
done
tp=$(cat $S/$PFX.tpid 2>/dev/null)
if [ -n "$tp" ] && cmdl $tp | grep -q 'fleet_train[.]py' && cmdl $tp | grep -q -- "--out [^ ]*/$N\( \|$\)"; then
  kids=$( { pgrep -P $tp 2>/dev/null || cat /proc/$tp/task/*/children 2>/dev/null; } | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')
  kill $tp 2>/dev/null && echo "killed trainer $N (pid $tp; loader children: ${kids:-none})"
  for i in $(seq 1 30); do kill -0 $tp 2>/dev/null || break; sleep 2; done
  kill -0 $tp 2>/dev/null && cmdl $tp | grep -q 'fleet_train[.]py' && { kill -9 $tp; sleep 3; echo "trainer $N needed SIGKILL"; }
  sleep 5
  for c in $kids; do cmdl $c | grep -q 'python' && kill -0 $c 2>/dev/null && { kill -9 $c 2>/dev/null; echo "killed leftover loader child $c"; }; done
elif [ -n "$tp" ]; then echo "trainer pid $tp is not the $N trainer any more: not killed"; fi
rm -f $S/$PFX.tpid
if [ "$STAGE" = smoke ]; then rm -rf $R/smoke/$N; echo "smoke dir of $KEY removed (log kept)"
else A=$FR/${N}_stopped_$(date +%m%d_%H%M%S)
  [ -e $FR/$N ] && mv -n $FR/$N $A && echo "moved $N aside to $(basename $A)"
  [ -e $FR/$N.log ] && mv -n $FR/$N.log $A.log
  echo "stopped by the box driver $(date '+%F %T')" > $S/$KEY.stopped; fi
rm -f $S/$PFX.running $S/$PFX.failed $S/$PFX.done
echo "$KEY ($STAGE) stopped"
