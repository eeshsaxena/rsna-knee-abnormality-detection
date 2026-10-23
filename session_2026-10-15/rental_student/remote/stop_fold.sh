#!/bin/bash
# stop_fold.sh F (instance; box driver after the smoke when fold F must leave its paired GPU): kills the lane(s) whose spec
# holds F, then F's trainer, moves the partial run aside (never deleted) and clears F's status files. Prints what it did.
set -u
R=${RSNA_ROOT:-/workspace/rsna}; S=$R/status; FR=$R/fleet_runs; . $R/kit/kit.env
F=$1; N=f${F}_s${SEED}_${RUNTAG}
for sp in $S/lane_*.spec; do [ -e "$sp" ] || continue
  case ",$(cut -d: -f2 "$sp")," in *",$F,"*) id=$(basename "$sp" .spec); id=${id#lane_}
    p=$(cat $S/lane_$id.pid 2>/dev/null); [ -n "$p" ] && kill "$p" 2>/dev/null && echo "killed lane $id (pid $p, spec $(cat "$sp"))"
    rm -f $S/lane_$id.pid; mv -f "$sp" "$sp.stopped";;
  esac
done
tp=$(cat $S/f$F.tpid 2>/dev/null)                      # the lane records its trainer pid (works without procps)
[ -n "$tp" ] && kill "$tp" 2>/dev/null && echo "killed trainer $N (pid $tp)"
command -v pkill > /dev/null && pkill -f -- "fleet_train[.]py .*--out $FR/$N( |\$)" && echo "killed trainer $N (pkill)"
for i in $(seq 1 30); do { [ -n "$tp" ] && kill -0 "$tp" 2>/dev/null; } || break; sleep 2; done
[ -n "$tp" ] && kill -0 "$tp" 2>/dev/null && { kill -9 "$tp"; sleep 3; }
command -v pgrep > /dev/null && pgrep -f -- "fleet_train[.]py .*--out $FR/$N( |\$)" > /dev/null && { pkill -9 -f -- "fleet_train[.]py .*--out $FR/$N( |\$)"; sleep 3; }
rm -f $S/f$F.tpid
A=$FR/${N}_smokepaired_$(date +%m%d_%H%M%S)
[ -e $FR/$N ] && mv -n $FR/$N $A && echo "moved $N aside to $(basename $A)"
[ -e $FR/$N.log ] && mv -n $FR/$N.log $A.log
rm -f $S/f$F.running $S/f$F.failed $S/f$F.skipped $S/f$F.done
echo "f$F stopped"
