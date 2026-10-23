#!/bin/bash
# auto_e2.sh: BOX side, run detached (setsid nohup). It keeps working through laptop network drops.
#  1. folds 0-3: poll /runs/_meta/fF_s1046_ttr288frac75_summary.json on the volume every 2 min; on "complete" run
#     e2_app.py::fetch (md5-verified, no epochs/*.pt) then place_fold.sh F (-> ~/fleet_runs + receipt)
#  2. fold-4 rule (coordinator 10-09): after f0-3, metered = `modal billing summary --json` metered_cost.
#     metered <= 22.0 -> timeout T = min(9900 s, (28.5 - metered - 0.30) / 2.456 $/h) [2.456 = L40S + CPU limit 8 +
#     mem limit 16 GiB, the worst-case rate], launch only if T >= 8820 s; else no launch. Decision -> logs/f4_decision.txt
#  3. fold 4 (if launched): same fetch + place
#  4. when every launched fold is placed with VERIFY_OK: delete /fleet_store from the volume
# Never touches the GPU, ~/queue, existing ~/fleet_runs dirs or any running process.
set -u
M=/home/training/modalenv/bin/modal; E=/home/training/modal_e2; L=$E/logs/auto_e2.log
say(){ echo "[auto $(date '+%m-%d %H:%M:%S')] $*" >> $L; }
nm(){ echo "f${1}_s1046_ttr288frac75"; }
summ(){ timeout 120 $M volume get rsna-e2 /runs/_meta/$(nm $1)_summary.json - 2>/dev/null | grep -v "Finished downloading"; }
placed_ok(){ grep -q '"verify": "OK"' /home/training/fleet_runs/modal_e2_$(nm $1)_receipt.json 2>/dev/null; }
fetch_place(){
  local F=$1 N; N=$(nm $1)
  placed_ok $F && { say "f$F already placed (VERIFY_OK)"; return 0; }
  for t in 1 2 3; do
    timeout 1800 $M run $E/e2_app.py::fetch --run $N --dest $E/dl >> $E/logs/fetch_f$F.out 2>&1 < /dev/null
    grep -q "FETCH_OK $N" $E/logs/fetch_f$F.out && break
    say "f$F fetch attempt $t failed: $(tail -n 2 $E/logs/fetch_f$F.out | tr '\n' ' ' | cut -c1-200)"; sleep 60
  done
  grep -q "FETCH_OK $N" $E/logs/fetch_f$F.out || { say "f$F FETCH FAILED after 3 attempts"; return 1; }
  say "f$F $(grep "FETCH_OK $N" $E/logs/fetch_f$F.out | tail -n 1)"
  bash $E/place_fold.sh $F >> $E/logs/place_f$F.out 2>&1
  say "f$F place: $(tail -n 1 $E/logs/place_f$F.out | cut -c1-400)"
  placed_ok $F
}
wait_folds(){   # wait_folds LAUNCHLOG F.. : resolve each fold to complete+placed / FAILED
  local LOG=$1; shift; local pend="$*" F s newp
  while [ -n "$pend" ]; do
    newp=""
    for F in $pend; do
      s=$(summ $F)
      if printf '%s' "$s" | grep -q '"status": "complete"'; then
        say "f$F complete on Modal: $(printf '%s' "$s" | grep -E '"closing"|"train_min"' | tr -d '\n' | cut -c1-200)"
        fetch_place $F || say "f$F NOT placed"
      elif printf '%s' "$s" | grep -q '"status": "FAILED"' || grep -q "fold $F ERROR" $LOG 2>/dev/null; then
        say "f$F FAILED on Modal: $(grep "fold $F ERROR" $LOG | cut -c1-300)"; echo "f$F" >> $E/logs/failed_folds.txt
      else newp="$newp $F"; fi
    done
    pend=$(echo $newp)
    [ -n "$pend" ] && sleep 120
  done
}
say "start: waiting for folds 0-3"
wait_folds $E/logs/train_f0-3.out 0 1 2 3
sleep 180    # let the bill settle after the last container stops
MET=$(timeout 120 $M billing summary --json 2>/dev/null | /home/training/modalenv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["metered_cost"])')
say "folds 0-3 resolved; metered (billing summary) = \$$MET"
DEC=$(/home/training/modalenv/bin/python - "$MET" <<'EOF'
import sys
try: m = float(sys.argv[1])
except Exception: print("NOLAUNCH metered unreadable"); sys.exit()
if m > 22.0: print("NOLAUNCH metered %.2f > 22.0" % m); sys.exit()
t = min(9900, int((28.5 - m - 0.30) / 2.456 * 3600))
print(("LAUNCH %d" % t) if t >= 8820 else "NOLAUNCH timeout %d s < 8820 s" % t)
EOF
)
echo "$(date '+%F %T') metered \$$MET -> $DEC" > $E/logs/f4_decision.txt; say "fold-4 rule: $DEC"
LAUNCHED="0 1 2 3"
case $DEC in LAUNCH*)
  T=${DEC#LAUNCH }
  sed -i -E "s/^TRAIN_TIMEOUT_S = [0-9]+ .*/TRAIN_TIMEOUT_S = $T         # f4: worst case keeps the month's metered total <= \$28.5 (coordinator rule)/" $E/e2_app.py
  grep -q "^TRAIN_TIMEOUT_S = $T " $E/e2_app.py || { say "could not set the f4 timeout; NOT launching"; exit 1; }
  E2_FOLDS=4 setsid nohup $M run --detach $E/e2_app.py::train > $E/logs/train_f4.out 2>&1 < /dev/null &
  say "launched fold 4 on L40S, timeout ${T}s (launcher pid $!)"
  LAUNCHED="0 1 2 3 4"
  sleep 300
  wait_folds $E/logs/train_f4.out 4;;
esac
ALL=1; for F in $LAUNCHED; do placed_ok $F || ALL=0; done
if [ $ALL = 1 ]; then
  timeout 600 $M volume rm -r rsna-e2 /fleet_store >> $L 2>&1 && say "deleted /fleet_store from the volume rsna-e2" || say "volume rm /fleet_store FAILED"
else
  say "not every launched fold is placed: /fleet_store kept on the volume for a rerun"
fi
say "DONE (metered now \$$(timeout 120 $M billing summary --json 2>/dev/null | /home/training/modalenv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["metered_cost"])'))"
