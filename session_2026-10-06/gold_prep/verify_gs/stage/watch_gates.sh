#!/bin/bash
# watch_gates.sh (2026-10-06 ~22:50 IST, Claude; gold_prep gate scorers). CPU ONLY, READ ONLY on runs.
# Polls every 5 min. When a registered run's log gets the trainer's closing line {"raw": .., "ema": ..}, runs
# score_run.py --gate <g> and blend_plan.py ONCE and writes ~/gold_prep/gates/out/<g>_final.txt + blend_after_<g>.txt.
# While a registered run is still training, refreshes out/<g>_partial.txt every 30 min (PROVISIONAL, gate PENDING).
# If the R50 probe exits without a closing line (auto_r50probe.log says "probe exited"), writes r50_final.txt with the
# NOT FINISHED reason plus a --partial read.
# Never touches the GPU (CUDA_VISIBLE_DEVICES=), the chains, PAUSE_CHAIN, ~/fleet_runs (read only) or any process.
# Writes only under ~/gold_prep/gates/out. Exits when both gates are final, or at 2026-10-08 12:00 IST.
# verify fixes 10-06 ~22:45 IST: (1) the R50 no-closing-line branch keys on auto_r50probe.log ("probe exited", "refusing",
# "giving up"), not on the run DIR, which the trainer creates only after model build (a crash before that left the
# watcher waiting to the deadline); it re-checks the closing line first (race). (2) closed() finds the trainer's closing
# line anywhere in the log (it is printed once, after the last epoch), so a trailing shutdown line cannot hide it.
PY=/home/training/miniconda3/envs/rsna/bin/python
D=${WG_D:-/home/training/gold_prep/gates}; O=${WG_OUT:-$D/out}; LOG=$O/watch_gates.log; FR=${WG_FR:-/home/training/fleet_runs}
NAP=${WG_SLEEP:-300}   # WG_* overrides exist for the dry tests only (test/run_watch_test.sh, staging)
mkdir -p $O
say(){ echo "[watch-gates $(date '+%m-%d %H:%M')] $*" >> $LOG; }
closed(){ grep -q '^{"raw": ' "$1" 2>/dev/null; }
pyrun(){ nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES= $PY "$@"; }
DEADLINE=$(date -d "2026-10-08 12:00" +%s)
T288=f0_s1044_ttr288mix50; R50=f0_s243_r50mix50
CANDS="f0_s42 f0_s1042_ttr f0_s142_f2v4"
done_t=0; done_r=0; last_partial=0
say "armed (pid $$)"
while [ $done_t -eq 0 ] || [ $done_r -eq 0 ]; do
  now=$(date +%s)
  if [ $now -gt $DEADLINE ]; then say "deadline reached (t288 done=$done_t r50 done=$done_r), exiting"; exit 0; fi
  if [ $done_t -eq 0 ] && closed $FR/$T288.log; then
    pyrun $D/score_run.py --runs_root $FR --gate t288 --json $O/t288_final.json > $O/t288_final.txt 2>&1
    pyrun $D/blend_plan.py --runs_root $FR $T288 $CANDS --json $O/blend_after_t288.json > $O/blend_after_t288.txt 2>&1
    G=$(grep '^GATE ' $O/t288_final.txt | head -1); [ -z "$G" ] && G="GATE T288 $T288: FAIL (unscorable, see t288_final.txt)"
    say "$G | $(grep '^best tag' $O/t288_final.txt | cut -c1-90)"; echo "[$(date '+%m-%d %H:%M')] $G" >> $O/gate_history.log
    done_t=1
  fi
  if [ $done_r -eq 0 ] && closed $FR/$R50.log; then
    pyrun $D/score_run.py --runs_root $FR --gate r50 --json $O/r50_final.json > $O/r50_final.txt 2>&1
    pyrun $D/blend_plan.py --runs_root $FR $R50 $T288 $CANDS --partial --json $O/blend_after_r50.json > $O/blend_after_r50.txt 2>&1
    G=$(grep '^GATE ' $O/r50_final.txt | head -1); [ -z "$G" ] && G="GATE R50 $R50: FAIL (unscorable, see r50_final.txt)"
    say "$G"; echo "[$(date '+%m-%d %H:%M')] $G" >> $O/gate_history.log
    done_r=1
  elif [ $done_r -eq 0 ] && PX=$(grep -E 'probe exited|refusing|giving up' $FR/auto_r50probe.log 2>/dev/null | tail -n 1) \
       && [ -n "$PX" ] && ! closed $FR/$R50.log; then
    { echo "R50 probe exited WITHOUT a closing line: $PX";
      echo "last log lines:"; tail -n 5 $FR/$R50.log; echo;
      pyrun $D/score_run.py --runs_root $FR --gate r50 --partial; } > $O/r50_final.txt 2>&1
    say "R50 probe exited without a closing line: GATE R50 FAIL (not finished); see out/r50_final.txt"
    echo "[$(date '+%m-%d %H:%M')] GATE R50 $R50: FAIL (exited without closing line) | probe: ${PX#*] }" >> $O/gate_history.log
    done_r=1
  fi
  if [ $((now - last_partial)) -ge 1800 ]; then
    [ $done_t -eq 0 ] && [ -e $FR/$T288 ] && pyrun $D/score_run.py --runs_root $FR --gate t288 --partial --brief > $O/t288_partial.txt 2>&1
    [ $done_r -eq 0 ] && [ -e $FR/$R50 ] && pyrun $D/score_run.py --runs_root $FR --gate r50 --partial --brief > $O/r50_partial.txt 2>&1
    last_partial=$now
  fi
  sleep $NAP
done
say "both gates final, exiting"
