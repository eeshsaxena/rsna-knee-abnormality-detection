#!/bin/bash
# ~/kc/code/kc_gate_watch.sh -- KC gate helper (BOX_PLAN_CRITIQUE_1015, adopted; PREREG_10-15 A6). Never touches ~/queue,
# never overwrites an existing PASS / FAIL line, exits when its work is done. Two phases:
# PHASE 1, gate-deadlock fix for ~/kc/gate/kc_f0.verdict (the AD / speculative lines wait on it with 'wait 1'). The f0
#   end hook (fleet_train_kc.py KC-5) writes it, and an atexit path writes FAIL on any Python exception / sys.exit /
#   SIGTERM. This watcher writes 'FAIL kc_f0 not run/crashed (...)' when, with no verdict line yet,
#     - ~/kc/KC_READY.verdict has a FAIL line (the f0 line will #SKIP), or
#     - ~/queue/history.txt shows the f0 line SKIP / BAD / REFUSED, or its FAILED / DONE end line (SIGKILL, segfault,
#       power loss) and the verdict is still missing GRACE s (10 min) after it.
# PHASE 2, the pre-registered s2002 label switch (A6): once kc_f0.verdict has a verdict, it writes
#     ~/kc/gate/s2002_tb.verdict      PASS iff f0 PASS and ~/kc/gate/s_read.verdict says PASS (S moved the best), else FAIL
#     ~/kc/gate/s2002_dread2.verdict  PASS iff f0 PASS and s_read.verdict says FAIL (S not positive), else FAIL
#   f0 FAIL -> both FAIL at once. f0 PASS -> waits for MAIN to write s_read.verdict ('PASS S positive ...' or
#   'FAIL S not positive ...'); until then both s2002 lines DEFER (s2003 / s2004 run).
# Arm (main): setsid nohup bash ~/kc/code/kc_gate_watch.sh > ~/kc/logs/gate_watch.log 2>&1 < /dev/null &
set -u
G=${KW_GATE_DIR:-$HOME/kc/gate}
V=${KW_VERDICT:-$G/kc_f0.verdict}
READY=${KW_READY:-$HOME/kc/KC_READY.verdict}
HIST=${KW_HIST:-$HOME/queue/history.txt}
SREAD=${KW_SREAD:-$G/s_read.verdict}
TB=${KW_TB:-$G/s2002_tb.verdict}; D2=${KW_D2:-$G/s2002_dread2.verdict}
NAME=${KW_NAME:-KC1 f0 s2001}
POLL=${KW_POLL:-120}; GRACE=${KW_GRACE:-600}
RE='^[[:space:]]*(VERDICT[[:space:]]*[:=]?[[:space:]]*)?(PASS|FAIL)([^A-Za-z0-9_]|$)'
verdict(){ [ -f "$1" ] && grep -Eq "$RE" "$1"; }
vword(){ grep -Em1 "$RE" "$1" | sed -E 's/^[[:space:]]*(VERDICT[[:space:]]*[:=]?[[:space:]]*)?(PASS|FAIL).*/\2/'; }
put(){ verdict "$1" && { echo "$(date '+%F %T') $1 already has a verdict; kept"; return; }
       printf '%s\n# written by kc_gate_watch.sh %s\n' "$2" "$(date '+%F %T')" > "$1.watch.tmp" && mv -f "$1.watch.tmp" "$1"
       echo "$(date '+%F %T') wrote $1: $2"; }
echo "$(date '+%F %T') kc_gate_watch start | f0 verdict $V | ready $READY | history $HIST | s_read $SREAD | name '$NAME'"
seen_end=0
while ! verdict "$V"; do                                                       # PHASE 1
  if [ -f "$READY" ] && grep -Eq '^[[:space:]]*FAIL' "$READY"; then
    put "$V" "FAIL kc_f0 not run/crashed (KC_READY FAIL: $(grep -Em1 '^[[:space:]]*FAIL' "$READY" | cut -c1-100))"; break
  fi
  h=$(grep -F "$NAME" "$HIST" 2>/dev/null | grep -E '^[0-9-]+ [0-9:]+ (SKIP|BAD|REFUSED|FAILED|DONE) ' | tail -n 1)
  case "$h" in
    *" SKIP "*|*" BAD "*|*" REFUSED "*) put "$V" "FAIL kc_f0 not run/crashed (history: $(printf '%s' "$h" | cut -c1-140))"; break;;
    *" FAILED "*|*" DONE "*)
      [ $seen_end -eq 0 ] && { seen_end=$(date +%s); echo "$(date '+%F %T') f0 end line seen, grace ${GRACE}s: $(printf '%s' "$h" | cut -c1-120)"; }
      if [ $(( $(date +%s) - seen_end )) -ge "$GRACE" ]; then
        put "$V" "FAIL kc_f0 not run/crashed (history end line without a verdict: $(printf '%s' "$h" | cut -c1-140))"; break
      fi;;
  esac
  sleep "$POLL"
done
f0=$(vword "$V"); echo "$(date '+%F %T') f0 verdict: $f0"
if [ "$f0" != PASS ]; then
  put "$TB" "FAIL s2002_tb (kc_f0 $f0)"; put "$D2" "FAIL s2002_dread2 (kc_f0 $f0)"; exit 0
fi
until verdict "$SREAD"; do sleep "$POLL"; done                                # PHASE 2
s=$(vword "$SREAD"); echo "$(date '+%F %T') S read verdict: $s ($(grep -Em1 "$RE" "$SREAD" | cut -c1-100))"
if [ "$s" = PASS ]; then
  put "$TB" "PASS s2002_tb (kc_f0 PASS; S positive)"; put "$D2" "FAIL s2002_dread2 (S positive: s2002 trains on T_b)"
else
  put "$TB" "FAIL s2002_tb (S not positive: s2002 = dread seed 2)"; put "$D2" "PASS s2002_dread2 (kc_f0 PASS; S not positive)"
fi
