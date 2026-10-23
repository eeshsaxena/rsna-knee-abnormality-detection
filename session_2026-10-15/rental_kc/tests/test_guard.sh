#!/bin/bash
# test_guard.sh (rental_kc; BOX or any Linux; GUARD_MOCK=1: no kill, no API call; writes only under ~/rkc_tests/guard): every
# trigger of remote/guard.sh fires (and wipes the mock root), the deadline can be tightened but never extended, a healthy run
# does not trigger, and the orphan fuse needs RUN_STARTED + every planned member terminal.
set -u
G=$HOME/rkc/src/remote/guard.sh; T=$HOME/rkc_tests/guard; rm -rf $T; mkdir -p $T; n=0; bad=0
mk(){ local r=$T/$1; mkdir -p $r/status $r/kc_store/store $r/kit; echo x > $r/kc_store/store/a.npz; echo "$r"; }
run(){ # name want(trig|none) secs root [env...]
  local name=$1 want=$2 secs=$3 r=$4; shift 4; n=$((n + 1))
  env GUARD_MOCK=1 GUARD_POLL_S=1 GUARD_RETRY_S=0 GUARD_TRIES=1 RSNA_ROOT=$r "$@" timeout $secs bash $G; local rc=$?
  local got=none; grep -q 'TRIGGER' $r/guard.log 2>/dev/null && got=trig
  local wiped=no; [ ! -e $r/kc_store ] && wiped=yes
  [ "$got" = "$want" ] && { [ $want = none ] || [ $wiped = yes ]; } && res=ok || { res=WRONG; bad=$((bad + 1)); }
  echo "$res | $name | expect $want got $got (wiped $wiped) | $(grep -E 'TRIGGER|destroy attempt' $r/guard.log | tail -n 2 | sed 's/^\[guard [^]]*\] //' | tr '\n' ' ' | cut -c1-150)"; }
now=$(date +%s)
r=$(mk abort); echo "smoke FAIL" > $r/status/ABORT; run "ABORT file" trig 6 $r RSNA_GUARD_DEADLINE=$((now + 3600))
r=$(mk fin); touch -d '-20 min' $r/status/FINAL_FETCH_OK; run "FINAL_FETCH_OK 20 min old" trig 6 $r RSNA_GUARD_DEADLINE=$((now + 3600))
r=$(mk finnew); touch $r/status/FINAL_FETCH_OK; touch $r/status/RUN_STARTED; run "FINAL_FETCH_OK fresh (no trigger yet)" none 4 $r RSNA_GUARD_DEADLINE=$((now + 3600))
r=$(mk hard); run "hard deadline" trig 8 $r RSNA_GUARD_DEADLINE=$(( $(date +%s) + 2 ))
r=$(mk tight); echo $(( $(date +%s) + 2 )) > $r/status/guard_deadline; run "deadline tightened by status/guard_deadline" trig 8 $r RSNA_GUARD_DEADLINE=$((now + 3600))
r=$(mk noext); echo $(( $(date +%s) + 3600 )) > $r/status/guard_deadline; run "deadline NOT extended (env +3 s, file +1 h)" trig 9 $r RSNA_GUARD_DEADLINE=$(( $(date +%s) + 3 ))
r=$(mk boot); run "boot fuse (no RUN_STARTED)" trig 8 $r RSNA_GUARD_DEADLINE=$((now + 3600)) RSNA_BOOT_DEADLINE=$(( $(date +%s) + 2 ))
r=$(mk bootok); touch $r/status/RUN_STARTED; run "boot fuse with RUN_STARTED (no trigger)" none 5 $r RSNA_GUARD_DEADLINE=$((now + 3600)) RSNA_BOOT_DEADLINE=$(( $(date +%s) + 1 ))
r=$(mk orphan); touch $r/status/RUN_STARTED; printf 'MEMBERS="s2003 s2002d2"\n' > $r/kit/plan.env; touch $r/status/s2003.done $r/status/s2002d2.skipped; touch -d '-10 min' $r/status/box_seen
run "orphan fuse (members terminal, box silent)" trig 8 $r RSNA_GUARD_DEADLINE=$((now + 3600)) GUARD_ORPHAN_AGE_S=3
r=$(mk running); touch $r/status/RUN_STARTED; printf 'MEMBERS="s2003 s2002d2"\n' > $r/kit/plan.env; touch $r/status/s2003.done; echo "$now 1" > $r/status/s2002d2.running; touch -d '-10 min' $r/status/box_seen
run "member still running (no orphan trigger)" none 5 $r RSNA_GUARD_DEADLINE=$((now + 3600)) GUARD_ORPHAN_AGE_S=3
r=$(mk prerun); printf 'MEMBERS=""\n' > $r/kit/plan.env; touch -d '-10 min' $r/status/box_seen
run "before RUN_STARTED (gate wait): no orphan trigger" none 5 $r RSNA_GUARD_DEADLINE=$((now + 3600)) GUARD_ORPHAN_AGE_S=3
echo "== $n guard checks, $bad WRONG"
