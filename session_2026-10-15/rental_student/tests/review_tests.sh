#!/bin/bash
# review_tests.sh (box CPU, seconds; only under ~/rstud_tests/review): controls for the 10-14 review fixes
set -u
T=$HOME/rstud_tests/review; rm -rf $T; mkdir -p $T; PY=$HOME/miniconda3/envs/rsna/bin/python; PC=$HOME/rstud/src/box/pack_check.py
G=$HOME/rstud/src/remote/guard.sh; SIM=$HOME/r7t_labels/sim_a1/final/b1; FRC=$HOME/r7t_labels/fleet/fleet_oof_receipt_fleetOF.json
TAB=$SIM/SIM_labels_r7t_b.csv; RC=$SIM/SIM_teacher_labels_receipt.json; n=0; bad=0
chk(){ n=$((n + 1)); if [ "$2" = "$3" ]; then echo "ok | $1 | $3"; else echo "WRONG | $1 | want $2 got $3"; bad=$((bad + 1)); fi; }
pc(){ "$@" > $T/out 2>&1; rc=$?; [ $rc = 0 ] && echo PASS || { [ $rc = 3 ] && echo STOP || echo ERR$rc; }; }
# pack_check: fleet gold component
$PY -c "import json; r=json.load(open('$RC')); r['inputs']['fleet_gold']['md5']='0'*32; json.dump(r, open('$T/rc_gold.json','w'))"
chk "receipt fleet GOLD md5 != the OAI-free 25-member gold mean" STOP $(pc $PY $PC table $TAB --mode DRY --receipt $T/rc_gold.json)
$PY -c "import json; r=json.load(open('$RC')); r['inputs']['fleet_oof']['sha256']='1'*64; json.dump(r, open('$T/rc_oofsha.json','w'))"
chk "build used a fleet OOF that is not the fleet receipt mean" STOP $(pc $PY $PC table $TAB --mode DRY --receipt $T/rc_oofsha.json --fleet_receipt $FRC)
$PY -c "import json; r=json.load(open('$FRC')); r['fleet_mean']['members']=['C','D','E','E2','DT']; json.dump(r, open('$T/fr_members.json','w'))"
chk "fleet receipt mean with a DT member" STOP $(pc $PY $PC table $TAB --mode DRY --receipt $RC --fleet_receipt $T/fr_members.json)
$PY -c "import json; r=json.load(open('$FRC')); r['sets']['E']['tag']='s1045_ttr288l13'; json.dump(r, open('$T/fr_etag.json','w'))"
chk "fleet set E with a different tag" STOP $(pc $PY $PC table $TAB --mode DRY --receipt $RC --fleet_receipt $T/fr_etag.json)
chk "clean DRY (SIM receipt + fleet receipt) still passes" PASS $(pc $PY $PC table $TAB --mode DRY --receipt $RC --fleet_receipt $FRC)
# guard: stale guard.pid (container restart) must not stop a new guard; a live guard must not be doubled
r=$T/g1; mkdir -p $r/status $r/fleet_store; echo x > $r/fleet_store/a.npz; sleep 60 & sp=$!; echo $sp > $r/status/guard.pid; echo "x" > $r/status/ABORT
env GUARD_MOCK=1 GUARD_POLL_S=1 GUARD_RETRY_S=0 GUARD_TRIES=1 RSNA_ROOT=$r timeout 6 bash $G; kill $sp 2>/dev/null
chk "stale guard.pid = a live NON-guard process: the new guard runs + triggers" yes $(grep -q TRIGGER $r/guard.log 2>/dev/null && echo yes || echo no)
r=$T/g2; mkdir -p $r/status $r/fleet_store; echo x > $r/fleet_store/a.npz
( env GUARD_MOCK=1 GUARD_POLL_S=1 RSNA_ROOT=$r RSNA_GUARD_DEADLINE=$(( $(date +%s) + 3600 )) timeout 8 bash $G & ); sleep 2
env GUARD_MOCK=1 GUARD_POLL_S=1 RSNA_ROOT=$r RSNA_GUARD_DEADLINE=$(( $(date +%s) + 3600 )) timeout 4 bash $G
chk "second guard while the first runs: exits (one start line)" 1 $(grep -c 'start pid' $r/guard.log)
sleep 7
# fetch_loop table-sha selection (lines extracted from the real script)
mkdir -p $T/f/state_X $T/f/bundle $T/f/fetch $T/final; L=$T/f/fetch/fetch.log
sel(){ ( D=$T/f; INSTANCE_ID=X; BUNDLE=$T/f/bundle; HOME_=$HOME; say(){ echo "$*" >> $L; }
        eval "$(sed -n '/^BJ=\$D\/state_/,/^say "verify table sha256/p' $HOME/rstud/src/box/fetch_loop.sh | sed "s#\$HOME/r7t_labels/final/LABELS_READY#$T/final/LABELS_READY#")"; echo "$TSHA" ); }
A=$(printf 'a%.0s' $(seq 1 64)); Bh=$(printf 'b%.0s' $(seq 1 64))
echo "{\"mode\": \"REAL\", \"table_sha256\": \"$A\"}" > $T/f/state_X/BUNDLE.json; echo "{\"mode\": \"DRY\", \"table_sha256\": \"$Bh\"}" > $T/f/bundle/BUNDLE.json
echo "table T_b labels_r7t_b.csv sha256 $A md5 x seed 1047" > $T/final/LABELS_READY
chk "fetch: snapshot (pushed, REAL) wins over a later DRY re-pack; sha in LABELS_READY" ${A:0:8} $(sel | cut -c1-8)
echo "table T_b labels_r7t_b.csv sha256 $Bh md5 x seed 1047" > $T/final/LABELS_READY
chk "fetch: pushed table not the LABELS_READY T_b table -> nothing placed" NOT_IN_LABELS_READY $(sel)
rm $T/f/state_X/BUNDLE.json; chk "fetch: no snapshot (tests) -> bundle BUNDLE.json (DRY: no LABELS_READY check)" ${Bh:0:8} $(sel | cut -c1-8)
echo "== $n review checks, $bad WRONG"
