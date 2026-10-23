#!/bin/bash
# guard_sim.sh: laptop-side test of remote/guard.sh in a scratch root (no Vast call: without CONTAINER_ID/KEY the
# api() helper returns "nokey" before any network access). Checks both triggers, the wipe, and that the log never
# contains a key-like string.
set -u
K=$(cd "$(dirname "$0")/.." && pwd); SIM=${TMPDIR:-/tmp}/vast320_guard_sim; pass=0; fail=0
ok(){ echo "PASS $*"; pass=$((pass + 1)); }; no(){ echo "FAIL $*"; fail=$((fail + 1)); }
mk(){ rm -rf $SIM; mkdir -p $SIM/fleet_store $SIM/fleet_runs/f2_s1045_ttr320frac75 $SIM/hf $SIM/novel $SIM/rsna_train $SIM/fast_build $SIM/status; echo x > $SIM/fleet_store/a.npz; }
unset CONTAINER_ID CONTAINER_API_KEY
mk; RSNA_ROOT=$SIM RSNA_GUARD_DEADLINE=$(( $(date +%s) - 5 )) GUARD_TRIES=1 GUARD_RETRY_S=1 bash "$K/remote/guard.sh" 2>/dev/null
grep -q 'TRIGGER: hard deadline reached' $SIM/guard.log && ok "hard deadline triggers" || no "deadline trigger"
[ ! -e $SIM/fleet_store ] && [ ! -e $SIM/fleet_runs ] && [ ! -e $SIM/hf ] && [ ! -e $SIM/fast_build ] && ok "competition data wiped" || no "wipe"
grep -q 'destroy attempt 1: http nokey' $SIM/guard.log && grep -q 'GAVE UP' $SIM/guard.log && ok "no key -> no API call, gives up loudly" || no "api path"
mk; touch -d '20 minutes ago' $SIM/status/FINAL_FETCH_OK
RSNA_ROOT=$SIM RSNA_GUARD_DEADLINE=$(( $(date +%s) + 99999 )) GUARD_TRIES=1 GUARD_RETRY_S=1 bash "$K/remote/guard.sh" 2>/dev/null
grep -q 'TRIGGER: final fetch done 15+ min ago' $SIM/guard.log && ok "FINAL_FETCH_OK older than 15 min triggers" || no "fetch trigger"
grep -E -q '[A-Za-z0-9]{32,}' $SIM/guard.log && no "key-like string in guard.log" || ok "guard.log holds no key-like string"
B64=$(gzip -9c "$K/remote/guard.sh" | base64 -w0); echo "$B64" | base64 -d | gunzip | cmp -s - "$K/remote/guard.sh" && ok "onstart gzip+base64 round trip (${#B64} chars)" || no "onstart encoding"
cat $SIM/guard.log | sed 's/^/  log: /'
echo "== guard_sim: $pass PASS, $fail FAIL"; rm -rf $SIM; [ $fail = 0 ]
