#!/bin/bash
# run_watch_test.sh (2026-10-06, Claude): dry test of watch_gates.sh on FAKE run roots made of read-only symlinks to
# finished runs (and the live TTR288 dir for the partial path). CPU only; writes only under ~/gold_prep/gates/test/.
T=/home/training/gold_prep/gates/test; W=/home/training/gold_prep/gates/watch_gates.sh; FR0=/home/training/fleet_runs
pass=0; fail=0
ok(){ if eval "$2"; then echo "  PASS $1"; pass=$((pass+1)); else echo "  FAIL $1"; fail=$((fail+1)); fi; }
mk(){   # root t288_src r50_src
  local R=$T/$1; rm -rf $R; mkdir -p $R
  for n in f0_s1043_ttr_mix50 f0_s42 f0_s1042_ttr f0_s142_f2v4; do ln -s $FR0/$n $R/$n; ln -s $FR0/$n.log $R/$n.log; done
  [ -n "$2" ] && { ln -s $FR0/$2 $R/f0_s1044_ttr288mix50; ln -s $FR0/$2.log $R/f0_s1044_ttr288mix50.log; }
  [ -n "$3" ] && { ln -s $FR0/$3 $R/f0_s243_r50mix50; ln -s $FR0/$3.log $R/f0_s243_r50mix50.log; }
  return 0
}
echo "A: both closed (t288 <- f0_s42 .8811 -> FAIL; r50 <- f0_s142_f2v4 -> FAIL)"
mk frA f0_s42 f0_s142_f2v4; rm -rf $T/outA
WG_FR=$T/frA WG_OUT=$T/outA WG_SLEEP=1 timeout 300 bash $W; ok "A exits by itself" "[ $? -eq 0 ]"
ok "A t288 FAIL" "grep -q 'GATE T288 f0_s1044_ttr288mix50: FAIL' $T/outA/gate_history.log"
ok "A r50 FAIL" "grep -q 'GATE R50 f0_s243_r50mix50: FAIL' $T/outA/gate_history.log"
ok "A blend files" "[ -s $T/outA/blend_after_t288.txt ] && [ -s $T/outA/blend_after_r50.txt ] && grep -q '^2-MEMBER' $T/outA/blend_after_r50.txt"
ok "A json" "[ -s $T/outA/t288_final.json ] && [ -s $T/outA/r50_final.json ]"
echo "B: t288 <- f1_s1043_ttr_mix50 (.8926, other fold) -> PASS; r50 <- diverged EV2S (no closing) + probe exited"
mk frB f1_s1043_ttr_mix50 f0_s143_f2mix50ttr; rm -rf $T/outB
echo "[auto-r50probe 10-07 04:00] probe exited rc=1 (ep 3 it 0/1733 loss nan)" > $T/frB/auto_r50probe.log
WG_FR=$T/frB WG_OUT=$T/outB WG_SLEEP=1 timeout 300 bash $W; ok "B exits by itself" "[ $? -eq 0 ]"
ok "B t288 PASS" "grep -q 'GATE T288 f0_s1044_ttr288mix50: PASS' $T/outB/gate_history.log"
ok "B r50 exited-without-closing FAIL" "grep -q 'exited without closing line' $T/outB/gate_history.log"
ok "B r50_final explains" "grep -q 'WITHOUT a closing line' $T/outB/r50_final.txt"
echo "C: t288 <- LIVE f0_s1044_ttr288mix50 (no closing yet), no r50 dir -> partial file, keeps waiting"
mk frC f0_s1044_ttr288mix50 ""; rm -rf $T/outC
WG_FR=$T/frC WG_OUT=$T/outC WG_SLEEP=5 timeout 40 bash $W; ok "C still waiting at timeout (rc 124)" "[ $? -eq 124 ]"
ok "C partial written, PENDING" "grep -q 'GATE T288 f0_s1044_ttr288mix50: PENDING' $T/outC/t288_partial.txt"
ok "C no final" "[ ! -e $T/outC/t288_final.txt ] && [ ! -e $T/outC/gate_history.log ]"
ok "C no r50 partial (no dir)" "[ ! -e $T/outC/r50_partial.txt ]"
echo "side effects: fleet_runs untouched by the test (no new entries named in the test roots)"
ok "no PAUSE_CHAIN change" "[ -e $FR0/PAUSE_CHAIN ] || grep -q 'PAUSE_CHAIN removed' $FR0/auto_r50probe.log"
echo "RESULT: $pass pass, $fail fail"
