#!/bin/bash
# test_queue_lines.sh (kaggle_frac75): read-only checks of the RUNBOOK's queue edits against the LIVE ~/queue files.
# Works on copies under ~/kt_frac75/test/q (never writes ~/queue). queue_runner.sh --check is read-only.
set -u
KT=$HOME/kt_frac75; T=$KT/test/q; L=$KT/box/queue_lines.txt; Q=$HOME/queue/queue.txt; R=$HOME/queue/queue_runner.sh
rm -rf "$T"; mkdir -p "$T"
pass=0; fail=0
ok(){ if [ "$1" = 0 ]; then echo "PASS $2"; pass=$((pass+1)); else echo "FAIL $2"; fail=$((fail+1)); fi; }
line(){ grep -v '^#' "$L" | grep "^$1 "; }
Q1B=$(line Q1b); Q2F1=$(line 'Q2 frac75 f1'); Q1=$(line 'Q1 frac75 f0')
grep -qxF "$Q1" "$Q"; ok $? "Q1 line == the live queue.txt Q1 line (byte for byte)"
grep -qxF "$Q2F1" "$Q"; ok $? "Q2 f1 line == the live queue.txt Q2 f1 line (byte for byte)"
a1=$(printf '%s' "$Q1" | cut -d'|' -f4- | sed 's#f0_s1043_ttr_frac75 #f0_s1043_ttr_frac75_b24 #')
ab=$(printf '%s' "$Q1B" | cut -d'|' -f4-)
[ "$a1" = "$ab" ]; ok $? "Q1b argv == Q1 argv with only --out renamed to f0_s1043_ttr_frac75_b24"
source $HOME/novel/frac75/argv_frac75.sh; frac75_argv 0
reg=$(printf '%s ' "${FRAC75_ARGV[@]}" | sed "s#$HOME#~#g; s#f0_s1043_ttr_frac75 #f0_s1043_ttr_frac75_b24 #")
[ "$(echo $reg)" = "$(echo $ab)" ]; ok $? "Q1b argv == argv_frac75.sh fold 0 (the registered probe argv), out renamed"
# tie-break edit: the EDITOR script on a copy, then the Q1b line on top (same awk as qctl top)
cp "$Q" "$T/tb.txt"
bash $KT/box/tiebreak_edit.sh "$T/tb.txt"; ok $? "tiebreak_edit.sh rewrote every active frac75 gate via"
QR_LINE="$Q1B" awk 'BEGIN { done = 0 } { t = $0; sub(/^[ \t\r]+/, "", t)
  if (!done && t != "" && substr(t, 1, 1) != "#") { print ENVIRON["QR_LINE"]; done = 1 } print }' "$T/tb.txt" > "$T/tb2.txt"
out=$(bash "$R" --check "$T/tb2.txt"); rc=$?; echo "$out" | sed 's/^/    /'
[ $rc = 0 ] && echo "$out" | grep -q ', 0 BAD'; ok $? "tie-break queue passes queue_runner.sh --check (0 BAD)"
first=$(echo "$out" | grep -m1 '^L')
echo "$first" | grep -q 'f0_s1043_ttr_frac75_b24 | gc 1 | fold 0 seed 1043 | gate -'; ok $? "tie-break: Q1b is the next job (gc 1, fold 0, seed 1043, no gate)"
[ "$(echo "$out" | grep -c 'via: F75_RUN=/home/training/fleet_runs/f0_s1043_ttr_frac75_b24 bash ~/queue/gates/frac75_gate.sh')" -ge 3 ]
ok $? "tie-break: the Q2 f2-f4 gates now score the b24 box probe"
# Kaggle run failed: the runner had marked Q1 / Q2 f1 #REFUSED; re-add Q2 f1 then Q1 with qctl top
sed -e "s#^Q1 frac75 f0 probe#\#REFUSED 10-08 10:05 (out_dir or its .log exists) Q1 frac75 f0 probe#" \
    -e "s#^Q2 frac75 f1 #\#REFUSED 10-08 10:05 (out_dir or its .log exists) Q2 frac75 f1 #" "$Q" > "$T/f.txt"
for ln in "$Q2F1" "$Q1"; do
  QR_LINE="$ln" awk 'BEGIN { done = 0 } { t = $0; sub(/^[ \t\r]+/, "", t)
    if (!done && t != "" && substr(t, 1, 1) != "#") { print ENVIRON["QR_LINE"]; done = 1 } print }' "$T/f.txt" > "$T/f2.txt"
  mv "$T/f2.txt" "$T/f.txt"
done
out=$(bash "$R" --check "$T/f.txt"); rc=$?
[ $rc = 0 ] && echo "$out" | grep -q ', 0 BAD'; ok $? "restore queue passes --check (0 BAD)"
j=$(echo "$out" | grep '^L' | head -2 | cut -d'|' -f3 | tr -d ' ' | tr '\n' ,)
[ "$j" = "f0_s1043_ttr_frac75,f1_s1043_ttr_frac75," ]; ok $? "restore: Q1 then Q2 f1 are the next two jobs ($j)"
echo "QUEUE-LINES TESTS: $pass/$((pass + fail)) PASS"
[ $fail = 0 ]
