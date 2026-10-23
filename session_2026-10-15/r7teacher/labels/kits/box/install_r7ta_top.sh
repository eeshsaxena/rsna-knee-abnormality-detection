#!/bin/bash
# install_r7ta_top.sh (r7teacher/labels, 10-14 16:10, PREREG_TEACHER Amendment A2; BOX). Plan change 15:56 (main): the T_b
# student (seed 1047) trains on a rented 4 x RTX 4090; the BOX trains the T_a student (seed 1048) ahead of the L13 filler, and
# keeps the 5 T_b lines as COMMENTS (#HOLD) as the fallback if the rental fails (un-hold = `bash ~/queue/qctl.sh edit` and drop
# the "#HOLD(...) " prefix of those lines; they then run in queue order).
# Resulting job order: [#HOLD T_b f0..f4 (comments)] [T_a f0, f1, f2, f3, f4] [L13 lines]. Only ~/queue/qctl.sh `top` is used
# (each = an edited copy of the live queue under ~/queue/queue.lock, validated by queue_runner.sh --check, 0 BAD, backup, atomic
# replace); the running job is never touched: the runner starts T_a f0 at its next job boundary.
# Refuses (nothing changes) unless: labels_gate.sh PASSes labels_r7t_a.csv NOW (REAL receipt, PREREG sha, L0 PASS/DROP, T_a passed
# L1 on its own, md5 unchanged), the sheets are the A1 ones, no s1048 r7ta / s1047 r7tb line or out dir exists.
#   bash ~/r7t_labels/install_r7ta_top.sh [--dry]    (--dry: every check + the resulting queue on a temp copy; no write)
set -u
D=$HOME/r7t_labels; SA=$D/sheets/queue_r7ta.txt; SB=$D/sheets/queue_r7tb.txt; Q=$HOME/queue/queue.txt
MD5_A=7b95041e616597b5d9b4c73f6b97561a; MD5_B=6196c8e48ff15687f524a4c180a4bda6      # A1 sheets
DRY=${1:-}
die(){ echo "INSTALL REFUSED: $*"; exit 1; }
[ "$(md5sum < $SA | cut -c1-32)" = $MD5_A ] || die "T_a sheet md5 != A1"
[ "$(md5sum < $SB | cut -c1-32)" = $MD5_B ] || die "T_b sheet md5 != A1"
bash $D/labels_gate.sh labels_r7t_a.csv | grep -q '^PASS' || die "labels_gate.sh does not PASS labels_r7t_a.csv ($(cat $D/real/labels_gate_labels_r7t_a.verdict 2>/dev/null))"
grep -q -- '--eval_labels ~/r7t_labels/real/labels_r7t_a.csv' $SA || die "T_a sheet lacks the A1 FX4 --eval_labels"
grep -Eq 's1048_ttr_r7ta|s1047_ttr_r7tb' $Q && die "an r7ta / r7tb line is already in the queue"
for F in 0 1 2 3 4; do for n in f${F}_s1048_ttr_r7ta f${F}_s1047_ttr_r7tb; do
  { [ -e $HOME/fleet_runs/$n ] || [ -e $HOME/fleet_runs/$n.log ]; } && die "out dir $n exists"; done; done
HOLD(){ printf '#HOLD(rental fallback, PREREG_TEACHER A2; un-hold only if the T_b rental fails) %s' "$(sed -n "$(( $1 + 1 ))p" $SB)"; }
ins(){   # $1 file, $2 line: insert before the first job line (qctl.sh top's own awk)
  QR_LINE="$2" awk 'BEGIN { done = 0 } { t = $0; sub(/^[ \t\r]+/, "", t)
    if (!done && t != "" && substr(t, 1, 1) != "#") { print ENVIRON["QR_LINE"]; done = 1 } print }
    END { if (!done) print ENVIRON["QR_LINE"] }' "$1" > "$1.2" && mv -f "$1.2" "$1"; }
if [ "$DRY" = --dry ]; then
  W=$(mktemp); cp $Q $W
  for F in 0 1 2 3 4; do ins $W "$(HOLD $F)"; done
  for F in 4 3 2 1 0; do ins $W "$(sed -n "$((F + 1))p" $SA)"; done
  bash $HOME/queue/queue_runner.sh --check $W | tail -n 12; echo "DRY: resulting order (non-blank lines from the first #HOLD):"
  awk '/^#HOLD/ { s = 1 } s && NF' $W | cut -c1-120; rm -f $W; exit 0
fi
for F in 0 1 2 3 4; do bash $HOME/queue/qctl.sh top "$(HOLD $F)" || die "qctl top #HOLD f$F failed"; done
for F in 4 3 2 1 0; do bash $HOME/queue/qctl.sh top "$(sed -n "$((F + 1))p" $SA)" || die "qctl top T_a f$F failed"; done
bash $HOME/queue/qctl.sh check | tail -n 12
echo "INSTALLED (T_a ahead of L13; T_b #HOLD). From the first #HOLD:"; awk '/^#HOLD/ { s = 1 } s && NF' $Q | cut -c1-120
