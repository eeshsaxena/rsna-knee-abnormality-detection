#!/bin/bash
# install_r7tb_top.sh (r7teacher/labels, 10-14, PREREG_TEACHER A1; BOX). NOT RUN BY ITS AUTHOR: the main session runs it after
# Eesh's "OK R7" AND after on_r7oof.sh printed L0 PASS (or DROP) + LABEL_GATE TRAIN_TABLE=labels_r7t_b.csv + labels_gate PASS.
# Puts the 5 T_b student lines (TTR224 s1047, 21 ep, gc0, checkpoint selection on the student's own table = A1 FX4) at the TOP
# of ~/queue/queue.txt, f0 first, ahead of the L13 s1044 filler lines (which keep their order behind them). Uses only
# ~/queue/qctl.sh (lock + --check + backup + atomic replace). It never starts, stops or signals anything: the runner picks
# f0 when the running job (L13 f2 by then) ends. 5 x 3.52 h = ~17.6 h (MEASURED rate).
# Refuses (nothing changes) unless: the T_b verdict file says PASS (installing earlier would let the line's `via` gate run
# once and write FAIL, which then sticks: queue_runner.sh semantics), the sheet md5 is the A1 one, no s1047 r7tb line or out
# dir exists yet. T_a lines are NOT installed here (see FIXES_A1.md: labels_gate.sh passes only the table the label gate chose).
#   usage (laptop):  scp install_r7tb_top.sh training@192.168.10.30:r7t_labels/ && ssh training@192.168.10.30 'bash ~/r7t_labels/install_r7tb_top.sh'
#   or on the box:   bash ~/r7t_labels/install_r7tb_top.sh [--dry]      (--dry: all checks + the resulting queue preview, no write)
set -u
D=$HOME/r7t_labels; SH=$D/sheets/queue_r7tb.txt; V=$D/real/labels_gate_labels_r7t_b.verdict
SHEET_MD5=6196c8e48ff15687f524a4c180a4bda6                       # sheets/queue_r7tb.txt as frozen in PREREG_TEACHER A1
DRY=${1:-}
die(){ echo "INSTALL REFUSED: $*"; exit 1; }
[ "$(md5sum < $SH | cut -c1-32)" = $SHEET_MD5 ] || die "sheet md5 != the A1 sheet"
[ "$(wc -l < $SH)" = 5 ] || die "sheet must hold 5 lines"
head -c 4 $V 2>/dev/null | grep -q PASS || die "T_b verdict is not PASS ($(cat $V 2>/dev/null || echo missing)); run on_r7oof.sh first"
bash $D/labels_gate.sh labels_r7t_b.csv | grep -q '^PASS' || die "labels_gate.sh does not PASS labels_r7t_b.csv now"
grep -q 's1047_ttr_r7tb' $HOME/queue/queue.txt && die "an s1047 r7tb line is already in the queue"
for F in 0 1 2 3 4; do [ -e $HOME/fleet_runs/f${F}_s1047_ttr_r7tb ] || [ -e $HOME/fleet_runs/f${F}_s1047_ttr_r7tb.log ] && die "out dir f${F}_s1047_ttr_r7tb exists"; done
grep -q -- '--eval_labels ~/r7t_labels/real/labels_r7t_b.csv' $SH || die "sheet lacks the A1 FX4 --eval_labels"
if [ "$DRY" = --dry ]; then
  W=$(mktemp); cp $HOME/queue/queue.txt $W
  for F in 4 3 2 1 0; do L=$(sed -n "$((F + 1))p" $SH)
    QR_LINE="$L" awk 'BEGIN { done = 0 } { t = $0; sub(/^[ \t\r]+/, "", t)
      if (!done && t != "" && substr(t, 1, 1) != "#") { print ENVIRON["QR_LINE"]; done = 1 } print }
      END { if (!done) print ENVIRON["QR_LINE"] }' $W > $W.2 && mv -f $W.2 $W; done
  bash $HOME/queue/queue_runner.sh --check $W && echo "DRY: the queue would be (job lines only):" && grep -v '^[[:space:]]*#' $W | grep -v '^[[:space:]]*$' | cut -c1-110
  rm -f $W; exit 0
fi
for F in 4 3 2 1 0; do bash $HOME/queue/qctl.sh top "$(sed -n "$((F + 1))p" $SH)" || die "qctl top f$F failed (queue unchanged from that line on)"; done
bash $HOME/queue/qctl.sh check
echo "INSTALLED: job lines now:"; grep -v '^[[:space:]]*#' $HOME/queue/queue.txt | grep -v '^[[:space:]]*$' | cut -c1-110
