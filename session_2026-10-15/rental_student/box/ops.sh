#!/bin/bash
# ops.sh (BOX helper for rental_student; ~/rstud/src/box/ops.sh). Never touches a running trainer, never deletes a run,
# never edits ~/queue (reads queue.txt only), never prints a private key. Overrides for tests: RSTUD_FR, RSTUD_Q.
#   argv-check                    argv_template == the frozen box T_b sheet line for every fold (token for token) and,
#                                 vs the T_a sheet the box trains, only --labels/--eval_labels/--seed/--out differ
#   keypub | keyinfo              the dedicated box key ~/.ssh/vast_rsna_ed25519 (public line only)
#   sshtest HOST PORT             box -> instance ssh with the dedicated key (own known_hosts file)
#   ph-status|ph-create|ph-release F..   EMPTY placeholder dirs ~/fleet_runs/f<F>_s1047_ttr_r7tb (queue runner REFUSES
#                                 a line whose out dir exists, so the box can never train a fold the rental owns)
set -u
SRC=$HOME/rstud/src; . $SRC/remote/kit.env
FR=${RSTUD_FR:-$HOME/fleet_runs}; Q=${RSTUD_Q:-$HOME/queue/queue.txt}
K=$HOME/.ssh/vast_rsna_ed25519; KH=$HOME/.ssh/known_hosts_rstud
name(){ echo "f${1}_s${SEED}_${RUNTAG}"; }
closing(){ tail -n 1 "$1" 2>/dev/null | tr -d '\r' | grep -Eq '^\{"raw": [^,]+, "ema": [^}]+\}$'; }
trainer_alive(){ pgrep -f -- "fleet_train[.]py .*--out [^ ]*/$1( |\$)" > /dev/null 2>&1; }
# 10-14 21:45: a line gated on BOX_TB_FALLBACK.verdict is the box FALLBACK for a rental fold, not a box claim: the rental may
# reserve that fold (the runner then REFUSES the line on the placeholder dir; tb_fallback.sh re-queues it if the rental fails)
qline(){ grep -E "^[[:space:]]*[^#[:space:]].*\| ~/fleet_runs/$1 \|" "$Q" 2>/dev/null | grep -vq "BOX_TB_FALLBACK[.]verdict" && echo active || echo none; }
cmd=${1:-}; shift || true
case $cmd in
argv-check)
  SH=$HOME/r7t_labels/sheets/$SHEET; TA=$HOME/r7t_labels/sheets/$SHEET_TA; rc=0
  m=$(md5sum < "$SH" | cut -c1-32); [ "$m" = "$SHEET_MD5" ] && echo "sheet $SHEET md5 $m == PREREG A1 frozen" || { echo "sheet $SHEET md5 $m != PREREG A1 $SHEET_MD5"; rc=1; }
  for F in $FOLDS; do n=$(name $F)
    l=$(grep -E "\| ~/fleet_runs/$n \|" "$SH" | tail -n 1); a=$(printf '%s' "${l##*|}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    w=$(sed -e 's#@R@#~#g' -e "s#@F@#$F#g" $SRC/remote/argv_template.txt | tr -d '\r\n')
    if [ -n "$a" ] && [ "$a" = "$w" ]; then echo "argv f$F: IDENTICAL to the box T_b sheet line ($(echo $a | wc -w) tokens)"
    else echo "argv f$F: DIFFERS from the box T_b sheet line"; diff <(tr ' ' '\n' <<< "$a") <(tr ' ' '\n' <<< "$w") | head; rc=1; fi
    la=$(sed -n "$((F + 1))p" "$TA"); b=$(printf '%s' "${la##*|}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    d=$(paste -d' ' <(tr ' ' '\n' <<< "$a") <(tr ' ' '\n' <<< "$b") | awk 'BEGIN{p=""} {if ($1 != $2) printf "%s ", p} {p=$1}')
    case "$d" in "--seed --labels --eval_labels --out ") echo "  vs T_a sheet (box): differs only at: $d";;
      *) echo "  vs T_a sheet (box): UNEXPECTED differences at: ${d:-none}"; rc=1;; esac
  done; exit $rc;;
keypub)
  [ -f $K ] || ssh-keygen -q -t ed25519 -N '' -C box-vast-rsna -f $K; [ -f $K.pub ] || ssh-keygen -y -f $K > $K.pub; cat $K.pub;;
keyinfo) [ -f $K.pub ] && ssh-keygen -l -f $K.pub || echo "no box key";;
sshtest)
  ssh -i $K -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH \
      -o LogLevel=ERROR -p "$2" root@"$1" 'echo box-to-instance-ok' 2>&1 | grep -v -E 'Welcome|Have fun|^$' | tail -n 2;;
ph-status)
  for F in "$@"; do n=$(name $F); D=$FR/$n
    echo "f$F: $D $( [ -d $D ] && { [ -z "$(ls -A $D)" ] && echo '(EMPTY placeholder)' || echo "($(find $D -type f | wc -l) files)"; } || echo '(absent)') | log $( [ -e $D.log ] && { closing $D.log && echo COMPLETE || echo partial; } || echo absent) | queue line $(qline $n) | box trainer $(trainer_alive $n && echo RUNNING || echo no)"
  done;;
ph-create)
  rc=0; for F in "$@"; do n=$(name $F); D=$FR/$n
    if trainer_alive $n; then echo "REFUSE f$F: a box trainer runs $n"; rc=1; continue; fi
    if [ "$(qline $n)" = active ]; then echo "REFUSE f$F: an ACTIVE box queue line trains $n (the box owns this fold)"; rc=1; continue; fi
    if [ -e $D.log ]; then echo "REFUSE f$F: $D.log exists"; rc=1; continue; fi
    if [ -d $D ]; then [ -z "$(ls -A $D)" ] && echo "OK f$F: empty placeholder already present" || { echo "REFUSE f$F: $D exists, not empty"; rc=1; }; continue; fi
    mkdir -p $FR && mkdir $D && echo "OK f$F: created EMPTY placeholder $D"
  done; exit $rc;;
ph-release)
  for F in "$@"; do n=$(name $F); D=$FR/$n
    if trainer_alive $n; then echo "KEEP f$F: a box trainer runs $n"; continue; fi
    if [ -d $D ] && [ -z "$(ls -A $D)" ] && [ ! -e $D.log ]; then rmdir $D && echo "f$F: removed the empty placeholder (the box may train $n now: its T_b sheet line is ready)"
    else echo "f$F: no empty placeholder to release ($( [ -e $D ] && echo 'dir has content' || echo absent))"; fi
  done;;
*) sed -n '2,12p' "$0"; exit 2;;
esac
