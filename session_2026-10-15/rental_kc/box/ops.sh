#!/bin/bash
# ops.sh (BOX helper for rental_kc; ~/rkc/src/box/ops.sh). Never touches a running trainer, never deletes a run, never edits
# ~/queue (reads queue.txt / history.txt only), never prints a private key. Test overrides: RKC_FR, RKC_Q, RKC_QCTL_FILE.
#   out KEY                       the member's out dir name (= the box queue line's out dir)
#   members OUTFILE               write the member registry (key, out, kind, prio, gate, argv template with ~ -> @R@) from
#                                 the LIVE queue line of each member (uncommented, else its newest commented copy), falling
#                                 back to ~/kc/code/KC_QCTL_1015.txt only when the queue has no line for it yet
#   argv-check REG KEY..          template in REG == the current queue line argv (~ -> @R@), token for token
#   code-check BUNDLE.json        CODE_SAME if every ~/fast_build/code2/*.py md5 == the bundle's code_md5, else CODE_CHANGED:files
#   line-state KEY                ACTIVE (uncommented) | RUN | REFUSED | SKIP | BAD | HOLD | NONE (newest line for that out)
#   keypub | keyinfo | sshtest HOST PORT   the dedicated box key ~/.ssh/vast_rsna_ed25519 (public line only)
#   ph-status|ph-create|ph-release KEY..   EMPTY placeholder dirs ~/fleet_runs/<out>: the queue runner REFUSES a line whose
#                                 out dir exists, so the box never trains a member the rental owns
set -u
SRC=${RKC_SRC:-$HOME/rkc/src}; . $SRC/remote/kit.env
FR=${RKC_FR:-$HOME/fleet_runs}; Q=${RKC_Q:-$HOME/queue/queue.txt}; QF=${RKC_QCTL_FILE:-$HOME/kc/code/KC_QCTL_1015.txt}
K=$HOME/.ssh/vast_rsna_ed25519; KH=$HOME/.ssh/known_hosts_rkc
out(){ case $1 in s2003) echo fA_s2003_kc256_dread;; s2002tb) echo fA_s2002_kc256_r7tb;; s2002d2) echo fA_s2002_kc256_dread2;;
  s2004) echo fA_s2004_kc256_dread3;; s2011) echo f0_s2011_kc256_ev2s_r7tb;; *) echo "BADKEY"; return 1;; esac; }
kind(){ case $1 in s2011) echo F0;; *) echo AD;; esac; }
prio(){ case $1 in s2003) echo 1;; s2002tb|s2002d2) echo 2;; s2004) echo 3;; s2011) echo 4;; esac; }
closing(){ tail -n 1 "$1" 2>/dev/null | tr -d '\r' | grep -Eq '^\{"raw": [^,]+, "ema": [^}]+\}$'; }
trainer_alive(){ pgrep -f -- "fleet_train[.]py .*--out [^ ]*/$1( |\$)" > /dev/null 2>&1; }
qlines(){ grep -E "\| ~/fleet_runs/$1 \|" "$Q" 2>/dev/null; }                       # every line (any state) for that out
qactive(){ qlines "$1" | grep -E '^[[:space:]]*[^#[:space:]]' | tail -n 1; }
argv_of(){ printf '%s' "${1##*|}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
gate_of(){ printf '%s' "$1" | awk -F'|' '{print $3}' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
state_of(){ local l; l=$(qlines "$1" | tail -n 1); [ -n "$l" ] || { echo NONE; return; }
  case "$l" in \#RUN*) echo RUN;; \#REFUSED*) echo REFUSED;; \#SKIP*) echo SKIP;; \#BAD*) echo BAD;; \#HOLD*) echo HOLD;; \#*) echo COMMENTED;; *) echo ACTIVE;; esac; }
cmd=${1:-}; shift || true
case $cmd in
out) out "$1";;
line-state) state_of "$(out "$1")";;
code-check)
  $HOME/miniconda3/envs/rsna/bin/python - "$1" ${RKC_CODE:-$HOME/fast_build/code2} <<'EOF'
import json, sys, glob, os, hashlib
b = json.load(open(sys.argv[1]))['code_md5']; c = {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest() for p in glob.glob(sys.argv[2] + '/*.py')}
d = sorted(k for k in set(b) | set(c) if b.get(k) != c.get(k)); print('CODE_SAME' if not d else 'CODE_CHANGED:' + ','.join(d))
EOF
  ;;
members)
  OF=$1; rm -f "$OF.tmp"; rc=0
  for k in $MEMBER_KEYS; do o=$(out $k)
    l=$(qactive $o); src=queue_active
    [ -n "$l" ] || { l=$(qlines $o | tail -n 1 | sed -E 's/^#[A-Z]+ [0-9-]+ [0-9:]+ (\([^)]*\) )?//'); src=queue_commented; }
    [ -n "$l" ] || { l=$(grep -F "| ~/fleet_runs/$o |" "$QF" 2>/dev/null | grep -o "'[^']*'" | tr -d "'" | tail -n 1); src=kc_qctl_file; }
    [ -n "$l" ] || { echo "member $k: no line for $o in the queue nor in $QF"; rc=1; continue; }
    a=$(argv_of "$l"); g=$(gate_of "$l")
    case " $a " in *" --out ~/fleet_runs/$o "*) ;; *) echo "member $k: argv --out is not ~/fleet_runs/$o"; rc=1; continue;; esac
    case "$a" in --kc\ *) ;; *) echo "member $k: argv does not start with --kc"; rc=1; continue;; esac
    t=$(printf '%s' "$a" | sed 's#~/#@R@/#g')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$k" "$o" "$(kind $k)" "$(prio $k)" "$src" "$g" "$t" >> "$OF.tmp"
    echo "member $k -> $o ($(kind $k), prio $(prio $k), from $src, $(echo $a | wc -w) tokens) gate: $g"
  done
  [ -s "$OF.tmp" ] && mv -f "$OF.tmp" "$OF"; exit $rc;;
argv-check)
  REG=$1; shift; rc=0
  for k in "$@"; do o=$(out $k); t=$(awk -F'\t' -v k=$k '$1==k {print $7}' "$REG")
    l=$(qactive $o); st=ACTIVE; [ -n "$l" ] || { l=$(qlines $o | tail -n 1); st=$(state_of $o); }
    if [ -z "$l" ]; then echo "argv $k: NO_LINE in the queue for $o"; rc=1; continue; fi
    w=$(argv_of "$l" | sed 's#~/#@R@/#g')
    if [ -n "$t" ] && [ "$t" = "$w" ]; then echo "argv $k: IDENTICAL to the box queue line ($st, $(echo $t | wc -w) tokens)"
    else echo "argv $k: DIFFERS from the box queue line ($st)"; diff <(tr ' ' '\n' <<< "$t") <(tr ' ' '\n' <<< "$w") | head -n 6; rc=1; fi
  done; exit $rc;;
keypub)
  [ -f $K ] || ssh-keygen -q -t ed25519 -N '' -C box-vast-rsna -f $K; [ -f $K.pub ] || ssh-keygen -y -f $K > $K.pub; cat $K.pub;;
keyinfo) [ -f $K.pub ] && ssh-keygen -l -f $K.pub || echo "no box key";;
sshtest)
  ssh -i $K -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH \
      -o LogLevel=ERROR -p "$2" root@"$1" 'echo box-to-instance-ok' 2>&1 | grep -v -E 'Welcome|Have fun|^$' | tail -n 2;;
ph-status)
  for k in "$@"; do n=$(out $k); D=$FR/$n
    echo "$k: $D $( [ -d $D ] && { [ -z "$(ls -A $D)" ] && echo '(EMPTY placeholder)' || echo "($(find $D -type f | wc -l) files)"; } || echo '(absent)') | log $( [ -e $D.log ] && { closing $D.log && echo COMPLETE || echo partial; } || echo absent) | queue line $(state_of $n) | box trainer $(trainer_alive $n && echo RUNNING || echo no)"
  done;;
ph-create)
  rc=0; for k in "$@"; do n=$(out $k); D=$FR/$n
    if trainer_alive $n; then echo "REFUSE $k: a box trainer runs $n"; rc=1; continue; fi
    s=$(state_of $n); case $s in RUN) echo "REFUSE $k: the box queue launched $n (#RUN line)"; rc=1; continue;; esac
    if [ -e $D.log ]; then echo "REFUSE $k: $D.log exists"; rc=1; continue; fi
    if [ -d $D ]; then [ -z "$(ls -A $D)" ] && echo "OK $k: empty placeholder already present" || { echo "REFUSE $k: $D exists, not empty"; rc=1; }; continue; fi
    mkdir -p $FR && mkdir $D && echo "OK $k: created EMPTY placeholder $D (queue line $s)"
  done; exit $rc;;
ph-release)
  for k in "$@"; do n=$(out $k); D=$FR/$n
    if trainer_alive $n; then echo "KEEP $k: a box trainer runs $n"; continue; fi
    if [ -d $D ] && [ -z "$(ls -A $D)" ] && [ ! -e $D.log ]; then rmdir $D && echo "$k: removed the empty placeholder $D (queue line now $(state_of $n))"
    else echo "$k: no empty placeholder to release ($( [ -e $D ] && echo 'dir has content' || echo absent))"; fi
  done;;
*) sed -n '2,14p' "$0"; exit 2;;
esac
