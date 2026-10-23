#!/bin/bash
# box_ops.sh: BOX-side helper for the vastR7T kit (copied to ~/vastR7T/ by the laptop scripts; run over ssh).
# vastR7T = the R7-teacher TTR224 student, kit.env RUNTAG/SEED, folds 0-4 (names fF_s<SEED>_<RUNTAG>). Its own dir ~/vastR7T.
# Every subcommand prints what it does. It never touches a running trainer, never deletes a run (things are moved
# aside), edits the queue only through ~/queue/qctl.sh, and never prints a private key.
#   paths [F..]                  read only: every box path the Vast run needs + queue/gate state
#   versions                     read only: pinned package versions of the rsna env (pip freeze subset)
#   verdict                      read only: the gate verdict file named in kit.env QGATE (or MISSING)
#   keyinfo | keypub             key fingerprint (read only) | ensure the dedicated key exists, print its PUBLIC line
#   sshtest HOST PORT            box -> instance ssh with the dedicated key
#   argv-check F..               argv_template.txt (~ <-> @R@) == the r7teacher box sheet line (~/r7t_labels/sheets/$SHEET) token for token
#   ph-status|ph-create|ph-release F..   empty placeholder dirs in ~/fleet_runs (see BOX_QUEUE_NOTE.md)
#   push-start HOST PORT R | push-status   detached box -> instance copy (box_push.sh)
#   parity F..                   parity.py on the box, CPU only, nice 19
#   pull NAME TARGET HOST PORT R EPOCHPT   instance -> box copy of one fold's outputs (no epochs/*.pt unless EPOCHPT=1)
#   verify NAME TARGET MODE IID GPU OFFER  md5 manifest on stdin; closing line, ckpt args, receipt json
set -u
PY=~/miniconda3/envs/rsna/bin/python
Q=${VAST320_Q:-$HOME/queue/queue.txt}; QCTL=${VAST320_QCTL:-$HOME/queue/qctl.sh}; FR=${VAST320_FR:-$HOME/fleet_runs}   # overrides: tests only
K=~/.ssh/vast_rsna_ed25519; KHV=~/.ssh/known_hosts_vastR7T   # own file: the L5 kit uses ~/.ssh/known_hosts_vast
TPL=~/vastR7T/argv_template.txt
. ~/vastR7T/kit.env                                # RUNTAG SEED TTR TSL TABLE SHEET QREF_TAG QNAME_TPL QGATE
KREF=$FR/f2_s1043_ttr_frac75                         # set D fold 2 (TTR224 frac75 s1043, 21 ep): recipe reference for the ckpt-args check
CLOSE_RE='^\{"raw": [^,]+, "ema": [^}]+\}$'
name(){ echo "f${1}_s${SEED}_${RUNTAG}"; }
closing(){ tail -n 1 "$1" 2>/dev/null | tr -d '\r' | grep -Eq "$CLOSE_RE"; }
trainer_alive(){ pgrep -f -- "fleet_train[.]py .*--out [^ ]*/$1( |\$)" > /dev/null 2>&1; }
qline_state(){   # NAME -> active | #RUN | #REFUSED | #SKIP | #BAD | none (first matching line wins: active beats comments)
  local n=$1 l
  if grep -Eq "^[[:space:]]*[^#[:space:]].*\| ~/fleet_runs/$n \|" "$Q"; then echo active; return; fi
  l=$(grep -E "^#[A-Z]+ .*\| ~/fleet_runs/$n \|" "$Q" | tail -n 1)
  [ -n "$l" ] && { echo "${l%% *}"; return; } ; echo none
}
vssh(){ echo "ssh -i $K -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KHV -o LogLevel=ERROR -p $1"; }
chk(){ if [ -e "$1" ]; then echo "OK      $2 ${3:-}"; else echo "MISSING $2"; BADN=$((BADN + 1)); fi; }

cmd=${1:-}; shift || true
case $cmd in
paths)
  BADN=0; FOLDS=${*:-0 1 2 3 4}; TL='~'
  echo "== box paths ($(hostname), $(date '+%F %T %Z'))"
  chk "$PY" "rsna env python ~/miniconda3/envs/rsna/bin/python"
  for f in fleet_train.py fleet_common.py fleet_model.py fleet_model_ttr.py fleet_aug.py; do
    chk ~/fast_build/code2/$f "~/fast_build/code2/$f" "md5 $(md5sum < ~/fast_build/code2/$f 2>/dev/null | cut -c1-12)"
  done
  echo "        code2 dir size (no __pycache__): $(du -sb --exclude=__pycache__ ~/fast_build/code2 | cut -f1) bytes"
  for f in ~/r7t_labels/real/$TABLE ~/r7t_labels/sheets/$SHEET ~/rsna_train/folds5.csv ~/rsna_train/gold74_eval_set.csv ~/rsna_train/llm_labels_v4_blend.csv; do
    chk "$f" "${f/#$HOME/$TL}" "md5 $(md5sum < "$f" 2>/dev/null | cut -c1-12) rows $(wc -l < "$f" 2>/dev/null)"
  done
  if [ -d ~/fleet_store ]; then
    echo "OK      ~/fleet_store: $(find ~/fleet_store -maxdepth 1 -name '*.npz' | wc -l) npz, $(du -sb ~/fleet_store | cut -f1) bytes"
  else echo "MISSING ~/fleet_store"; BADN=$((BADN + 1)); fi
  H=~/.cache/huggingface/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k
  B=$H/blobs/77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a
  chk "$H/snapshots/bc48a87f119bae1b06418f5340bfa97ef75d4609/model.safetensors" "HF cache convnext_tiny.fb_in22k_ft_in1k rev bc48a87f (snapshot link)"
  if [ -f "$B" ]; then s=$(sha256sum "$B" | cut -c1-64)
    [ "$s" = 77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a ] && echo "OK      HF blob sha256 77b619e1.. matches ($(stat -c %s "$B") bytes)" || { echo "BAD     HF blob sha256 $s"; BADN=$((BADN + 1)); }
  else echo "MISSING HF blob 77b619e1.."; BADN=$((BADN + 1)); fi
  chk ~/queue/queue.txt "~/queue/queue.txt"; chk ~/queue/qctl.sh "~/queue/qctl.sh"; chk ~/queue/queue_runner.sh "~/queue/queue_runner.sh"
  chk ~/queue/gates "~/queue/gates/"; chk "$TPL" "~/vastR7T/argv_template.txt"; chk "$KREF" "~/fleet_runs/f2_s1043_ttr_frac75 (set D fold 2, recipe reference)"
  echo "        run family: $RUNTAG (ttr $TTR) | gate: $QGATE -> $(bash "$0" verdict | head -c 160)"
  r=$(bash ~/queue/queue_runner.sh --status 2>/dev/null | grep -E '^runner:' | head -n 1); echo "        queue runner: ${r:-unknown}"
  for F in $FOLDS; do n=$(name "$F")
    echo "        f$F: queue line $(qline_state "$n") | dir $( [ -d $FR/$n ] && { [ -z "$(ls -A $FR/$n)" ] && echo empty || echo NONEMPTY; } || echo absent) | log $( [ -e $FR/$n.log ] && echo present || echo absent) | box trainer $(trainer_alive "$n" && echo RUNNING || echo no)"
  done
  echo "        free disk ~: $(df -h ~ | tail -n 1 | awk '{print $4}') | box key: $( [ -f $K ] && echo present || echo 'absent (vast_launch.sh creates it)')"
  echo "== box paths: $BADN problem(s)"; [ $BADN -eq 0 ];;
versions)
  "${PY%python}pip" list --format=freeze 2>/dev/null | grep -i -E '^(opencv-python-headless|opencv-python|numpy|pandas|scikit-learn|scipy|timm|huggingface[-_]hub|safetensors|pyyaml|pillow)==';;
verdict)
  [ "$QGATE" = "-" ] && { echo "PASS (no gate: Eesh asked for these folds directly, 10-09)"; exit 0; }
  vf=$(printf '%s' "$QGATE" | sed -E 's/.* from ([^ ]+).*/\1/'); vf=${vf/#\~/$HOME}
  if [ -f "$vf" ]; then cat "$vf"; else echo MISSING; fi;;
keyinfo)
  if [ -f $K.pub ]; then echo "box key present: $(ssh-keygen -l -f $K.pub)"; else echo "box key absent: $K (created by vast_launch.sh)"; fi;;
keypub)
  if [ ! -f $K ]; then ssh-keygen -q -t ed25519 -N '' -C "box-vast-rsna320" -f $K && echo "created dedicated key $K" >&2; fi
  [ -f $K.pub ] || ssh-keygen -y -f $K > $K.pub
  cat $K.pub;;
sshtest)
  $(vssh "$2") root@"$1" 'echo box-to-instance-ok' 2>&1 | grep -v -E 'Welcome|Have fun|^$' | tail -n 2;;
argv-check)
  rc=0; SH=~/r7t_labels/sheets/$SHEET
  for F in "$@"; do n=$(name "$F")
    l=$(grep -E "\| ~/fleet_runs/$n \|" "$SH" 2>/dev/null | tail -n 1)
    if [ -z "$l" ]; then echo "argv-check f$F: NO sheet line for ~/fleet_runs/$n in $SH"; rc=1; continue; fi
    a=$(printf '%s' "${l##*|}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    want=$(sed -e 's#@R@#~#g' -e "s#@F@#$F#g" "$TPL")
    if [ "$a" = "$want" ]; then echo "argv-check f$F: IDENTICAL to the box sheet line ($SHEET, $(echo "$a" | wc -w) tokens)"
    else echo "argv-check f$F: DIFFERS from the box sheet line"; diff <(tr ' ' '\n' <<< "$a") <(tr ' ' '\n' <<< "$want") | head; rc=1; fi
  done; exit $rc;;
ph-status)
  for F in "$@"; do n=$(name "$F")
    echo "f$F: ~/fleet_runs/$n $( [ -d $FR/$n ] && { [ -z "$(ls -A $FR/$n)" ] && echo '(EMPTY placeholder)' || echo "($(find $FR/$n -type f | wc -l) files)"; } || echo '(absent)') | .log $( [ -e $FR/$n.log ] && { closing $FR/$n.log && echo 'present, COMPLETE' || echo 'present, no closing line'; } || echo absent) | queue line $(qline_state "$n") | box trainer $(trainer_alive "$n" && echo RUNNING || echo no)"
  done;;
ph-create)
  rc=0
  for F in "$@"; do n=$(name "$F"); D=$FR/$n
    if trainer_alive "$n"; then echo "REFUSE f$F: a box trainer is running $n"; rc=1; continue; fi
    if [ "$(qline_state "$n")" = "#RUN" ]; then echo "REFUSE f$F: the box queue already launched $n (#RUN)"; rc=1; continue; fi
    if [ -e "$D.log" ]; then echo "REFUSE f$F: $D.log exists"; rc=1; continue; fi
    if [ -d "$D" ]; then
      if [ -z "$(ls -A "$D")" ]; then echo "OK f$F: empty placeholder already present ($D)"; else echo "REFUSE f$F: $D exists and is not empty"; rc=1; fi
      continue
    fi
    mkdir "$D" && echo "OK f$F: created EMPTY placeholder $D (the queue runner will REFUSE its line)"
  done; exit $rc;;
ph-release)
  rc=0
  for F in $(printf '%s\n' "$@" | sort -rn); do n=$(name "$F"); D=$FR/$n   # descending: qctl top keeps f-order
    if trainer_alive "$n"; then echo "REFUSE f$F: a box trainer is running $n"; rc=1; continue; fi
    if closing "$D.log"; then echo "KEEP f$F: $D.log is a COMPLETE run; not released"; continue; fi
    if [ -d "$D" ] && [ -z "$(ls -A "$D")" ] && [ ! -e "$D.log" ]; then rmdir "$D" && echo "f$F: removed empty placeholder $D"
    elif [ -e "$D" ] || [ -e "$D.log" ]; then
      A=${D}_vastpartial_$(date +%m%d_%H%M); { [ -e "$A" ] || [ -e "$A.log" ]; } && A=${A}_$$
      [ -e "$D" ] && mv -n "$D" "$A"; [ -e "$D.log" ] && mv -n "$D.log" "$A.log"
      echo "f$F: partial Vast content moved aside to $A (+ .log), nothing deleted"
    else echo "f$F: no placeholder present"; fi
    # vastR7T: no queue line is added here; to train the fold on the box install its line from ~/r7t_labels/sheets; the fold is
    # simply not trained here. To train it later: argv = argv_template.txt with @R@ -> ~ and @F@ -> F.
    echo "f$F: not re-queued (no box queue line for $n, by design)"
  done; exit $rc;;
push-start)
  S=~/vastR7T/push.status
  if [ -f ~/vastR7T/push.pid ] && kill -0 "$(cat ~/vastR7T/push.pid)" 2>/dev/null; then echo "push already running (pid $(cat ~/vastR7T/push.pid))"; exit 0; fi
  if grep -q '^DONE' $S 2>/dev/null && [ "$(sed -n 2p $S)" = "$1:$2:$3" ]; then echo "push already DONE for $1:$2"; exit 0; fi
  setsid nohup bash ~/vastR7T/box_push.sh "$1" "$2" "$3" > /dev/null 2>&1 < /dev/null &
  echo $! > ~/vastR7T/push.pid; echo "push started on the box (pid $!), log ~/vastR7T/push.log";;
push-status)
  echo "push status: $(head -n 1 ~/vastR7T/push.status 2>/dev/null || echo NONE) | pid alive: $( [ -f ~/vastR7T/push.pid ] && kill -0 "$(cat ~/vastR7T/push.pid)" 2>/dev/null && echo yes || echo no)"
  tail -n 4 ~/vastR7T/push.log 2>/dev/null;;
parity)
  cd ~/fast_build/code2 && TTR=$TTR CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 nice -n 19 "$PY" ~/vastR7T/parity.py --code ~/fast_build/code2 \
    --store ~/fleet_store --labels ~/r7t_labels/real/$TABLE --folds_csv ~/rsna_train/folds5.csv \
    --exclude ~/rsna_train/gold74_eval_set.csv --folds "$@" 2> /tmp/vastR7T_parity.err || { tail -n 5 /tmp/vastR7T_parity.err; exit 1; };;
pull)
  N=$1 T=$2 H=$3 P=$4 RR=$5 EP=${6:-0}; E=$(vssh "$P"); D=$FR/$T; M=$FR/.$T.pulling
  if trainer_alive "$N"; then echo "REFUSE pull $N: a box trainer is running it"; exit 1; fi
  $E root@"$H" "cat $RR/fleet_runs/$N.log" > $FR/.$T.log.part 2>/dev/null || { echo "pull $N: cannot read the instance log"; rm -f $FR/.$T.log.part; exit 1; }
  if [ -e "$D.log" ] && ! cmp -s "$D.log" $FR/.$T.log.part; then
    echo "REFUSE pull $N: $D.log exists and differs from the instance log (not ours)"; rm -f $FR/.$T.log.part; exit 1
  fi
  if [ -d "$D" ] && [ -n "$(ls -A "$D")" ] && [ ! -e "$D.log" ] && [ ! -e "$M" ]; then
    echo "REFUSE pull $N: $D is not empty and was not filled by an earlier vastR7T pull"; rm -f $FR/.$T.log.part; exit 1
  fi
  mkdir -p "$D"; touch "$M"
  inc=(--include='/*_best.pt' --include='/oof_*.csv' --include='/epochs/' --include='/epochs/oof_*.csv')
  [ "$EP" = 1 ] && inc+=(--include='/epochs/*.pt')
  if $E root@"$H" 'command -v rsync' > /dev/null 2>&1; then
    rsync -a --partial -e "$E" "${inc[@]}" --exclude='*' root@"$H":"$RR/fleet_runs/$N/" "$D/" || { echo "pull $N: rsync failed"; exit 1; }
  else
    ep="-name 'oof_*.csv'"; [ "$EP" = 1 ] && ep="$ep -o -name '*.pt'"
    $E root@"$H" "cd $RR/fleet_runs/$N && { find . -maxdepth 1 -type f \( -name '*_best.pt' -o -name 'oof_*.csv' \); find ./epochs -maxdepth 1 -type f \( $ep \) 2>/dev/null; } | tar -cf - -T -" | tar -C "$D" -xf - || { echo "pull $N: tar copy failed"; exit 1; }
  fi
  mv -f $FR/.$T.log.part "$D.log"; rm -f "$M"
  echo "pulled $N -> ~/fleet_runs/$T (+ .log): $(find "$D" -type f | wc -l) files, $(du -sh "$D" | cut -f1)";;
verify)
  N=$1 T=$2 MODE=$3 IID=$4 GPU=$5 OFFER=$6; D=$FR/$T; F=${N#f}; F=${F%%_*}
  MAN=$(sed -e "s#  $N/#  $T/#" -e "s#  $N[.]log\$#  $T.log#"); fail=""
  cd $FR || exit 1
  bad=$(printf '%s\n' "$MAN" | md5sum -c --quiet 2>&1); [ -n "$bad" ] && fail="$fail md5:[$bad]"
  nman=$(printf '%s\n' "$MAN" | grep -c .)
  extra=$(comm -13 <(printf '%s\n' "$MAN" | awk '{print $2}' | sort) <( { find "$T" -type f; echo "$T.log"; } | sort) | grep -v -E '/epochs/[^/]+[.]pt$' || true)
  [ -n "$extra" ] && fail="$fail extra-files:[$(echo $extra | head -c 300)]"
  nb_raw=$(find "$T" -maxdepth 1 -name "f${F}_s${SEED}_raw_*_best.pt" | wc -l); nb_ema=$(find "$T" -maxdepth 1 -name "f${F}_s${SEED}_ema_*_best.pt" | wc -l)
  noof=$(find "$T" -maxdepth 1 -name "oof_f${F}_s${SEED}_*.csv" | wc -l); nep=$(find "$T/epochs" -maxdepth 1 -name 'oof_*.csv' 2>/dev/null | wc -l)
  cl=$(tail -n 1 "$T.log" | tr -d '\r' | cut -c1-120)
  if [ "$MODE" = canonical ]; then
    closing "$T.log" || fail="$fail no-closing-line"
    [ "$nb_raw" = 1 ] && [ "$nb_ema" = 1 ] || fail="$fail best-ckpts(raw $nb_raw, ema $nb_ema)"
    [ "$noof" = 2 ] || fail="$fail top-oof($noof)"
  fi
  ck=$(cd ~ && TTR=$TTR TSL=$TSL SEED=$SEED TABLE=$TABLE KREF=$KREF CUDA_VISIBLE_DEVICES= nice -n 19 "$PY" - "$D" "$F" <<'PYEOF' 2>&1
import sys, glob, os, torch
d, F = sys.argv[1], int(sys.argv[2]); out = []; bad = []
refp = sorted(glob.glob(os.environ['KREF'] + '/f2_s1043_raw_*_best.pt'))
ref = torch.load(refp[0], map_location='cpu', weights_only=False)['args'] if refp else None
SKIP = {'fold', 'epochs', 'save_epochs', 'grad_ckpt', 'seed', 'train_labels', 'sel_labels'}   # intended differences vs set D f2 (checked explicitly above)
for p in sorted(glob.glob(f'{d}/*_best.pt')):
    a = torch.load(p, map_location='cpu', weights_only=False); g = a['args']
    ok = (g.get('fold') == F and g.get('seed') == int(os.environ['SEED']) and g.get('ttr_size') == int(os.environ['TTR'])
          and g.get('model_sha') == 'f141aeed136dfc01' and g.get('train_labels') == os.environ['TABLE']
          and g.get('sel_labels') == os.environ['TABLE'] and g.get('grad_ckpt') is False
          and g.get('train_slices') == int(os.environ['TSL']) and g.get('epochs') == 21 and g.get('save_epochs') == '6-21')
    if ref is None:
        rmsg = 'no Kaggle ref'; ok = False
    else:
        common = sorted((set(g) & set(ref)) - SKIP); diff = [k for k in common if g[k] != ref[k]]
        only = sorted((set(g) ^ set(ref)) - SKIP)
        rmsg = f"recipe vs set D f2 s1043: {len(common) - len(diff)}/{len(common)} keys identical" + (f" DIFF {diff}" if diff else '') + (f" ONE-SIDED {only}" if only else '')
        ok = ok and not diff and not only
    out.append(f"{os.path.basename(p)}: epoch {a['epoch'] + 1} cv {a['cv']:.4f} seed {g.get('seed')} fold {g.get('fold')} ttr {g.get('ttr_size')} slices {g.get('train_slices')} ep {g.get('epochs')} gc {g.get('grad_ckpt')} labels {g.get('train_labels')} | {rmsg} | args {'OK' if ok else 'MISMATCH'}")
    if not ok: bad.append(os.path.basename(p))
print(' ; '.join(out) or 'no best ckpts'); sys.exit(1 if bad else 0)
PYEOF
); [ $? -eq 0 ] || fail="$fail ckpt-args:[$ck]"
  "$PY" - "$FR/vastR7T_${T}_receipt.json" <<PYEOF
import json, sys, time
json.dump(dict(run="$T", source_run="$N", mode="$MODE", vast_instance="$IID", gpu="$GPU", offer="$OFFER",
               fetched_at=time.strftime('%Y-%m-%d %H:%M:%S %Z'), files=$nman, best_raw=$nb_raw, best_ema=$nb_ema,
               top_oof=$noof, epoch_oof=$nep, closing_line=$(printf '%s' "$cl" | "$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
               ckpts=$(printf '%s' "$ck" | "$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
               verify=$(printf '%s' "${fail:-OK}" | "$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')),
          open(sys.argv[1], 'w'), indent=1)
PYEOF
  echo "verify $T ($MODE): files $nman | best raw $nb_raw ema $nb_ema | top oof $noof | epoch oof $nep | closing: $cl"
  echo "  ckpts: $ck"
  if [ -n "$fail" ]; then echo "VERIFY_FAIL $T:$fail"; exit 1; fi
  echo "VERIFY_OK $T (receipt ~/fleet_runs/vastR7T_${T}_receipt.json)";;
*) sed -n '2,17p' "$0"; exit 2;;
esac
