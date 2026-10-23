#!/bin/bash
# vast_fetch.sh: copy fold outputs from the instance back to the BOX, verify, record a receipt. Laptop side.
#   bash vast_fetch.sh [FOLDS..]       COMPLETE folds -> ~/fleet_runs/fF_s1044_ttr_frac75/ + .log (the placeholders)
#   bash vast_fetch.sh --auto          every planned fold: complete -> canonical; incomplete with output -> partial
#   bash vast_fetch.sh --partial F..   incomplete folds -> ~/fleet_runs/fF_s1044_ttr_frac75_vastpartial_<time>/
#   bash vast_fetch.sh --to-laptop ..  fallback when the box is unreachable: copy to ./fetched/ (then --push-pending)
#   bash vast_fetch.sh --push-pending  laptop copies -> box, verify, delete the laptop copies
#   --dry-run                          show what would be copied, copy nothing
# Copied: top-level *_best.pt (raw + ema), oof_*.csv, epochs/oof_*.csv and the trainer log. NOT epochs/*.pt
# (set FETCH_EPOCH_PT=1 to include them). The box pulls with its dedicated key (box_ops.sh pull); verification on the
# box: md5 of every file vs the instance, closing {"raw": .., "ema": ..} line, ckpt args (fold, seed 1044, ttr 224,
# 16 slices, 21 ep, model_sha f141aeed136dfc01, frac75 labels, grad_ckpt False, and every other key == the Kaggle
# fold-0 s1044 ckpt), receipt ~/fleet_runs/vastS1044_<run>_receipt.json.
# Idempotent: a verified fold (FETCHED_fF=ok) is skipped; re-pulls only send what changed.
SCRIPT_TAG=fetch; source "$(dirname "$0")/config.sh"
MODE=canonical; DRY=0; AUTO=0; LAPTOP=0; PUSH=0; ARGS=()
for a in "$@"; do case $a in --dry-run) DRY=1;; --auto) AUTO=1;; --partial) MODE=partial;; --to-laptop) LAPTOP=1;;
  --push-pending) PUSH=1;; [1-4]) ARGS+=("$a");; *) die "unknown argument $a";; esac; done
SF=$(st_get FOLDS); FOLDS=${ARGS[*]:-${SF:-1 2 3 4}}
IID=$(st_get INSTANCE_ID); EP=${FETCH_EPOCH_PT:-0}; FD=$KIT/fetched
log "== vast_fetch | instance ${IID:-none} | folds [$FOLDS] | mode $([ $AUTO = 1 ] && echo auto || echo $MODE)$([ $LAPTOP = 1 ] && echo ' to-laptop')$([ $PUSH = 1 ] && echo ' push-pending') | epochs/*.pt $([ "$EP" = 1 ] && echo INCLUDED || echo excluded)"

if [ $PUSH = 1 ]; then      # ---- laptop copies -> box ----
  box_up || die "box unreachable"; sync_box_ops || die "box helpers"
  for F in $FOLDS; do
    T=$(st_get PENDING_PUSH_f$F); [ -n "$T" ] || continue; N=$(run_name $F)
    [ -d "$FD/$T" ] || die "laptop copy $FD/$T missing"
    log "push f$F: $FD/$T -> box ~/fleet_runs/$T"
    [ $DRY = 1 ] && continue
    if [ "$T" = "$N" ]; then bops ph-status $F | grep -q 'EMPTY placeholder' || die "box ~/fleet_runs/$N is not an empty placeholder: not overwriting"; fi
    (cd "$FD" && tar -cf - "$T" "$T.log") | bx_in "mkdir -p ~/fleet_runs && tar -C ~/fleet_runs -xf -" || die "push f$F failed"
    md=$(cd "$FD" && { find "$T" -type f; echo "$T.log"; } | sort | xargs md5sum)
    m=$([ "$T" = "$N" ] && echo canonical || echo partial)
    printf '%s\n' "$md" | bx_in "bash ~/$BOXDIR/box_ops.sh" verify "$N" "$T" "$m" "$IID" "$(st_get GPU_NAME | tr ' ' _)" "$(st_get OFFER_ID)" | sed 's/^/  box /'
    if [ "${PIPESTATUS[1]}" = 0 ]; then st_set PENDING_PUSH_f$F ""; [ $m = canonical ] && st_set FETCHED_f$F ok; rm -rf "$FD/$T" "$FD/$T.log"; log "f$F pushed + verified; laptop copy deleted"
    else die "verify failed after push f$F (laptop copy kept in $FD)"; fi
  done; exit 0
fi

[ -n "$IID" ] && [ "$(st_get DESTROYED)" != 1 ] || die "no live instance in state.env"
ix true || die "laptop cannot reach the instance"
BH=$(st_get B_HOST); BP=$(st_get B_PORT)
if [ $LAPTOP = 0 ]; then box_up && sync_box_ops || { log "box unreachable: falling back to --to-laptop"; LAPTOP=1; }; fi
okall=1
for F in $FOLDS; do
  N=$(run_name $F)
  if [ "$(st_get FETCHED_f$F)" = ok ]; then log "f$F already fetched + verified (FETCHED_f$F=ok)"; continue; fi
  ST=$(ix "bash $R/kit/status.sh" | sed -n "s/^FOLD f$F state=\([^ ]*\).*/\1/p")
  case $ST in done*) m=canonical;; *) m=partial;; esac
  if [ $AUTO = 0 ] && [ $MODE = canonical ] && [ $m = partial ]; then log "f$F is '$ST', not complete: skipped (use --partial or --auto)"; okall=0; continue; fi
  [ $MODE = partial ] && m=partial
  if ! ix "test -e $R/fleet_runs/$N.log"; then log "f$F: nothing on the instance (state '$ST')"; [ $m = canonical ] && okall=0; continue; fi
  if [ $m = canonical ]; then T=$N; else T=${N}_vastpartial_$(date +%m%d_%H%M); fi
  MAN=$(ix "EPOCHPT=$EP bash $R/kit/status.sh manifest $F") || { log "f$F: manifest failed"; okall=0; continue; }
  log "f$F ($ST) -> $([ $LAPTOP = 1 ] && echo "laptop $FD/$T" || echo "box ~/fleet_runs/$T"): $(printf '%s\n' "$MAN" | grep -c .) files"
  if [ $DRY = 1 ]; then printf '%s\n' "$MAN" | awk '{print "    " $2}' | head -n 8; continue; fi
  if [ $LAPTOP = 1 ]; then
    mkdir -p "$FD/$T"
    ix "cd $R/fleet_runs/$N && { find . -maxdepth 1 -type f \( -name '*_best.pt' -o -name 'oof_*.csv' \); find ./epochs -maxdepth 1 -type f -name 'oof_*.csv' 2>/dev/null; [ $EP = 1 ] && find ./epochs -maxdepth 1 -type f -name '*.pt'; } | tar -cf - -T -" | tar -C "$FD/$T" -xf - || { log "f$F laptop copy failed"; okall=0; continue; }
    ix "cat $R/fleet_runs/$N.log" > "$FD/$T.log"
    bad=$(printf '%s\n' "$MAN" | sed -e "s#  $N/#  $T/#" -e "s#  $N[.]log\$#  $T.log#" | (cd "$FD" && md5sum -c --quiet 2>&1))
    [ -z "$bad" ] || { log "f$F laptop copy md5 mismatch: $bad"; okall=0; continue; }
    st_set PENDING_PUSH_f$F "$T"; ix "touch $R/status/f$F.fetched_laptop"
    log "f$F copied to the laptop + md5 verified; push later: bash vast_fetch.sh --push-pending"; continue
  fi
  bops pull "$N" "$T" "$BH" "$BP" "$R" "$EP" | sed 's/^/  box /'
  [ "${PIPESTATUS[0]}" = 0 ] || { log "f$F pull failed"; okall=0; continue; }
  printf '%s\n' "$MAN" | bx_in "bash ~/$BOXDIR/box_ops.sh" verify "$N" "$T" "$m" "$IID" "$(st_get GPU_NAME | tr ' ' _)" "$(st_get OFFER_ID)" | sed 's/^/  box /'
  if [ "${PIPESTATUS[1]}" = 0 ]; then
    if [ $m = canonical ]; then st_set FETCHED_f$F ok; ix "touch $R/status/f$F.fetched_ok"; log "f$F FETCHED + VERIFIED into ~/fleet_runs/$T"
    else st_set PARTIAL_f$F "$T"; log "f$F partial output kept on the box as ~/fleet_runs/$T (the fold goes back to the box queue)"; okall=0; fi
  else log "f$F VERIFY FAILED (box copy left in place for inspection)"; okall=0; fi
done
if [ $DRY = 0 ]; then
  all=1; for F in $(st_get FOLDS); do [ "$(st_get FETCHED_f$F)" = ok ] || [ -n "$(st_get PENDING_PUSH_f$F)" ] || all=0; done
  [ $all = 1 ] && ix "date > $R/status/FINAL_FETCH_OK" && log "all planned folds secured: FINAL_FETCH_OK written on the instance (guard self-destroys 15 min later if the laptop destroy does not land)"
fi
[ $okall = 1 ] && log "fetch OK" || { log "fetch finished with incomplete folds (see above)"; exit 1; }
