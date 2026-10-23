#!/bin/bash
# vast_launch.sh: RENT the Vast instance for the TTR224 frac75 SEED 1044 folds 1-4 (this is the step that spends money),
# wait for ssh, attach the box's PUBLIC key. The box placeholders are created by vast_setup.sh AFTER setup verifies.
# Run from the laptop (Git Bash), lab network. The concurrent L5 run (../vastL5) is only READ, never touched.
#   bash vast_launch.sh --search [--folds "2 3 4"]           read only: current offers ranked for the plan
#   bash vast_launch.sh --dry-run OFFER_ID [--folds "2 3 4"] read only: every check + the exact create command
#   bash vast_launch.sh OFFER_ID [--folds "2 3 4"]           create, or RESUME the instance already in state.env
# Options: --no-gate (skip the PASS check of the kit.env QGATE verdict; only for tests), --force (launch although the worst-case
# MODEL run does not fit the time cap: the lanes then skip the folds that cannot finish; their placeholders are removed).
# Idempotent: an instance recorded in state.env (not destroyed) is resumed, never rented twice; a create whose answer
# was lost is looked up by its unique label before anything else is tried.
SCRIPT_TAG=launch; source "$(dirname "$0")/config.sh"
MODE=run; OFFER=""; FOLDS_ARG=""; GATE=1; FORCE=0
while [ $# -gt 0 ]; do case $1 in
  --search) MODE=search;; --dry-run) MODE=dry;; --folds) FOLDS_ARG=$2; shift;; --no-gate) GATE=0;; --force) FORCE=1;;
  [0-9]*) OFFER=$1;; *) die "unknown argument $1";; esac; shift; done
FOLDS=${FOLDS_ARG:-$(st_get FOLDS)}; FOLDS=${FOLDS:-1 2 3 4}
NF=$(echo $FOLDS | wc -w)
for F in $FOLDS; do case $F in 1|2|3|4) ;; *) die "folds must be from 1-4 (f0 s1044 exists, from Kaggle): got $F";; esac; done
BUDGET_MAX=$("$PYL" -c "print(round($CAP_USD - $RESERVE_USD, 2))")
QUERY='rentable=true verified=true num_gpus>=1 num_gpus<=4 gpu_ram>=20 cuda_vers>=12.1 disk_space>=60 inet_down>=500 reliability>=0.98'   # GPU models: offers_rank.py SPEED table

if [ $MODE = search ]; then
  f=$LOGDIR/offers_$(date +%m%d_%H%M%S).json
  log "search (read only): $QUERY | storage ${DISK_GB} GB priced in"
  "$VAST_BIN" search offers -n "$QUERY" --storage "$DISK_GB" --limit 400 -o dph_total --raw > "$f" 2>/dev/null || die "search failed"
  "$PYL" "$KIT/offers_rank.py" "$f" --folds "$NF" --budget "$BUDGET_MAX" --v100_fold_h "$V100_FOLD_H" --top 25
  log "raw offers saved to $f. Pick the top 'fits=yes' row, verified datacenter (V DC) first, then V. Then:"
  log "  bash vast_launch.sh --dry-run <offer> --folds \"$FOLDS\"   and, if clean,   bash vast_launch.sh <offer> --folds \"$FOLDS\""
  exit 0
fi

IID=$(st_get INSTANCE_ID)
if [ -n "$IID" ] && [ "$(st_get DESTROYED)" != 1 ]; then
  log "RESUME: state.env already holds instance $IID (label $(st_get LABEL)); not renting another"
  OFFER=$(st_get OFFER_ID); FOLDS=$(st_get FOLDS)
elif [ -n "$IID" ]; then
  [ $MODE = dry ] || { mv "$STATE" "$KIT/state.$(date +%m%d_%H%M%S).env"; log "archived the state of destroyed instance $IID"; }
  IID=""
fi
[ -n "$OFFER" ] || die "usage: vast_launch.sh [--dry-run] OFFER_ID [--folds \"2 3 4\"]   (find offers with --search)"
log "== vast_launch $MODE | offer $OFFER | Vast folds [$FOLDS] | cap \$$CAP_USD, floor \$$FLOOR_USD, reserve \$$RESERVE_USD"

# ---- 1 box: reachable, gate verdict, placeholders possible ----
box_up || die "box $BOX unreachable (laptop must be on the lab network)"
sync_box_ops || die "could not copy the kit helpers to the box"
V=$(bops verdict); log "gate verdict ($RUNTAG): ${V:0:160}"
if [ $GATE = 1 ] && ! printf '%s' "$V" | grep -Eq "^(VERDICT[: ]*)?PASS"; then
  [ $MODE = dry ] && log "DRY: the gate has not PASSed yet; a real launch would stop here" || die "gate verdict is not PASS: do not launch (RUNBOOK step 1)"
fi
bops ph-status $FOLDS | sed 's/^/  box /'
bops argv-check $FOLDS | sed 's/^/  box /'; [ "${PIPESTATUS[0]}" = 0 ] || die "box queue argv differs from argv_template.txt"
bops keyinfo | sed 's/^/  box /'

# ---- 2 Vast account: 2FA session, credit, existing instances ----
if ! C0=$(credit_now); then
  [ $MODE = dry ] && { log "DRY: no 2FA session now (expected; RUNBOOK step 0 opens one right before the launch)"; C0=""; } || die "Vast 2FA session needed: RUNBOOK step 0" 3
fi
if [ -n "$C0" ]; then
  L5H=$(l5_holdback)
  BUDGET=$("$PYL" -c "print(round(min($CAP_USD, $C0 - $FLOOR_USD - $L5H), 2))")
  log "credit \$$C0 -> spend budget \$$BUDGET (= min(cap $CAP_USD, credit - floor $FLOOR_USD - L5 hold-back $L5H))"
  num_ge "$BUDGET" 1.5 || die "budget \$$BUDGET too small for this plan"
  IJ=$(vast_raw show instances) && printf '%s' "$IJ" | jq_py '
for i in (d if isinstance(d, list) else d.get("instances", [])):
    print("  account instance %s label %s status %s dph %.3f" % (i.get("id"), i.get("label"), i.get("actual_status"), i.get("dph_total") or 0))' \
    || log "WARN: could not list instances"
else BUDGET=$CAP_USD; log "DRY: no credit read -> budget \$$BUDGET"; fi

# ---- 3 the offer, re-checked now ----
if [ -z "$IID" ]; then
  of=$LOGDIR/offer_$OFFER.json
  # the offers API does not filter on id: re-run the plan query and pick the id out of the fresh result
  # Vast has a DAILY SEARCH ROW QUOTA (429 "Daily search row quota exhausted", hit 10-09): poll_launch.sh passes its narrow query + small limit
  "$VAST_BIN" search offers -n "${LAUNCH_QUERY:-$QUERY}" --storage "$DISK_GB" --limit "${LAUNCH_LIMIT:-400}" -o dph_total --raw > "$of" 2>/dev/null
  RJ=$("$PYL" "$KIT/offers_rank.py" "$of" --folds "$NF" --budget "$("$PYL" -c "print($BUDGET - $RESERVE_USD)")" --json --id "$OFFER" --v100_fold_h "$V100_FOLD_H")
  [ -n "$RJ" ] && [ "$RJ" != "[]" ] || die "offer $OFFER is gone, not rentable or not a known GPU: re-run --search"
  eval "$(printf '%s' "$RJ" | jq_py '
import shlex
r = d[0]
for k in ("gpu", "n", "dph", "ver", "dc", "geo", "vcpu", "rel", "cuda", "rounds", "fold_h", "h_mid", "h_worst", "c_mid", "c_worst", "fits", "fold_s_worst", "cuda_ok", "ram_ok", "xfer", "ok_cpu"):
    print("O_%s=%s" % (k, shlex.quote(str(r[k]))))')"
  log "offer $OFFER: ${O_n}x $O_gpu | \$$O_dph/h ($DISK_GB GB storage incl.) + data-in ~\$${O_xfer:0:5} | ver $O_ver $O_dc | $O_geo | vCPU $O_vcpu | rel $O_rel | cuda $O_cuda"
  log "MODEL: $O_rounds round(s), fold ~${O_fold_h:0:4} h | run mid ${O_h_mid:0:4} h \$${O_c_mid:0:4} | worst ${O_h_worst:0:4} h \$${O_c_worst:0:4} | fits: $O_fits"
  [ "$O_cuda_ok" = True ] || die "offer CUDA < 12.1"
  [ "$O_ver" = V ] || die "offer is not verified"
  [ "$O_ok_cpu" = True ] || die "offer has < 8 vCPU per GPU"
  [ "$O_ram_ok" = True ] || log "WARN: host RAM < 48 GB"
  GUARD_H=$("$PYL" -c "print(round(min(($BUDGET - $RESERVE_USD - $O_xfer) / $O_dph, $MAX_RUN_H + $GUARD_MARGIN_H), 2))")
  CAP_H=$("$PYL" -c "print(round($GUARD_H - $GUARD_MARGIN_H, 2))")
  log "time caps: watcher fetch+destroy at ${CAP_H} h, on-instance self-destroy at ${GUARD_H} h (spend at that point <= \$$BUDGET - reserve)"
  if ! num_ge "$CAP_H" "$O_h_worst"; then
    [ $FORCE = 1 ] || [ $MODE = dry ] || die "worst-case MODEL run ${O_h_worst:0:4} h > time cap $CAP_H h: pick a faster offer, fewer folds, or --force"
    log "WARN: worst-case run exceeds the cap; lanes will skip folds that cannot finish (they return to the box)"
  fi
  [ "$O_fits" = True ] || log "WARN: the ranker says this offer does not fit the plan (see --search)"
  # L5 protection: worst case, both runs bill until L5_END_BY; the credit must stay above L5's finalise line + margin
  if [ -n "$C0" ] && l5_alive; then
    L5T=$(l5_trig); HL=$("$PYL" -c "import time; print(max(0.0, ($L5_END_BY - time.time()) / 3600))")
    CW=$("$PYL" -c "print(round($C0 - $(l5_get DPH) * $HL - ($O_dph * min($HL, $CAP_H + $GUARD_MARGIN_H) + $O_xfer), 3))")
    log "L5 (instance $(l5_get INSTANCE_ID)) still alive: worst-case credit at $(date -d @$L5_END_BY '+%H:%M') = \$$CW vs L5 finalise line \$$L5T + margin $L5_MARGIN"
    num_ge "$CW" "$("$PYL" -c "print($L5T + $L5_MARGIN)")" || die "this rental could push the credit to L5's finalise line before L5 ends: wait for L5 or pick a cheaper offer"
  fi
fi

# ---- 4 the box key (dedicated; only the public line leaves the box) ----
if [ $MODE = dry ]; then log "DRY: would ensure ~/.ssh/vast_rsna_ed25519 on the box and read its .pub"; BOX_PUB="ssh-ed25519 AAAA...(box pub)"
else BOX_PUB=$(bops keypub | tail -n 1); case $BOX_PUB in ssh-ed25519\ *) ;; *) die "could not read the box public key";; esac
  log "box public key: ${BOX_PUB:0:24}... (the dedicated box key)"; fi

# ---- 5 create (only when no instance is recorded) ----
if [ -z "$IID" ]; then
  LABEL=rsna-s${SEED}-${RUNTAG//_/-}-$(date +%m%d%H%M%S)
  NOW=$(date +%s); GUARD_DL=$("$PYL" -c "print(int($NOW + $GUARD_H * 3600))"); DL=$("$PYL" -c "print(int($NOW + $CAP_H * 3600))")
  B64=$(gzip -9c "$KIT/remote/guard.sh" | base64 -w0)
  ONSTART="mkdir -p $R/status; echo $B64 | base64 -d | gunzip > $R/guard.sh; chmod 700 $R/guard.sh; (setsid nohup bash $R/guard.sh >> $R/guard.out 2>&1 < /dev/null &)"
  log "create: vastai create instance $OFFER --image $IMAGE --disk $DISK_GB --ssh --direct --label $LABEL --env '-e RSNA_GUARD_DEADLINE=$GUARD_DL' --onstart-cmd '<guard bootstrap, ${#ONSTART} chars>' --cancel-unavail"
  if [ $MODE = dry ]; then log "DRY: nothing created. Plan: deadline $(date -d @$DL '+%m-%d %H:%M'), self-destroy $(date -d @$GUARD_DL '+%m-%d %H:%M')"; exit 0; fi
  st_set LABEL "$LABEL"; st_set CREATE_ATTEMPT "$NOW"
  rm -f "$KH"; bx 'rm -f ~/.ssh/known_hosts_vastS1044'   # host keys of an earlier rental (ports get reused); this kit's OWN file (L5 keeps ~/.ssh/known_hosts_vast)
  CJ=$(vast_raw create instance "$OFFER" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --label "$LABEL" \
        --env "-e RSNA_GUARD_DEADLINE=$GUARD_DL" --onstart-cmd "$ONSTART" --cancel-unavail); rc=$?
  IID=$(printf '%s' "$CJ" | jq_py 'print(d.get("new_contract", ""))' 2>/dev/null)
  if [ -z "$IID" ]; then   # lost answer? look the label up before anything else
    sleep 20
    IID=$(vast_raw show instances 2>/dev/null | jq_py "
print(next((str(i['id']) for i in (d if isinstance(d, list) else d.get('instances', [])) if i.get('label') == '$LABEL'), ''))" 2>/dev/null)
    [ -n "$IID" ] || die "create failed (rc $rc) and no instance carries label $LABEL: nothing rented. Re-run --search"
    log "create answer lost but instance $IID carries our label: adopting it"
  fi
  st_set INSTANCE_ID "$IID"; st_set OFFER_ID "$OFFER"; st_set LAUNCH_EPOCH "$NOW"; st_set DPH "$O_dph"
  st_set GPU_NAME "$O_gpu"; st_set NGPU "$O_n"; st_set FOLDS "$FOLDS"; st_set CREDIT0 "$C0"; st_set BUDGET "$BUDGET"
  st_set DEADLINE_EPOCH "$DL"; st_set GUARD_DEADLINE "$GUARD_DL"; st_set XFER "$O_xfer"; st_set CAP_RUN "$CAP_USD"; st_set MODEL_FOLD_S "$("$PYL" -c "print(int(float('$O_fold_h') * 3600))")"; st_set DESTROYED 0   # MODEL_FOLD_S = MODEL mid: first-fold admission; later folds use the measured time
  log "RENTED instance $IID ($LABEL) at \$$O_dph/h. Start the watcher NOW in the background: bash vast_watch.sh"
fi
[ $MODE = dry ] && { log "DRY: instance $IID already exists; nothing to do in a dry run"; exit 0; }

# ---- 6 wait for running + ssh ----
log "waiting for instance $IID to run (max 20 min)"
t0=$(date +%s); badn=0
while :; do
  J=$(vast_raw show instance "$IID") || { [ $? = 3 ] && die "2FA session lost while waiting" 3; }
  S=$(printf '%s' "$J" | jq_py 'print(d.get("actual_status"))' 2>/dev/null)
  [ "$S" = running ] && break
  case $S in exited|offline|unknown) badn=$((badn + 1));; esac
  if [ $badn -ge 4 ] || [ $(( $(date +%s) - t0 )) -gt 1200 ]; then
    log "instance $IID never reached running (status $S): destroying it so it cannot bill"
    bash "$KIT/vast_destroy.sh" --yes --no-wipe --reason "never ran"; die "launch failed; pick another offer (--search)" 5
  fi
  sleep 20
done
eval "$(printf '%s' "$J" | jq_py '
p = (d.get("ports") or {}).get("22/tcp") or []
print("DH=%s; DP=%s; PH=%s; PP=%s" % (d.get("public_ipaddr") or "", (p[0].get("HostPort") if p else "") or "", d.get("ssh_host") or "", d.get("ssh_port") or ""))')"
st_set DIRECT_HOST "$DH"; st_set DIRECT_PORT "$DP"; st_set PROXY_HOST "$PH"; st_set PROXY_PORT "$PP"
log "running after $(( ($(date +%s) - t0) / 60 )) min | direct ${DH}:${DP} | proxy ${PH}:${PP}"
ok=""
for i in $(seq 1 12); do
  for rt in "$DH $DP" "$PH $PP"; do set -- $rt; [ -n "${1:-}" ] && [ -n "${2:-}" ] || continue
    st_set I_HOST "$1"; st_set I_PORT "$2"
    if ix 'echo laptop-ok' 2>/dev/null | grep -q laptop-ok; then ok="$1:$2"; break 2; fi
  done; sleep 15
done
[ -n "$ok" ] || die "laptop cannot ssh to instance $IID (is the laptop key registered with Vast?). Destroy with vast_destroy.sh --yes if this persists"
log "laptop -> instance ssh OK via $ok"

# ---- 7 box key on the instance (vastai attach ssh; public key only) ----
if [ "$(st_get BOX_ROUTE_OK)" != 1 ]; then
  vast_raw attach ssh "$IID" "$BOX_PUB" > /dev/null && log "attached the box public key to instance $IID" || log "WARN: vastai attach ssh failed"
  bok=""
  for i in $(seq 1 12); do
    for rt in "$DH $DP" "$PH $PP"; do set -- $rt; [ -n "${1:-}" ] && [ -n "${2:-}" ] || continue
      if bops sshtest "$1" "$2" | grep -q box-to-instance-ok; then bok="$1 $2"; break 2; fi
    done; sleep 15
  done
  if [ -z "$bok" ]; then
    log "attach did not reach sshd yet: appending the box PUBLIC key to /root/.ssh/authorized_keys over the laptop session"
    printf '%s\n' "$BOX_PUB" | ix 'mkdir -p /root/.ssh && touch /root/.ssh/authorized_keys && k=$(cat) && (grep -qxF "$k" /root/.ssh/authorized_keys || echo "$k" >> /root/.ssh/authorized_keys) && chmod 600 /root/.ssh/authorized_keys && echo appended'
    for rt in "$DH $DP" "$PH $PP"; do set -- $rt; [ -n "${1:-}" ] || continue
      bops sshtest "$1" "$2" | grep -q box-to-instance-ok && { bok="$1 $2"; break; }; done
  fi
  [ -n "$bok" ] || die "box cannot ssh to the instance; setup cannot copy data (vast_destroy.sh --yes to stop billing)"
  set -- $bok; st_set B_HOST "$1"; st_set B_PORT "$2"; st_set BOX_ROUTE_OK 1
  log "box -> instance ssh OK via $1:$2"
fi
G=$(ix "tail -n 3 $R/guard.log 2>/dev/null" | tr '\n' ' '); log "on-instance guard: ${G:-no guard.log yet}"
printf '%s' "$G" | grep -q 'http 200' || log "WARN: the self-destroy backstop could not prove its key (see guard.log); the laptop watcher and the 2FA session MUST stay alive"

# ---- 8 box placeholders: DEFERRED (Eesh 10-09: create them only after setup verifies) ----
# vast_setup.sh creates EMPTY ~/fleet_runs/fF_s1044_ttr_frac75 after S1-S5 pass; vast_train.sh refuses to start without them.
bops ph-status $FOLDS | sed 's/^/  box /'
log "LAUNCH DONE: instance $IID, folds [$FOLDS]. Next: bash vast_setup.sh (creates the placeholders at the end), then bash vast_train.sh (watcher must already be running)"
