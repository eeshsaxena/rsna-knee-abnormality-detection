#!/bin/bash
# poll_launch.sh [UNTIL_EPOCH]: watch the Vast offers and, when an acceptable one appears, LAUNCH it, run setup and start
# training, then exit (vast_watch.sh, running in the background, guards, fetches and destroys). Laptop side.
# Coordinator 10-09 15:20 IST:
#   - offer: verified, RTX 3090 x2, dph_total <= $0.34/h, reliability >= 0.98, NOT CN, >= 8 vCPU/GPU, >= 500 Mbps
#   - hard cap $2.30 for this run (spend of a failed attempt in this run is subtracted from the next attempt's cap)
#   - budget = min(cap, credit - $0.15 - L5 hold-back while L5 lives); L5 protection line kept (launch + watcher)
#   - launch even if only 2 folds fit the cap (the lanes then skip f3/f4); stop polling at 10-10 02:00 IST
# Spends money only through vast_launch.sh. logs/poll.active tells vast_watch.sh to idle (not exit) between attempts.
SCRIPT_TAG=poll; source "$(dirname "$0")/config.sh"
END=${1:-$(date -d '2026-10-10 02:00' +%s)}
MAXDPH=${MAXDPH:-0.34}; RUNCAP=$CAP_USD; SPENT_FILE=$LOGDIR/run_spent.txt
PFOLDS=${PFOLDS:-1 2}   # coordinator 10-09 19:05: Vast trains ONLY f1+f2 (round 1); f3+f4 are assigned to Lightning AI (another agent)
QUERY='rentable=true verified=true num_gpus=2 gpu_ram>=20 cuda_vers>=12.1 disk_space>=60 inet_down>=500 reliability>=0.98'
touch "$LOGDIR/poll.active"; trap 'rm -f "$LOGDIR/poll.active"' EXIT
spent(){ [ -f "$SPENT_FILE" ] && awk '{s+=$1} END {printf "%.3f", s+0}' "$SPENT_FILE" || echo 0; }
attempt_cost(){   # cost of the attempt in state.env: max(credit drop since its launch - L5's share, DPH x rented time)
  local c; c=$(credit_now 2>/dev/null); c=${c:-0}
  "$PYL" -c "print(round(max($(st_get CREDIT0) - $c - $(l5_spend_since "$(st_get LAUNCH_EPOCH)") if $c > 0 else 0, $(st_get DPH) * (max(0, $(date -d "$(st_get DESTROYED_AT)" +%s 2>/dev/null || date +%s) - $(st_get LAUNCH_EPOCH))) / 3600), 3))"; }
BL=$LOGDIR/offer_blacklist.txt; touch "$BL"   # offers that failed to start (e.g. docker pull stuck): never retried
abort_attempt(){  # $1 reason: wipe + destroy the current instance, book its cost against this run's cap, keep polling
  echo "$(st_get OFFER_ID)" >> "$BL"; bash "$KIT/vast_destroy.sh" --yes --reason "$1"
  [ "$(st_get DESTROYED)" = 1 ] || { log "DESTROY FAILED for $(st_get INSTANCE_ID): stop polling (check the console)"; exit 6; }
  c=$(attempt_cost); echo "$c $(st_get INSTANCE_ID) $1" >> "$SPENT_FILE"; log "attempt $(st_get INSTANCE_ID) aborted ($1), booked ~\$$c; run spent so far \$$(spent)"; }
log "poll until $(date -d @$END '+%m-%d %H:%M') | folds [$PFOLDS] | 2x RTX 3090 <= \$$MAXDPH/h, not CN | run cap \$$RUNCAP (already spent this run: \$$(spent)) | floor \$$FLOOR_USD"
C=""; CT=0; n=0
while [ "$(date +%s)" -lt "$END" ]; do
  if [ $(( $(date +%s) - CT )) -ge 300 ] || [ -z "$C" ]; then C=$(credit_now 2>/dev/null) || { log "credit read failed (2FA?)"; touch "$LOGDIR/ALERT_2FA"; C=""; }; CT=$(date +%s); fi
  [ -n "$C" ] || { sleep 60; continue; }
  CAPNOW=$("$PYL" -c "print(round($RUNCAP - $(spent), 3))"); L5D=$(l5_get DPH); L5D=${L5D:-0}; HB=$(l5_holdback); LT=$(l5_trig)
  f=$LOGDIR/offers_poll.json
  "$VAST_BIN" search offers -n "$QUERY" --storage "$DISK_GB" --limit 400 -o dph_total --raw > "$f" 2>/dev/null
  PICK=$("$PYL" "$KIT/offers_rank.py" "$f" --folds 4 --budget 2.27 --json 2>/dev/null | "$PYL" -c "
import json, sys, math, time
rows = json.load(sys.stdin); ok = []; bl = set(open(r'$(cygpath -w "$BL")').read().split())
C, cap, fl, hb = $C, $CAPNOW, $FLOOR_USD, $HB
B = min(cap, C - fl - hb); R, GM, LM, S = $RESERVE_USD, $GUARD_MARGIN_H, $LANE_MARGIN_S / 3600, 0.5
hl = max(0.0, ($L5_END_BY - time.time()) / 3600); alive = $(l5_alive && echo 1 || echo 0)
for r in rows:
    if str(r['id']) in bl or r['gpu'] != 'RTX 3090' or r['n'] != 2 or r['dph'] > $MAXDPH or r['ver'] != 'V' or 'CN' in r['geo'] or not r['ok_cpu'] or not r['cuda_ok']:
        continue
    F = r['fold_h']; lane_dl = (B - R - r['xfer']) / r['dph'] - GM - LM
    folds = 4 if S + F + 1.03 * F <= lane_dl else (2 if S + 1.05 * F <= lane_dl else 0)
    cw = C - $L5D * hl - (r['dph'] * hl + r['xfer']) if alive else 99
    if folds and cw >= $LT + $L5_MARGIN:
        ok.append((-folds, r['dph'] + r['xfer'] / 6, r['id'], r['geo'], r['dph'], round(B, 2), folds, round(lane_dl, 2)))
ok.sort()
print(' | '.join('%s %s \$%.3f/h budget \$%.2f folds-that-fit %d lane-deadline %.2fh' % o[2:] for o in ok[:3]))
")
  n=$((n + 1))
  [ "${DRYPOLL:-0}" = 1 ] && { echo "DRYPOLL pick (no launch): ${PICK:-none} | credit $C cap $CAPNOW holdback $HB L5line $LT"; exit 0; }
  if [ -z "$PICK" ]; then [ $((n % 20)) = 1 ] && echo "[$(ts)] no acceptable offer (credit \$$C, cap now \$$CAPNOW)" >> "$LOGDIR/poll.log"; sleep 45; continue; fi
  log "ACCEPTABLE: $PICK"; ID=${PICK%% *}
  CAP_USD=$CAPNOW bash "$KIT/vast_launch.sh" "$ID" --folds "$PFOLDS" --force; rc=$?
  if [ $rc != 0 ]; then
    if [ -n "$(st_get INSTANCE_ID)" ] && [ "$(st_get DESTROYED)" != 1 ]; then
      for t in 1 2; do log "launch rc $rc with an instance recorded: resume try $t"; sleep 30; CAP_USD=$CAPNOW bash "$KIT/vast_launch.sh"; rc=$?; [ $rc = 0 ] && break; done
      [ $rc = 0 ] || { abort_attempt "launch failed after rental (rc $rc)"; continue; }
    else
      if [ "$(st_get OFFER_ID)" = "$ID" ] && [ "$(st_get DESTROYED)" = 1 ] && [ $(( $(date +%s) - $(st_get LAUNCH_EPOCH) )) -lt 3600 ]; then
        c=$(attempt_cost); echo "$c $(st_get INSTANCE_ID) launch rc $rc (rented, destroyed by vast_launch.sh)" >> "$SPENT_FILE"; echo "$ID" >> "$BL"
        log "launch of $ID rented $(st_get INSTANCE_ID) but failed (rc $rc); booked ~\$$c, offer blacklisted; run spent \$$(spent)"
      else log "launch of $ID failed before any rental (rc $rc): keep polling"; fi
      sleep 45; continue; fi
  fi
  log "LAUNCHED $(st_get INSTANCE_ID) (offer $ID). Setup..."
  ok=0; for t in 1 2; do bash "$KIT/vast_setup.sh" && { ok=1; break; }; log "setup try $t failed"; sleep 30; done
  [ $ok = 1 ] || { abort_attempt "setup failed twice"; continue; }
  bash "$KIT/vast_train.sh" || { abort_attempt "vast_train.sh failed"; continue; }
  log "TRAINING STARTED on $(st_get INSTANCE_ID): poll_launch done; vast_watch.sh guards, fetches and destroys"
  exit 0
done
log "poll_launch: no acceptable offer before $(date -d @$END '+%m-%d %H:%M'); polling stopped"; exit 4
