#!/bin/bash
# poll_launch.sh [UNTIL_EPOCH]: watch the Vast offers and, when an acceptable one appears, LAUNCH it, run setup and start
# training, then exit (vast_watch.sh, running in the background, guards, fetches and destroys). Laptop side.
# Coordinator 10-09 23:40 IST (revised plan, after the 2FA renewal):
#   - offer: ONE verified GPU (RTX 3090 / 3090 Ti / 4090 / A5000 / A6000 / A40), dph_total <= $0.25/h incl. storage,
#     reliability >= 0.98, NOT CN, >= 8 vCPU, >= 500 Mbps
#   - folds run SEQUENTIALLY on it: f1 -> f2 -> f3 (-> f4 only if it clearly fits); the lane starts the next fold only if
#     the MEASURED fold time fits the time cap derived from the budget (+3 %, +10 % for f4)
#   - run cap $2.10 (failed attempts in this run are subtracted); budget = min(cap, credit - $0.15)
#   - stop polling at 10-10 12:00 IST (2-GPU version kept as poll_launch.2gpu.sh)
# Spends money only through vast_launch.sh. logs/poll.active tells vast_watch.sh to idle (not exit) between attempts.
SCRIPT_TAG=poll; source "$(dirname "$0")/config.sh"
END=${1:-$(date -d '2026-10-10 12:00' +%s)}
MAXDPH=${MAXDPH:-0.25}; RUNCAP=$CAP_USD; SPENT_FILE=$LOGDIR/run_spent_1gpu.txt
PFOLDS=${PFOLDS:-1 2 3 4}
# narrow query + small limit: Vast's DAILY SEARCH ROW QUOTA was exhausted on 10-09 by wide searches (429)
QUERY="rentable=true verified=true num_gpus=1 gpu_ram>=20 cuda_vers>=12.1 disk_space>=60 inet_down>=500 reliability>=0.98 cpu_cores_effective>=8 dph_total<=$MAXDPH gpu_name in [RTX_3090,RTX_3090_Ti,RTX_4090,RTX_A5000,RTX_A6000,A40]"
LIMIT=${LIMIT:-25}; EVERY=${EVERY:-120}; QLOG=0
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
log "poll until $(date -d @$END '+%m-%d %H:%M') | folds [$PFOLDS] sequential | 1x 3090/4090/A5000-class <= \$$MAXDPH/h, not CN | run cap \$$RUNCAP (already spent this run: \$$(spent)) | floor \$$FLOOR_USD"
C=""; CT=0; n=0
while [ "$(date +%s)" -lt "$END" ]; do
  if [ $(( $(date +%s) - CT )) -ge 300 ] || [ -z "$C" ]; then C=$(credit_now 2>/dev/null) || { log "credit read failed (2FA?)"; touch "$LOGDIR/ALERT_2FA"; C=""; }; CT=$(date +%s); fi
  [ -n "$C" ] || { sleep 60; continue; }
  CAPNOW=$("$PYL" -c "print(round($RUNCAP - $(spent), 3))"); L5D=$(l5_get DPH); L5D=${L5D:-0}; HB=$(l5_holdback); LT=$(l5_trig)
  f=$LOGDIR/offers_poll.json
  "$VAST_BIN" search offers -n "$QUERY" --storage "$DISK_GB" --limit "$LIMIT" -o dph_total --raw > "$f" 2>&1
  if grep -q '"error"' "$f" || ! grep -q '^\[' "$f"; then
    [ $(( $(date +%s) - QLOG )) -ge 3600 ] && { log "search refused: $(head -c 160 "$f" | tr '\n' ' ') -> retry every 15 min"; QLOG=$(date +%s); }
    sleep 900; continue
  fi
  PICK=$("$PYL" "$KIT/offers_rank.py" "$f" --folds 4 --budget 2.27 --json 2>/dev/null | "$PYL" -c "
import json, sys, math, time
rows = json.load(sys.stdin); ok = []; bl = set(open(r'$(cygpath -w "$BL")').read().split())
C, cap, fl, hb = $C, $CAPNOW, $FLOOR_USD, $HB
B = min(cap, C - fl - hb); R, GM, LM, S = $RESERVE_USD, $GUARD_MARGIN_H, $LANE_MARGIN_S / 3600, 0.5
hl = max(0.0, ($L5_END_BY - time.time()) / 3600); alive = $(l5_alive && echo 1 || echo 0)
for r in rows:
    if (str(r['id']) in bl or r['gpu'] not in ('RTX 3090', 'RTX 3090 Ti', 'RTX 4090', 'RTX A5000', 'RTX A6000', 'A40') or r['n'] != 1
            or r['dph'] > $MAXDPH or r['ver'] != 'V' or 'CN' in r['geo'] or not r['ok_cpu'] or not r['cuda_ok']):
        continue
    F = r['fold_h']; lane_dl = (B - R - r['xfer']) / r['dph'] - GM - LM
    t, folds = S, 0
    for k, fac in ((1, 1.05), (2, 1.03), (3, 1.03), (4, 1.10)):   # sequential admission, as in lane.sh
        if t + fac * F > lane_dl: break
        t += F; folds = k
    cw = C - $L5D * hl - (r['dph'] * hl + r['xfer']) if alive else 99
    if folds and cw >= $LT + $L5_MARGIN:
        ok.append((-folds, r['dph'] * F, r['id'], r['gpu'].replace(' ', '_'), r['geo'], r['dph'], round(B, 2), folds, round(F, 2), round(lane_dl, 2)))
ok.sort()
print(' | '.join('%s %s %s \$%.3f/h budget \$%.2f folds-that-fit %d fold-h-MODEL %.2f lane-deadline %.2fh' % o[2:] for o in ok[:3]))
")
  n=$((n + 1))
  [ "${DRYPOLL:-0}" = 1 ] && { echo "DRYPOLL pick (no launch): ${PICK:-none} | credit $C cap $CAPNOW holdback $HB L5line $LT"; exit 0; }
  if [ -z "$PICK" ]; then [ $((n % 10)) = 1 ] && echo "[$(ts)] no acceptable offer (search ok, $(grep -o '"id"' "$f" | wc -l) rows; credit \$$C, cap now \$$CAPNOW)" >> "$LOGDIR/poll.log"; sleep "$EVERY"; continue; fi
  log "ACCEPTABLE: $PICK"; ID=${PICK%% *}
  CAP_USD=$CAPNOW LAUNCH_QUERY="$QUERY" LAUNCH_LIMIT=$LIMIT bash "$KIT/vast_launch.sh" "$ID" --folds "$PFOLDS" --force; rc=$?
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
