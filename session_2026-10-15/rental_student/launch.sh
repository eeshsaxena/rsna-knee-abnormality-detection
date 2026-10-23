#!/bin/bash
# launch.sh (laptop, Git Bash, lab network): rent the GPU for the T_b student and hand the run to the BOX. Spends money only
# in run mode, and only after every check below passes.
#   bash launch.sh --search            read only: 2 narrow searches (limit 25 each: 4 x RTX 4090 | 2 x A100), ranked
#   bash launch.sh --dry-run OFFER     read only: preflight + fresh offer check + plan + spend projection; creates nothing
#   bash launch.sh OFFER               preflight -> projection guard -> create (on-instance guard armed via onstart) -> wait
#                                      running -> laptop + box ssh -> box key on the instance -> run.env + box driver.sh
#                                      (push, setup, parity, placeholders, 10-min smoke, full run, fetch loop) -> laptop
#                                      backup watcher. Resumable via state.env (never rents twice).
# Preflight: box reachable; kit synced to ~/rstud/src; argv == the frozen box T_b sheet (and only labels/seed/out differ
# from the T_a sheet); bundle REAL + BUNDLE_OK + its table sha256 == the CURRENT ~/r7t_labels/final/LABELS_READY + kit
# files in the bundle == the synced kit; 2FA session; NO instance on the account; the old vastR7T kit has no live instance.
# Money: ledger CREDIT_START (set once); spent = credit drop since; budget = min(CAP - spent - reserve, credit - floor);
# abort unless the offer's WORST-case MODEL cost fits the budget; guard hard deadline = min(budget / $/h, 7 h).
SCRIPT_TAG=launch; source "$(dirname "$0")/config.sh"
MODE=run; OFFER=""
while [ $# -gt 0 ]; do case $1 in --search) MODE=search;; --dry-run) MODE=dry;; [0-9]*) OFFER=$1;; *) die "bad arg $1";; esac; shift; done
BASEQ="rentable=true verified=true reliability>=0.98 cuda_vers>=12.1 disk_space>=$DISK_GB inet_down>=300"
# launch 10-14 18:35: Vast search shows ONE machine per host per call (host 150178 has machines 26415 / 26418 / 26529, all
# 4 x 4090 $1.831/h), so the fresh-offer re-check of OFFER saw another machine and refused 3 x. PIN_MACHINE=<machine_id> only
# NARROWS both queries to that machine (every filter and check below is unchanged).
# coordinator 10-14 19:10 ("A100 OK"): PER_GPU_MAX / MAX_DPH / MAX_RUN_H may be set per launch (defaults = the reviewed values)
PER_GPU_MAX=${PER_GPU_MAX:-0.55}; MAX_DPH=${MAX_DPH:-99}
RSTUD_FOLDS=${RSTUD_FOLDS:-0 1 2 3 4}; for F in $RSTUD_FOLDS; do case $F in 0|1|2|3|4) ;; *) die "RSTUD_FOLDS: folds must be 0-4";; esac; done
PINQ=""; [ -n "${PIN_MACHINE:-}" ] && { case $PIN_MACHINE in *[!0-9]*) die "PIN_MACHINE must be digits";; esac; PINQ=" machine_id=$PIN_MACHINE"; }
Q4="$BASEQ num_gpus=4 gpu_name=RTX_4090 gpu_ram>=23 cpu_cores_effective>=32$PINQ"
QA="$BASEQ num_gpus=2 gpu_name in [A100_SXM4,A100_PCIE] cpu_cores_effective>=16$PINQ"
search(){ local f1=$LOGDIR/offers_4090x4_$1.json f2=$LOGDIR/offers_a100x2_$1.json
  "$VAST_BIN" search offers -n "$Q4" --storage "$DISK_GB" --limit 25 -o dph_total --raw > "$f1" 2>"$f1.err" || die "search 1 failed: $(head -c 200 "$f1.err") (2FA?)"
  "$VAST_BIN" search offers -n "$QA" --storage "$DISK_GB" --limit 25 -o dph_total --raw > "$f2" 2>"$f2.err" || die "search 2 failed: $(head -c 200 "$f2.err")"
  grep -q -i "two factor" "$f1" "$f2" && die "searches need a 2FA session" 3
  echo "$f1 $f2"; }
cap_left(){ local s; s=$(spent_total) || return $?; "$PYL" -c "print(round($CAP_USD - $s - $RESERVE_USD, 2))"; }
if [ $MODE = search ]; then
  CL=$(cap_left) || die "2FA session needed for the credit read" 3
  F=$(search $(date +%m%d_%H%M%S)); log "search (read only, 2 x limit 25) | cap left \$$CL | $Q4 || $QA"
  "$PYL" "$KIT/offers_rank.py" $F --cap_left "$CL" --max_h "$MAX_RUN_H" --per_gpu_max "$PER_GPU_MAX" --nfolds "$(echo $RSTUD_FOLDS | wc -w)"
  log "then: bash launch.sh --dry-run <id>   and   bash launch.sh <id>"; exit 0
fi
[ -n "$OFFER" ] || die "usage: launch.sh --search | --dry-run OFFER | OFFER"
# coordinator 10-14 ~19:58: one owner only. OWNER.lock names it; a launch (dry or run) needs RSTUD_OWNER == that owner
LOCKO=$(kv_get "$KIT/OWNER.lock" owner); [ -n "$LOCKO" ] || die "no owner in $KIT/OWNER.lock: refusing"
[ "${RSTUD_OWNER:-}" = "$LOCKO" ] || die "OWNER.lock names $LOCKO, not '${RSTUD_OWNER:-unset}': refusing to rent"
log "== launch $MODE | offer $OFFER | cap \$$CAP_USD total (floor \$$FLOOR_USD, reserve \$$RESERVE_USD)"

# ---- 0 preflight (box, kit, bundle, argv) ----
bx true || die "box unreachable (lab network needed: the box drives the run and pulls the folds)"
bx "mkdir -p ~/rstud/src/box ~/rstud/src/remote ~/rstud/src/tests ~/rstud/logs" || die "box mkdir"
for f in box/*.sh box/*.py remote/* tests/*; do [ -f "$KIT/$f" ] || continue
  bx_in "cat > ~/rstud/src/$f.tmp && mv -f ~/rstud/src/$f.tmp ~/rstud/src/$f" < "$KIT/$f" || die "sync $f to the box"; done
bx "chmod +x ~/rstud/src/box/*.sh ~/rstud/src/remote/*.sh"
AC=$(bx "bash ~/rstud/src/box/ops.sh argv-check"); arc=$?; printf '%s\n' "$AC" | sed 's/^/  box /'
[ $arc = 0 ] || die "argv-check failed: the rental argv must equal the frozen box T_b sheet"
BCHK=$(bx 'B=~/rstud/bundle; [ -e $B/BUNDLE_OK ] || { echo "NO_BUNDLE"; exit; }
  m=$(sed -n "s/.*\"mode\": \"\([A-Z]*\)\".*/\1/p" $B/BUNDLE.json); t=$(sed -n "s/.*\"table_sha256\": \"\([0-9a-f]*\)\".*/\1/p" $B/BUNDLE.json)
  r=$(sed -n "s/.*\"labels_ready_sha256\": \"\([0-9a-f]*\)\".*/\1/p" $B/BUNDLE.json)
  rn=$(sha256sum ~/r7t_labels/final/LABELS_READY 2>/dev/null | cut -c1-64); tn=$(sha256sum ~/r7t_labels/final/labels_r7t_b.csv 2>/dev/null | cut -c1-64)
  k=OK; for f in ~/rstud/src/remote/*; do cmp -s $f $B/kit/$(basename $f) || k="STALE:$(basename $f)"; done
  echo "mode=$m table_in_bundle=${t:0:16} table_final_now=${tn:0:16} ready_in_bundle=${r:0:16} ready_now=${rn:0:16} kit=$k"')
log "bundle: $BCHK"
case "$BCHK" in NO_BUNDLE*) [ $MODE = dry ] && log "DRY: no bundle yet (pack.sh REAL after LABELS_READY)" || die "no bundle: run pack.sh REAL on the box";;
  *) eval "$(printf '%s' "$BCHK" | tr ' ' '\n' | sed 's/^/B_/')"
     if [ "$B_mode" != REAL ] || [ "$B_table_in_bundle" != "$B_table_final_now" ] || [ "$B_ready_in_bundle" != "$B_ready_now" ] || [ "$B_kit" != OK ]; then
       [ $MODE = dry ] && log "DRY: bundle not launchable yet (needs REAL, current LABELS_READY + table, current kit)" || die "bundle not launchable: re-run pack.sh REAL"; fi;;
esac

# ---- 1 account: 2FA, no other instance, ledger ----
IID=$(st_get INSTANCE_ID)
if [ -n "$IID" ] && [ "$(st_get DESTROYED)" != 1 ]; then log "RESUME: state.env holds instance $IID: not renting another"; else
  [ -n "$IID" ] && { mv "$STATE" "$KIT/state.$(date +%m%d_%H%M%S).env"; IID=""; }
  C0=$(credit_now) || die "Vast 2FA session needed: Eesh runs vastai tfa send-email, then vastai tfa login -t email -s <secret> -c <code>" 3
  NI=$(n_instances) || die "cannot list instances"
  [ "$NI" = 0 ] || die "$NI instance(s) already on the account: refusing (one rental at a time; check the vastR7T / Raptor kits)"
  O7=$KIT/../r7teacher/labels/kits/vast_r7t/state.env
  [ -f "$O7" ] && [ -n "$(kv_get "$O7" INSTANCE_ID)" ] && [ "$(kv_get "$O7" DESTROYED)" != 1 ] && die "the vastR7T kit records a live instance: refusing"
  SPENT=$(spent_total); CL=$("$PYL" -c "print(round($CAP_USD - $SPENT - $RESERVE_USD, 2))")
  BUDGET=$("$PYL" -c "print(round(min($CL, $C0 - $FLOOR_USD), 2))")
  log "credit \$$C0 | kit spend so far \$$SPENT | budget for this rental \$$BUDGET (= min(cap - spent - reserve, credit - floor))"
  num_ge "$BUDGET" 3 || die "budget \$$BUDGET too small"
  # ---- 2 the offer, re-checked now, and the spend projection guard ----
  F=$(search o$OFFER)
  RJ=$("$PYL" "$KIT/offers_rank.py" $F --cap_left "$BUDGET" --max_h "$MAX_RUN_H" --per_gpu_max "$PER_GPU_MAX" --nfolds "$(echo $RSTUD_FOLDS | wc -w)" --json --id "$OFFER")
  [ -n "$RJ" ] && [ "$RJ" != "[]" ] || die "offer $OFFER not in a fresh search (gone or outside the plan): --search again"
  eval "$(printf '%s' "$RJ" | jq_py 'import shlex
r = d[0]
for k in ("gpu", "n", "dph", "per_gpu", "geo", "rel", "vcpu", "fold_h", "run_mid", "run_w", "c_mid", "c_w", "din", "fits", "why", "fold_est_s", "dc"): print("O_%s=%s" % (k, shlex.quote(str(r[k]))))')"
  log "offer $OFFER: ${O_n} x $O_gpu \$$O_dph/h (\$$O_per_gpu/GPU-h) | $O_geo | rel $O_rel | vCPU $O_vcpu | DC $O_dc | MODEL fold ${O_fold_h} h | run ${O_run_mid} h mid / ${O_run_w} h worst | cost \$$O_c_mid mid / \$$O_c_w worst (data in+out \$$O_din) | fits $O_fits ${O_why}"
  num_ge "$MAX_DPH" "$O_dph" || die "offer \$$O_dph/h above MAX_DPH \$$MAX_DPH/h"
  [ "$O_fits" = True ] || die "offer fails the rules or the projection (worst \$$O_c_w vs budget \$$BUDGET): $O_why"
  PROJ=$("$PYL" -c "print(round($SPENT + $O_c_w, 2))")
  num_ge "$CAP_USD" "$PROJ" || die "SPEND GUARD: projected total \$$PROJ > cap \$$CAP_USD"
  GUARD_H=$("$PYL" -c "print(round(min(($BUDGET - $O_din) / $O_dph, $MAX_RUN_H), 2))")
  num_ge "$GUARD_H" "$("$PYL" -c "print($O_run_w + 0.75)")" || die "guard window ${GUARD_H} h < worst run ${O_run_w} h + 0.75 h"
  NOW=$(date +%s); GDL=$("$PYL" -c "print(int($NOW + $GUARD_H * 3600))"); BDL=$((NOW + BOOT_MAX_MIN * 60)); LDL=$((GDL - 2700))
  log "SPEND GUARD OK: projected total \$$PROJ <= \$$CAP_USD | guard self-destroy $(date -d @$GDL '+%m-%d %H:%M') (${GUARD_H} h) | boot fuse $(date -d @$BDL '+%H:%M') | lanes must end by $(date -d @$LDL '+%m-%d %H:%M')"
  B64=$(grep -v '^#' "$KIT/remote/guard.sh" | gzip -9c | base64 -w0)
  ONSTART="mkdir -p $R/status; echo $B64 | base64 -d | gunzip > $R/guard.sh; chmod 700 $R/guard.sh; (setsid nohup bash $R/guard.sh >> $R/guard.out 2>&1 < /dev/null &)"
  log "onstart = guard bootstrap (${#ONSTART} chars)"
  [ $MODE = dry ] && { log "DRY: would create: --image $IMAGE --disk $DISK_GB --ssh --direct --env '-e RSNA_GUARD_DEADLINE=$GDL -e RSNA_BOOT_DEADLINE=$BDL'; nothing rented"; exit 0; }
  [ -n "$(kv_get "$LEDGER" CREDIT_START)" ] || { kv_set "$LEDGER" CREDIT_START "$C0"; kv_set "$LEDGER" CREDIT_START_AT "$(date '+%F %T')"; log "ledger: CREDIT_START \$$C0"; }
  LABEL=rsna-rstud-s1047-$(date +%m%d%H%M%S); st_set LABEL $LABEL; rm -f "$KH"; bx 'rm -f ~/.ssh/known_hosts_rstud'
  CJ=$(vast_raw create instance "$OFFER" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --label "$LABEL" \
        --env "-e RSNA_GUARD_DEADLINE=$GDL -e RSNA_BOOT_DEADLINE=$BDL" --onstart-cmd "$ONSTART" --cancel-unavail)
  IID=$(printf '%s' "$CJ" | jq_py 'print(d.get("new_contract", ""))' 2>/dev/null)
  if [ -z "$IID" ]; then sleep 20; IID=$(instances_json | jq_py "
print(next((str(i['id']) for i in (d if isinstance(d, list) else d.get('instances', [])) if i.get('label') == '$LABEL'), ''))"); fi
  [ -n "$IID" ] || die "create failed: nothing rented"
  st_set INSTANCE_ID $IID; st_set DESTROYED 0; st_set OFFER $OFFER; st_set DPH $O_dph; st_set NGPU $O_n; st_set GPU "$O_gpu"; st_set DIN $O_din
  st_set LAUNCH_EPOCH $NOW; st_set GUARD_DEADLINE $GDL; st_set BOOT_DEADLINE $BDL; st_set LANE_DEADLINE $LDL; st_set SPENT_BEFORE $SPENT
  st_set FOLD_EST_S $O_fold_est_s; st_set CREDIT_AT_CREATE $C0
  log "RENTED $IID ($LABEL) \$$O_dph/h"
fi
[ $MODE = dry ] && exit 0

# ---- 3 running + laptop ssh ----
t0=$(date +%s); badn=0
while :; do
  J=$(vast_raw show instance "$IID") || { [ $? = 3 ] && die "2FA lost while waiting (the guard boot fuse destroys the instance by $(date -d @$(st_get BOOT_DEADLINE) '+%H:%M'))" 3; }
  S=$(printf '%s' "$J" | jq_py 'print(d.get("actual_status"))' 2>/dev/null); [ "$S" = running ] && break
  case $S in exited|offline|unknown) badn=$((badn + 1));; esac
  if [ $badn -ge 4 ] || [ $(( $(date +%s) - t0 )) -gt 1200 ]; then log "never reached running ($S): destroying"; bash "$KIT/destroy.sh" --yes; die "launch failed: --search again"; fi
  sleep 20
done
eval "$(printf '%s' "$J" | jq_py '
p = (d.get("ports") or {}).get("22/tcp") or []
print("DH=%s; DP=%s; PH=%s; PP=%s" % (d.get("public_ipaddr") or "", (p[0].get("HostPort") if p else "") or "", d.get("ssh_host") or "", d.get("ssh_port") or ""))')"
ok=""; for i in $(seq 1 40); do for rt in "$DH $DP" "$PH $PP"; do set -- $rt; [ -n "${1:-}" ] && [ -n "${2:-}" ] || continue
  st_set I_HOST "$1"; st_set I_PORT "$2"; ix 'echo laptop-ok' 2>/dev/null | grep -q laptop-ok && { ok="$1:$2"; break 2; }; done; sleep 15; done
[ -n "$ok" ] || { bash "$KIT/destroy.sh" --yes; die "laptop cannot ssh to $IID: destroyed"; }
# review 10-14: the guard is the ONLY destroy path that does not need this laptop (the box has no Vast key), so an
# instance whose guard is not armed (self-test not http 200, or hard deadline / boot fuse not the ones set here) is destroyed now
GDLU="hard deadline $(date -u -d @$(st_get GUARD_DEADLINE) '+%F %T')Z"; BDLU="boot fuse $(date -u -d @$(st_get BOOT_DEADLINE) '+%F %T')Z"; gok=""
for i in $(seq 1 12); do
  ix "grep -qE 'self-test: .*-> http 200 ' $R/guard.log && grep -qF '$GDLU' $R/guard.log && grep -qF '$BDLU' $R/guard.log" 2>/dev/null && { gok=1; break; }
  sleep 10; done
log "laptop -> instance ssh OK via $ok | guard: $(ix "tail -n 2 $R/guard.log 2>/dev/null" | tr '\n' ' ')"
[ -n "$gok" ] || { log "GUARD NOT ARMED (want self-test http 200 + '$GDLU' + '$BDLU' in guard.log)"; bash "$KIT/destroy.sh" --yes; die "guard not armed: destroyed; --search again"; }
log "guard ARMED: self-test http 200 | $GDLU | $BDLU"

# ---- 4 box PUBLIC key on the instance (newline-safe: lesson 10-13) + box route ----
BOX_PUB=$(bx 'bash ~/rstud/src/box/ops.sh keypub' | tail -n 1); case $BOX_PUB in ssh-ed25519\ *) ;; *) bash "$KIT/destroy.sh" --yes; die "box pubkey unreadable: destroyed";; esac
printf '%s\n' "$BOX_PUB" | ix 'mkdir -p /root/.ssh && k=$(cat) && (grep -qxF "$k" /root/.ssh/authorized_keys 2>/dev/null || printf "\n%s\n" "$k" >> /root/.ssh/authorized_keys) && sed -i "/^$/d" /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys'
# 10-15 01:20 (incident 55960494): Vast can re-sync authorized_keys from the account keys mid-run and drop the box key ->
# keep a copy of the box PUBLIC key on the instance and a keeper that re-appends it (newline-safe) every 30 s
printf '%s\n' "$BOX_PUB" | ix 'cat > /root/.ssh/box_key.pub && chmod 644 /root/.ssh/box_key.pub' || log "WARN: box_key.pub copy failed"
ix 'cat > /root/key_keeper.sh && chmod 700 /root/key_keeper.sh && (setsid nohup bash /root/key_keeper.sh > /dev/null 2>&1 < /dev/null &) && sleep 2 && kill -0 $(cat /root/key_keeper.pid)' < "$KIT/instance_key_keeper.sh" \
  && log "box key keeper started on the instance (/root/key_keeper.sh, 30 s)" || log "WARN: box key keeper NOT started (the box loses the fetch route if Vast re-syncs authorized_keys)"
bok=""; for i in $(seq 1 8); do for rt in "$DH $DP" "$PH $PP"; do set -- $rt; [ -n "${1:-}" ] && [ -n "${2:-}" ] || continue
  bx "bash ~/rstud/src/box/ops.sh sshtest $1 $2" | grep -q box-to-instance-ok && { bok="$1 $2"; break 2; }; done; sleep 15; done
[ -n "$bok" ] || { bash "$KIT/destroy.sh" --yes; die "the box cannot ssh to the instance (no fetch possible): destroyed"; }
set -- $bok; st_set B_HOST $1; st_set B_PORT $2; log "box -> instance ssh OK via $1:$2"

# ---- 5 run.env + box driver (from here the laptop is not needed) ----
bx_in "cat > ~/rstud/run.env" <<EOF
INSTANCE_ID=$IID
OFFER=$(st_get OFFER)
GPU_NAME=$(st_get GPU | tr ' ' _)
NG=$(st_get NGPU)
DPH=$(st_get DPH)
DIN=$(st_get DIN)
SPENT_BEFORE=$(st_get SPENT_BEFORE)
CAP=$CAP_USD
LAUNCH_EPOCH=$(st_get LAUNCH_EPOCH)
GUARD_DEADLINE=$(st_get GUARD_DEADLINE)
BOOT_DEADLINE=$(st_get BOOT_DEADLINE)
LANE_DEADLINE=$(st_get LANE_DEADLINE)
FOLD_EST_S=$(st_get FOLD_EST_S)
I_HOST=$1
I_PORT=$2
I_ROOT=$R
FOLDS="$RSTUD_FOLDS"
SMOKE_MIN=10
SMOKE_MAX=20
PYI=/opt/conda/bin/python
EOF
[ -n "$(st_get DRIVER_STARTED)" ] || { bx "setsid -f bash ~/rstud/src/box/driver.sh > ~/rstud/logs/driver.out 2>&1 < /dev/null" && st_set DRIVER_STARTED "$(date +%s)"; }
log "box driver started (log ~/rstud/driver.log, then ~/rstud/fetch/fetch.log): push -> setup -> parity -> placeholders -> 10-min smoke -> run -> fetch"
( nohup bash "$KIT/watch.sh" > "$LOGDIR/watch.out" 2>&1 < /dev/null & ); log "laptop backup watcher started (dies with this session; box driver + on-instance guard are primary)"
log "LAUNCH DONE. Monitor: bash $KIT/status.sh"
