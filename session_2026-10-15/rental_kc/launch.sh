#!/bin/bash
# launch.sh (rental_kc, laptop, Git Bash, lab network): rent 2 x A100 for the KC1 all-data members (+ KC2 f0) and hand the
# run to the BOX. Spends money only in run mode, and only after every check below passes. No em dashes.
#   bash launch.sh --search            read only: 1 narrow search (2 x A100, limit 25), ranked (offers_rank.py)
#   bash launch.sh --dry-run OFFER     read only: preflight + fresh offer check + plan + spend projection; creates nothing
#   bash launch.sh OFFER               preflight -> projection guard -> create (on-instance guard armed via onstart) -> wait
#                                      running -> laptop + box ssh -> guard ARMED check -> box key + key keeper on the
#                                      instance -> run.env + box driver.sh (push + rate gate, setup, parity, smoke, WAIT for the
#                                      f0 early G1, GO, fetch loop) -> laptop backup watcher. Resumable via state.env.
# Knobs: RKC_OWNER (must equal OWNER.lock), PIN_MACHINE=<machine_id> (narrows the search to that machine; Vast lists one
# machine per host per call), GATE_LATEST_HM=HH:MM (today, IST: the driver aborts if no early G1 by then; default now + 3 h),
# WAIT_H (expected wait for the gate, for the projection; default 0.75), V100_H (MODEL member hours on the V100), CANDIDATES.
# Money: ledger CREDIT_START (set once, at the first rental of this kit); spent = credit drop since; budget = min(CAP - spent -
# reserve, credit - floor); abort unless the plan-2 WORST cost fits; guard hard deadline = min((budget - data) / $/h, 7 h);
# boot fuse = GATE_LATEST + 20 min (no RUN_STARTED by then -> the guard self-destroys).
SCRIPT_TAG=launch; source "$(dirname "$0")/config.sh"
MODE=run; OFFER=""
while [ $# -gt 0 ]; do case $1 in --search) MODE=search;; --dry-run) MODE=dry;; [0-9]*) OFFER=$1;; *) die "bad arg $1";; esac; shift; done
PINQ=""; [ -n "${PIN_MACHINE:-}" ] && { case $PIN_MACHINE in *[!0-9]*) die "PIN_MACHINE must be digits";; esac; PINQ=" machine_id=$PIN_MACHINE"; }
QA="rentable=true verified=true reliability>=0.98 cuda_vers>=12.1 disk_space>=$DISK_GB inet_down>=300 num_gpus=2 gpu_name in [A100_SXM4,A100_PCIE] gpu_ram>=39 cpu_cores_effective>=16$PINQ"
RANKA=(--max_dph "$MAX_DPH" --max_h "$MAX_RUN_H" --wait_h "${WAIT_H:-0.75}" --v100_h "${V100_H:-5.1}" --bad_hosts "$BAD_HOSTS")
search(){ local f=$LOGDIR/offers_a100x2_$1.json
  "$VAST_BIN" search offers -n "$QA" --storage "$DISK_GB" --limit 25 -o dph_total --raw > "$f" 2>"$f.err" || die "search failed: $(head -c 200 "$f.err") (2FA?)"
  grep -q -i "two factor" "$f" && die "search needs a 2FA session" 3
  echo "$f"; }
cap_left(){ local s; s=$(spent_total) || return $?; "$PYL" -c "print(round($CAP_USD - $s - $RESERVE_USD, 2))"; }
if [ $MODE = search ]; then
  CL=$(cap_left) || die "2FA session needed for the credit read" 3
  F=$(search $(date +%m%d_%H%M%S)); log "search (read only, limit 25) | cap left \$$CL | $QA"
  "$PYL" "$KIT/offers_rank.py" $F --cap_left "$CL" "${RANKA[@]}"
  log "then: RKC_OWNER=<owner> bash launch.sh --dry-run <id>   and   RKC_OWNER=<owner> PIN_MACHINE=<machine> bash launch.sh <id>"; exit 0
fi
[ -n "$OFFER" ] || die "usage: launch.sh --search | --dry-run OFFER | OFFER"
LOCKO=$(kv_get "$KIT/OWNER.lock" owner); [ -n "$LOCKO" ] || die "no owner in $KIT/OWNER.lock: refusing"
[ "${RKC_OWNER:-}" = "$LOCKO" ] || die "OWNER.lock names $LOCKO, not '${RKC_OWNER:-unset}': refusing to rent"
log "== launch $MODE | offer $OFFER | cap \$$CAP_USD total new spend (floor \$$FLOOR_USD, reserve \$$RESERVE_USD) | max \$$MAX_DPH/h"

# ---- 0 preflight (box, kit, bundle, argv, code) ----
bx true || die "box unreachable (lab network needed: the box drives the run and pulls the members)"
bx "mkdir -p ~/$BXD/src/box ~/$BXD/src/remote ~/$BXD/src/tests ~/$BXD/logs" || die "box mkdir"
for f in box/*.sh box/*.py remote/* tests/*; do [ -f "$KIT/$f" ] || continue
  bx_in "cat > ~/$BXD/src/$f.tmp && mv -f ~/$BXD/src/$f.tmp ~/$BXD/src/$f" < "$KIT/$f" || die "sync $f to the box"; done
bx "chmod +x ~/$BXD/src/box/*.sh ~/$BXD/src/remote/*.sh"
BCHK=$(bx 'B=~/rkc/bundle; [ -e $B/BUNDLE_OK ] || { echo "NO_BUNDLE"; exit; }
  k=OK; for f in ~/rkc/src/remote/*; do cmp -s $f $B/kit/$(basename $f) || k="STALE:$(basename $f)"; done
  c=$(bash ~/rkc/src/box/ops.sh code-check $B/BUNDLE.json | tr " " _)
  a=$(bash ~/rkc/src/box/ops.sh argv-check $B/kit/members.tsv s2003 2>&1 | head -n 1 | grep -c IDENTICAL)
  echo "kit=$k code=$c argv_s2003=$a"')
log "bundle: $BCHK"
case "$BCHK" in NO_BUNDLE*) [ $MODE = dry ] && log "DRY: no bundle yet (pack.sh after C8 + M4 + C5)" || die "no bundle: run ~/rkc/src/box/pack.sh on the box";;
  *) eval "$(printf '%s' "$BCHK" | tr ' ' '\n' | sed 's/^/B_/')"
     if [ "$B_kit" != OK ] || [ "$B_code" != CODE_SAME ] || [ "$B_argv_s2003" != 1 ]; then
       [ $MODE = dry ] && log "DRY: bundle not launchable yet (kit $B_kit, code $B_code, s2003 argv identical $B_argv_s2003)" || die "bundle not launchable: re-run pack.sh ($BCHK)"; fi;;
esac

# ---- 1 account: 2FA, no other instance, ledger ----
IID=$(st_get INSTANCE_ID)
if [ -n "$IID" ] && [ "$(st_get DESTROYED)" != 1 ]; then log "RESUME: state.env holds instance $IID: not renting another"; else
  [ -n "$IID" ] && { mv "$STATE" "$KIT/state.$(date +%m%d_%H%M%S).env"; IID=""; }
  C0=$(credit_now) || die "Vast 2FA session needed: Eesh runs vastai tfa send-email, then vastai tfa login -t email -s <secret> -c <code>" 3
  NI=$(n_instances) || die "cannot list instances"
  [ "$NI" = 0 ] || die "$NI instance(s) already on the account: refusing (exactly one rental at a time)"
  SPENT=$(spent_total); CL=$("$PYL" -c "print(round($CAP_USD - $SPENT - $RESERVE_USD, 2))")
  BUDGET=$("$PYL" -c "print(round(min($CL, $C0 - $FLOOR_USD), 2))")
  log "credit \$$C0 | kit spend so far \$$SPENT | budget for this rental \$$BUDGET (= min(cap - spent - reserve, credit - floor))"
  num_ge "$BUDGET" 2.5 || die "budget \$$BUDGET too small for even one round of 2 members"
  # ---- 2 the offer, re-checked now, and the spend projection guard ----
  F=$(search o$OFFER)
  RJ=$("$PYL" "$KIT/offers_rank.py" $F --cap_left "$BUDGET" "${RANKA[@]}" --json --id "$OFFER")
  [ -n "$RJ" ] && [ "$RJ" != "[]" ] || die "offer $OFFER not in a fresh search (gone or outside the plan): --search again"
  eval "$(printf '%s' "$RJ" | jq_py 'import shlex
r = d[0]
for k in ("gpu", "n", "dph", "geo", "rel", "vcpu", "member_h", "run_mid", "run_w", "c_mid", "c_w", "c4_mid", "din", "fits", "why", "member_est_s", "dc", "host", "machine"): print("O_%s=%s" % (k, shlex.quote(str(r[k]))))')"
  log "offer $OFFER: ${O_n} x $O_gpu \$$O_dph/h | $O_geo | host $O_host machine $O_machine | rel $O_rel | vCPU $O_vcpu | MODEL member ${O_member_h} h | plan-2 run ${O_run_mid} h mid / ${O_run_w} h worst | \$$O_c_mid mid / \$$O_c_w worst (data \$$O_din) | plan-4 \$$O_c4_mid mid | fits $O_fits ${O_why}"
  num_ge "$MAX_DPH" "$O_dph" || die "offer \$$O_dph/h above MAX_DPH \$$MAX_DPH/h"
  [ "$O_fits" = True ] || die "offer fails the rules or the projection (plan-2 worst \$$O_c_w vs budget \$$BUDGET): $O_why"
  PROJ=$("$PYL" -c "print(round($SPENT + $O_c_w, 2))")
  num_ge "$CAP_USD" "$PROJ" || die "SPEND GUARD: projected total \$$PROJ > cap \$$CAP_USD"
  GUARD_H=$("$PYL" -c "print(round(min(($BUDGET - $O_din) / $O_dph, $MAX_RUN_H), 2))")
  num_ge "$GUARD_H" "$O_run_w" || die "guard window ${GUARD_H} h < plan-2 worst run ${O_run_w} h"
  NOW=$(date +%s); GDL=$("$PYL" -c "print(int($NOW + $GUARD_H * 3600))")
  if [ -n "${GATE_LATEST_HM:-}" ]; then GL=$(date -d "$(date +%F) $GATE_LATEST_HM" +%s) || die "GATE_LATEST_HM"; else GL=$((NOW + 10800)); fi
  [ $GL -gt $((NOW + 1800)) ] || die "GATE_LATEST is less than 30 min away"
  BDL=$((GL + 1200)); [ $BDL -lt $GDL ] || die "boot fuse (gate latest + 20 min) is not before the guard deadline"
  log "SPEND GUARD OK: projected total \$$PROJ <= \$$CAP_USD | guard self-destroy $(date -d @$GDL '+%m-%d %H:%M') (${GUARD_H} h) | gate latest $(date -d @$GL '+%H:%M') | boot fuse $(date -d @$BDL '+%H:%M')"
  B64=$(grep -v '^#' "$KIT/remote/guard.sh" | gzip -9c | base64 -w0)
  ONSTART="mkdir -p $R/status; echo $B64 | base64 -d | gunzip > $R/guard.sh; chmod 700 $R/guard.sh; (setsid nohup bash $R/guard.sh >> $R/guard.out 2>&1 < /dev/null &)"
  log "onstart = guard bootstrap (${#ONSTART} chars)"
  [ $MODE = dry ] && { log "DRY: would create: --image $IMAGE --disk $DISK_GB --ssh --direct --env '-e RSNA_GUARD_DEADLINE=$GDL -e RSNA_BOOT_DEADLINE=$BDL'; nothing rented"; exit 0; }
  NI=$(n_instances) || die "cannot list instances"; [ "$NI" = 0 ] || die "$NI instance(s) appeared on the account: refusing"
  [ -n "$(kv_get "$LEDGER" CREDIT_START)" ] || { kv_set "$LEDGER" CREDIT_START "$C0"; kv_set "$LEDGER" CREDIT_START_AT "$(date '+%F %T')"; log "ledger: CREDIT_START \$$C0"; }
  LABEL=rsna-rkc-$(date +%m%d%H%M%S); st_set LABEL $LABEL; rm -f "$KH"; bx 'rm -f ~/.ssh/known_hosts_rkc'
  CJ=$(vast_raw create instance "$OFFER" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --label "$LABEL" \
        --env "-e RSNA_GUARD_DEADLINE=$GDL -e RSNA_BOOT_DEADLINE=$BDL" --onstart-cmd "$ONSTART" --cancel-unavail)
  IID=$(printf '%s' "$CJ" | jq_py 'print(d.get("new_contract", ""))' 2>/dev/null)
  if [ -z "$IID" ]; then sleep 20; IID=$(instances_json | jq_py "
print(next((str(i['id']) for i in (d if isinstance(d, list) else d.get('instances', [])) if i.get('label') == '$LABEL'), ''))"); fi
  [ -n "$IID" ] || die "create failed: nothing rented"
  st_set INSTANCE_ID $IID; st_set DESTROYED 0; st_set OFFER $OFFER; st_set DPH $O_dph; st_set NGPU $O_n; st_set GPU "$O_gpu"; st_set DIN $O_din
  st_set LAUNCH_EPOCH $NOW; st_set GUARD_DEADLINE $GDL; st_set BOOT_DEADLINE $BDL; st_set GATE_LATEST $GL; st_set SPENT_BEFORE $SPENT
  st_set CREDIT_AT_CREATE $C0; st_set HOST "$O_host"; st_set MACHINE "$O_machine"; st_set GEO "$O_geo"
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
GDLU="hard deadline $(date -u -d @$(st_get GUARD_DEADLINE) '+%F %T')Z"; BDLU="boot fuse $(date -u -d @$(st_get BOOT_DEADLINE) '+%F %T')Z"; gok=""
for i in $(seq 1 12); do
  ix "grep -qE 'self-test: .*-> http 200 ' $R/guard.log && grep -qF '$GDLU' $R/guard.log && grep -qF '$BDLU' $R/guard.log" 2>/dev/null && { gok=1; break; }
  sleep 10; done
log "laptop -> instance ssh OK via $ok | guard: $(ix "tail -n 2 $R/guard.log 2>/dev/null" | tr '\n' ' ')"
[ -n "$gok" ] || { log "GUARD NOT ARMED (want self-test http 200 + '$GDLU' + '$BDLU' in guard.log)"; bash "$KIT/destroy.sh" --yes; die "guard not armed: destroyed; --search again"; }
log "guard ARMED: self-test http 200 | $GDLU | $BDLU"

# ---- 4 box PUBLIC key on the instance (newline-safe) + key keeper (Vast re-syncs authorized_keys: 10-15 incident) ----
BOX_PUB=$(bx "bash ~/$BXD/src/box/ops.sh keypub" | tail -n 1); case $BOX_PUB in ssh-ed25519\ *) ;; *) bash "$KIT/destroy.sh" --yes; die "box pubkey unreadable: destroyed";; esac
printf '%s\n' "$BOX_PUB" | ix 'mkdir -p /root/.ssh && k=$(cat) && (grep -qxF "$k" /root/.ssh/authorized_keys 2>/dev/null || printf "\n%s\n" "$k" >> /root/.ssh/authorized_keys) && sed -i "/^$/d" /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys'
printf '%s\n' "$BOX_PUB" | ix 'cat > /root/.ssh/box_key.pub && chmod 644 /root/.ssh/box_key.pub' || log "WARN: box_key.pub copy failed"
ix 'cat > /root/key_keeper.sh && chmod 700 /root/key_keeper.sh && (setsid nohup bash /root/key_keeper.sh > /dev/null 2>&1 < /dev/null &) && sleep 2 && kill -0 $(cat /root/key_keeper.pid)' < "$KIT/instance_key_keeper.sh" \
  && log "box key keeper started on the instance (/root/key_keeper.sh, 30 s)" || log "WARN: box key keeper NOT started (the box loses the fetch route if Vast re-syncs authorized_keys)"
bok=""; for i in $(seq 1 8); do for rt in "$DH $DP" "$PH $PP"; do set -- $rt; [ -n "${1:-}" ] && [ -n "${2:-}" ] || continue
  bx "bash ~/$BXD/src/box/ops.sh sshtest $1 $2" | grep -q box-to-instance-ok && { bok="$1 $2"; break 2; }; done; sleep 15; done
[ -n "$bok" ] || { bash "$KIT/destroy.sh" --yes; die "the box cannot ssh to the instance (no fetch possible): destroyed"; }
set -- $bok; st_set B_HOST $1; st_set B_PORT $2; log "box -> instance ssh OK via $1:$2"

# ---- 5 run.env + box driver (from here the laptop is not needed) + box fallback writer ----
bx_in "cat > ~/$BXD/run.env" <<EOF
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
GATE_LATEST=$(st_get GATE_LATEST)
I_HOST=$1
I_PORT=$2
I_ROOT=$R
PYI=/opt/conda/bin/python
SMOKE1_MIN=${SMOKE1_MIN:-8}
SMOKE2_MIN=${SMOKE2_MIN:-6}
CANDIDATES="${CANDIDATES:-}"
EOF
[ -n "$(st_get DRIVER_STARTED)" ] || { bx "setsid -f bash ~/$BXD/src/box/driver.sh > ~/$BXD/logs/driver.out 2>&1 < /dev/null" && st_set DRIVER_STARTED "$(date +%s)"; }
log "box driver started (log ~/$BXD/driver.log, then ~/$BXD/fetch/fetch.log): push -> setup -> parity -> smoke -> WAIT early G1 -> GO -> fetch"
# main 10-15 15:05: fallback armed WITH the automatic re-queue (KC_FALLBACK_QCTL=1, as tb_fallback 10-14); every qctl call logged
bx "KC_FALLBACK_QCTL=${KC_FALLBACK_QCTL:-1} setsid -f bash ~/$BXD/src/box/kc_fallback.sh --loop > /dev/null 2>&1 < /dev/null" && log "box kc_fallback.sh --loop armed (KC_FALLBACK_QCTL=${KC_FALLBACK_QCTL:-1}: re-queues a reserved member the rental did not deliver after its line was REFUSED; log ~/$BXD/logs/kc_fallback.log)"
( nohup bash "$KIT/watch.sh" > "$LOGDIR/watch.out" 2>&1 < /dev/null & ); log "laptop backup watcher started (dies with this session; box driver + on-instance guard are primary)"
log "LAUNCH DONE. Monitor: bash $KIT/status.sh"
