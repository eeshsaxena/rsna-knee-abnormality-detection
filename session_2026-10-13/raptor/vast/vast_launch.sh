#!/bin/bash
# vast_launch.sh: RSNA-ONLY Raptor seeds on a rented GPU (path b). Laptop (Git Bash), lab network. ONE command after a search:
#   bash vast_launch.sh --search [V100_SPS]        read only: 2 narrow searches (limit 25 each), ranked for SEEDS (config.sh)
#   bash vast_launch.sh --dry-run OFFER [V100_SPS] read only: checks + the exact plan
#   bash vast_launch.sh OFFER [V100_SPS]           rent (spends money) -> guard -> box key -> code + urls -> setup -> lanes
#                                                  -> box-side fetch loop -> laptop backup watcher. Resumable via state.env.
# V100_SPS = the box smoke's measured s/step (default .65); rental s/step is modelled from it and then MEASURED at launch.
SCRIPT_TAG=launch; source "$(dirname "$0")/config.sh"
MODE=run; OFFER=""; V=0.65
while [ $# -gt 0 ]; do case $1 in --search) MODE=search;; --dry-run) MODE=dry;; [0-9]*.[0-9]*) V=$1;; [0-9]*) OFFER=$1;; *) die "bad arg $1";; esac; shift; done
NS=$(echo $SEEDS | wc -w)
BASE='rentable=true verified=true reliability>=0.98 inet_down>=500 cuda_vers>=12.1 disk_space>=200 gpu_ram>=23'
Q1="$BASE gpu_name=RTX_4090 num_gpus>=2 num_gpus<=4 cpu_cores_effective>=16"
Q2="$BASE num_gpus=1 gpu_name in [A100_SXM4,A100_PCIE,H100_SXM,H100_PCIE,L40S] cpu_cores_effective>=8"
search(){ local f1=$LOGDIR/offers_4090_$1.json f2=$LOGDIR/offers_big_$1.json
  "$VAST_BIN" search offers -n "$Q1" --storage $DISK_GB --limit 25 -o dph_total --raw > $f1 2>/dev/null || die "search 1 failed (2FA?)"
  "$VAST_BIN" search offers -n "$Q2" --storage $DISK_GB --limit 25 -o dph_total --raw > $f2 2>/dev/null || die "search 2 failed (2FA?)"
  echo "$f1 $f2"; }
if [ $MODE = search ]; then
  F=$(search $(date +%m%d_%H%M%S)); log "search (read only, 2 x limit 25): $Q1 || $Q2"
  "$PYL" "$KIT/offers_rank.py" $F --v100_sps $V --seeds $NS --budget $("$PYL" -c "print($CAP_USD - $RESERVE_USD)") --max_h $MAX_RUN_H
  log "then: bash vast_launch.sh --dry-run <id> $V   and   bash vast_launch.sh <id> $V"; exit 0
fi
[ -n "$OFFER" ] || die "usage: vast_launch.sh [--dry-run] OFFER [V100_SPS]"

# ---- 0 pre-flight: box reachable, smoke passed, kit complete, nothing OAI in the kit ----
bx true || die "box unreachable (lab network needed: the box runs the fetch loop)"
# main 20:10: no wait for the box smoke; the rental runs its OWN bf16 smoke (step 5) before any seed starts
for f in remote/setup.sh remote/lanes.sh remote/guard.sh remote/labels_raptor_rsna_blendjev_sol.csv remote/gold74_eval_set.csv \
         ../box/train_raptor.py ../box/raptor_ref.py ../box/unpack_d96.py ../prep/fetch_urls.py box_fetch.sh; do [ -s "$KIT/$f" ] || die "kit file missing: $f"; done
grep -l -i 'oai_' "$KIT/remote/labels_raptor_rsna_blendjev_sol.csv" "$KIT/remote/gold74_eval_set.csv" && die "OAI id in a kit csv"
log "== vast_launch $MODE | offer $OFFER | seeds [$SEEDS] | V100 s/step $V | cap \$$CAP_USD floor \$$FLOOR_USD"

# ---- 1 account ----
IID=$(st_get INSTANCE_ID)
if [ -n "$IID" ] && [ "$(st_get DESTROYED)" != 1 ]; then log "RESUME instance $IID"; else
  C0=$(credit_now) || { [ $MODE = dry ] && C0=25 && log "DRY: no 2FA session; assuming credit \$25"; } || die "2FA session needed" 3
  BUDGET=$("$PYL" -c "print(round(min($CAP_USD, $C0 - $FLOOR_USD), 2))"); log "credit \$$C0 -> budget \$$BUDGET"
  num_ge "$BUDGET" 5 || die "budget too small"
  F=$(search o$OFFER)
  RJ=$("$PYL" "$KIT/offers_rank.py" $F --v100_sps $V --seeds $NS --budget $BUDGET --max_h $MAX_RUN_H --json --id $OFFER)
  [ -n "$RJ" ] && [ "$RJ" != "[]" ] || die "offer $OFFER not in a fresh search (gone or filtered): --search again"
  eval "$(printf '%s' "$RJ" | jq_py 'import shlex
r = d[0]
for k in ("gpu", "n", "dph", "geo", "rel", "vcpu", "sps", "seed_h", "rounds", "run_h", "cost", "fits", "din"): print("O_%s=%s" % (k, shlex.quote(str(r[k]))))')"
  log "offer $OFFER: ${O_n}x $O_gpu \$$O_dph/h | $O_geo | rel $O_rel | vCPU $O_vcpu | MODEL s/step $O_sps, seed ${O_seed_h} h, $O_rounds round(s), run ${O_run_h} h, cost \$$O_cost (data-in \$$O_din) | fits $O_fits"
  [ "$O_fits" = True ] || [ $MODE = dry ] || die "offer does not fit budget / time"
  GUARD_H=$("$PYL" -c "print(round(min(($BUDGET - $RESERVE_USD - $O_din) / $O_dph, $MAX_RUN_H), 2))")
  NOW=$(date +%s); GDL=$("$PYL" -c "print(int($NOW + $GUARD_H * 3600))"); DL=$((GDL - 1800))
  log "plan: self-destroy at $(date -d @$GDL '+%m-%d %H:%M') (${GUARD_H} h, spend <= budget - reserve); seeds must finish by $(date -d @$DL '+%m-%d %H:%M')"
  B64=$(gzip -9c "$KIT/remote/guard.sh" | base64 -w0)
  ONSTART="mkdir -p $R/status; echo $B64 | base64 -d | gunzip > $R/guard.sh; chmod 700 $R/guard.sh; (setsid nohup bash $R/guard.sh >> $R/guard.out 2>&1 < /dev/null &)"
  [ $MODE = dry ] && { log "DRY: would create with --image $IMAGE --disk $DISK_GB --ssh --direct, env RSNA_GUARD_DEADLINE=$GDL"; exit 0; }
  LABEL=rsna-raptor-$(date +%m%d%H%M%S); st_set LABEL $LABEL; rm -f "$KH"
  CJ=$(vast_raw create instance "$OFFER" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --label "$LABEL" \
        --env "-e RSNA_GUARD_DEADLINE=$GDL" --onstart-cmd "$ONSTART" --cancel-unavail)
  IID=$(printf '%s' "$CJ" | jq_py 'print(d.get("new_contract", ""))' 2>/dev/null)
  if [ -z "$IID" ]; then sleep 20; IID=$(vast_raw show instances | jq_py "
print(next((str(i['id']) for i in (d if isinstance(d, list) else d.get('instances', [])) if i.get('label') == '$LABEL'), ''))"); fi
  [ -n "$IID" ] || die "create failed: nothing rented"
  st_set INSTANCE_ID $IID; st_set DESTROYED 0; st_set DPH $O_dph; st_set NGPU $O_n; st_set GPU "$O_gpu"; st_set LAUNCH_EPOCH $NOW
  st_set GUARD_DEADLINE $GDL; st_set DEADLINE $DL; st_set CREDIT0 $C0; st_set BUDGET $BUDGET; st_set SEED_S $("$PYL" -c "print(int($O_seed_h * 3600))")
  log "RENTED $IID ($LABEL) \$$O_dph/h. Backup watcher: bash $KIT/vast_watch.sh (started at the end of this script)"
fi
[ $MODE = dry ] && exit 0

# ---- 2 running + ssh (laptop) ----
t0=$(date +%s)
while :; do
  J=$(vast_raw show instance "$IID") || die "2FA lost"
  S=$(printf '%s' "$J" | jq_py 'print(d.get("actual_status"))'); [ "$S" = running ] && break
  [ $(( $(date +%s) - t0 )) -gt 1200 ] && { log "never reached running: destroying"; bash "$KIT/vast_destroy.sh" --yes; die "launch failed"; }
  sleep 20
done
eval "$(printf '%s' "$J" | jq_py '
p = (d.get("ports") or {}).get("22/tcp") or []
print("DH=%s; DP=%s; PH=%s; PP=%s" % (d.get("public_ipaddr") or "", (p[0].get("HostPort") if p else "") or "", d.get("ssh_host") or "", d.get("ssh_port") or ""))')"
ok=""; for i in $(seq 1 40); do for rt in "$DH $DP" "$PH $PP"; do set -- $rt; [ -n "${1:-}" ] && [ -n "${2:-}" ] || continue
  st_set I_HOST "$1"; st_set I_PORT "$2"; ix 'echo laptop-ok' 2>/dev/null | grep -q laptop-ok && { ok="$1:$2"; break 2; }; done; sleep 15; done
[ -n "$ok" ] || die "laptop cannot ssh to $IID (vast_destroy.sh --yes to stop billing)"
log "laptop -> instance ssh OK via $ok | guard: $(ix "tail -n 2 $R/guard.log 2>/dev/null" | tr '\n' ' ')"

# ---- 3 box PUBLIC key on the instance (the box only PULLS) ----
BOX_PUB=$(bx 'cat ~/.ssh/vast_rsna_ed25519.pub'); case $BOX_PUB in ssh-ed25519\ *) ;; *) die "box pubkey unreadable";; esac
printf '%s\n' "$BOX_PUB" | ix 'mkdir -p /root/.ssh && k=$(cat) && (grep -qxF "$k" /root/.ssh/authorized_keys 2>/dev/null || printf "\n%s\n" "$k" >> /root/.ssh/authorized_keys) && sed -i "/^$/d" /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys'
bx "rm -f ~/.ssh/known_hosts_vast; ssh -i ~/.ssh/vast_rsna_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=~/.ssh/known_hosts_vast -p $(st_get I_PORT) root@$(st_get I_HOST) echo box-ok" | grep -q box-ok \
  || die "box cannot ssh to the instance (fetch would be impossible): vast_destroy.sh --yes"
log "box -> instance ssh OK"

# ---- 4 code + RSNA csvs + fresh signed corpus URLs (laptop -> instance; NOTHING from the box) ----
( cd "$RAP/prep" && "$PYL" fetch_urls.py | head -n 1 ) || die "url listing failed"
ix "mkdir -p $R/kit $R/status"
iscp "$KIT/remote/setup.sh" "$KIT/remote/lanes.sh" "$KIT/remote/labels_raptor_rsna_blendjev_sol.csv" "$KIT/remote/gold74_eval_set.csv" \
     "$RAP/box/train_raptor.py" "$RAP/box/raptor_ref.py" "$RAP/box/unpack_d96.py" "$RAP/prep/urls.tsv" "root@$(st_get I_HOST):$R/kit/" || die "scp kit"
CL=$(bx "grep -o 'CHANNELS_LAST [01]' ~/raptor_store/smoke/verdict.txt | awk '{print \$2}'"); CL=${CL:-1}
printf 'DEADLINE_EPOCH=%s\nSEED_S=%s\nCL=%s\nWORKERS=%s\nEPOCHS=16\n' "$(st_get DEADLINE)" "$(st_get SEED_S)" "$CL" 6 | ix "cat > $R/kit/plan.env"
ix "cd $R/kit && md5sum *" | sed 's/^/  kit /'
[ "$(st_get SETUP_OK)" = 1 ] || {
  ix "setsid -f bash $R/kit/setup.sh > /dev/null 2>&1 < /dev/null"; log "setup started (download 37 GB from Kaggle, unpack, OAI assertion, pretrained)"
  while :; do s=$(ix "ls $R/status/" 2>/dev/null); case "$s" in *setup.ok*) break;; *setup.failed*) die "setup FAILED: $(ix "tail -n 5 $R/setup.log")";; esac; sleep 60; done
  st_set SETUP_OK 1; log "SETUP OK: $(ix "grep -E 'OAI_ASSERT|PARTS_VERIFIED|sha ok|versions' $R/setup.log" | tr '\n' ' ' | cut -c1-400)"; }

# ---- 5 rental SMOKE (main 20:10): bf16, <= 15 min, GPU 0, 600 study-steps -> no NaN, s/step, memory, gold58 eval, SWA, strict serve-load ----
if [ "$(st_get SMOKE_OK)" != 1 ]; then
  ix "rm -rf $R/runs/smoke $R/runs/smoke.out $R/status/smoke.rc"
  ix "cd $R/kit && setsid -f bash -c 'CUDA_VISIBLE_DEVICES=0 HF_HOME=$R/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 timeout 900 python train_raptor.py --index $R/rsna_index.csv --labels_rsna $R/kit/labels_raptor_rsna_blendjev_sol.csv --gold $R/kit/gold74_eval_set.csv --out $R/runs/smoke --seed 7 --amp bf16 --channels_last $CL --workers 6 --forbid_oai 1 --smoke_steps 600 > $R/runs/smoke.out 2>&1; echo \$? > $R/status/smoke.rc' < /dev/null > /dev/null 2>&1"
  log "rental smoke started (600 study-steps, bf16, GPU 0, cap 15 min)"
  while ! ix "test -e $R/status/smoke.rc" 2>/dev/null; do sleep 30; done
  SL=$(ix "cat $R/runs/smoke/train.log 2>/dev/null; tail -n 3 $R/runs/smoke.out"); SRC=$(ix "cat $R/status/smoke.rc")
  printf '%s\n' "$SL" | grep -E 's/step|^=== ep|^=== SWA|SERVE|ABORT|Error' | cut -c1-230 | while read -r l; do log "  smoke | $l"; done
  SPS=$(printf '%s\n' "$SL" | grep -oE ' [0-9.]+s/step' | sed 's#s/step##' | tail -n 2 | "$PYL" -c "import sys; v=sorted(float(x) for x in sys.stdin.read().split()); print(v[len(v)//2] if v else '')")
  EVS=$(printf '%s\n' "$SL" | grep -oE 'eval [0-9]+s' | head -n 1 | tr -dc '0-9'); EVS=${EVS:-30}
  NONF=$(printf '%s\n' "$SL" | grep -oE 'nonfinite_loss [0-9]+' | tail -n 1 | awk '{print $2}')
  MEM=$(printf '%s\n' "$SL" | grep -oE 'peak mem [0-9.]+' | tail -n 1 | awk '{print $3}')
  if [ "$SRC" != 0 ] || [ -z "$SPS" ] || [ "${NONF:-1}" != 0 ] || ! printf '%s' "$SL" | grep -q 'SERVE-LOAD OK'; then
    die "RENTAL SMOKE FAILED (rc $SRC, s/step '$SPS', nonfinite '$NONF'): the instance is still billing -> fix, or bash vast_destroy.sh --yes"
  fi
  SEED_S=$("$PYL" -c "print(int($SPS * 4349 * 16 + 16 * $EVS + 120))"); st_set SEED_S $SEED_S; st_set SMOKE_SPS $SPS; st_set SMOKE_OK 1
  printf 'DEADLINE_EPOCH=%s\nSEED_S=%s\nCL=%s\nWORKERS=%s\nEPOCHS=16\n' "$(st_get DEADLINE)" "$SEED_S" "$CL" 6 | ix "cat > $R/kit/plan.env"
  NOWS=$(date +%s); ROUNDS=$("$PYL" -c "import math; print(math.ceil($NS / min($NS, $(st_get NGPU))))")
  END=$((NOWS + ROUNDS * SEED_S + 1800)); SPEND=$("$PYL" -c "print(round($(st_get DPH) * ($END - $(st_get LAUNCH_EPOCH)) / 3600, 2))")
  log "RENTAL SMOKE OK: s/step $SPS (bf16), peak mem ${MEM} GB, gold eval ${EVS}s, nonfinite 0 | per seed $((SEED_S / 60)) min x $ROUNDS round(s) -> seeds done ~$(date -d @$((NOWS + ROUNDS * SEED_S)) '+%m-%d %H:%M') | projected total spend \$$SPEND | guard self-destroy $(date -d @$(st_get GUARD_DEADLINE) '+%m-%d %H:%M')"
fi

# ---- 6 lanes + box fetch loop + laptop backup watcher ----
ix "setsid -f bash $R/kit/lanes.sh '$SEEDS' > /dev/null 2>&1 < /dev/null"; log "lanes started: seeds [$SEEDS] on $(st_get NGPU) GPU(s)"
ssh "${BOX_SSH_OPTS[@]}" "$BOX" "mkdir -p ~/raptor2/vast && cat > ~/raptor2/vast/box_fetch.sh" < "$KIT/box_fetch.sh"
bx "setsid -f bash ~/raptor2/vast/box_fetch.sh $(st_get I_HOST) $(st_get I_PORT) $(st_get GUARD_DEADLINE) '$SEEDS' > /dev/null 2>&1 < /dev/null"
log "box fetch loop started (log ~/raptor_store/rented/fetch.log)"
( nohup bash "$KIT/vast_watch.sh" > "$LOGDIR/watch.out" 2>&1 < /dev/null & ) ; log "laptop backup watcher started"
sleep 900; log "first measured s/step: $(ix "grep -h 's/step' $R/runs/s*/train.log | tail -n $(st_get NGPU)" | cut -c1-160 | tr '\n' ' ')"
log "LAUNCH DONE"
