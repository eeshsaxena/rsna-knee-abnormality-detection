#!/bin/bash
# lanes.sh "43 44 45" (instance, detached): one lane per GPU, seeds spread round-robin, each lane runs its seeds one after
# another. Each seed = train_raptor.py, the box recipe minus OAI (RSNA rows only, --forbid_oai 1), bf16 (Ampere / Ada),
# 16 epochs, out $R/runs/s<seed>. A seed is only STARTED if it can finish before plan.env DEADLINE_EPOCH (estimate =
# measured duration of the lane's previous seed, else SEED_S from plan.env). On exit: status/s<seed>.rc + runs/s<seed>/SHA256SUMS.
set -u
R=/workspace/rsna; K=$R/kit; S=$R/status; . $K/plan.env          # DEADLINE_EPOCH SEED_S CL WORKERS EPOCHS
SEEDS=($1); NG=$(nvidia-smi -L | wc -l); PY=$(command -v python)
[ -f $S/setup.ok ] || { echo "setup not OK" > $S/lanes.failed; exit 1; }
lane(){
  local g=$1; shift; local prev="" s out t0 rc dt est
  for s in "$@"; do
    out=$R/runs/s$s
    [ -f $S/s$s.rc ] && [ "$(cat $S/s$s.rc)" = 0 ] && continue
    est=${prev:-$SEED_S}
    if [ $(( $(date +%s) + est * 105 / 100 )) -gt $DEADLINE_EPOCH ]; then echo "skipped: est $(( est / 60 )) min > time left" > $S/s$s.skipped; continue; fi
    mkdir -p $out; date +%s > $S/s$s.running; t0=$(date +%s)
    ( cd $K && exec env CUDA_VISIBLE_DEVICES=$g HF_HOME=$R/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 $PY train_raptor.py \
        --index $R/rsna_index.csv --labels_rsna $K/labels_raptor_rsna_blendjev_sol.csv --gold $K/gold74_eval_set.csv \
        --out $out --seed $s --epochs $EPOCHS --amp bf16 --channels_last $CL --workers $WORKERS --forbid_oai 1 ) > $out.out 2>&1 < /dev/null
    rc=$?; dt=$(( $(date +%s) - t0 )); rm -f $S/s$s.running
    ( cd $out && sha256sum * > SHA256SUMS.tmp 2>/dev/null; mv SHA256SUMS.tmp SHA256SUMS )
    echo $rc > $S/s$s.rc; echo "seed $s gpu $g rc $rc in $(( dt / 60 )) min" >> $R/lanes.log
    [ $rc = 0 ] && prev=$dt
  done
}
echo "[$(date -u '+%m-%d %H:%M')Z] lanes start: seeds ${SEEDS[*]} on $NG GPU(s), deadline $(date -u -d @$DEADLINE_EPOCH '+%m-%d %H:%M')Z" >> $R/lanes.log
for g in $(seq 0 $(( NG - 1 ))); do
  mine=(); for i in "${!SEEDS[@]}"; do [ $(( i % NG )) = $g ] && mine+=(${SEEDS[$i]}); done
  [ ${#mine[@]} -gt 0 ] && lane $g "${mine[@]}" &
done
wait
echo "[$(date -u '+%m-%d %H:%M')Z] lanes end" >> $R/lanes.log; date +%s > $S/lanes.done
