#!/bin/bash
# vast_train.sh [FOLDS..] [--dry-run]: start the TTR320 frac75 folds on the instance, one fleet_train.py process per
# GPU (CUDA_VISIBLE_DEVICES=G), exact box argv (remote/argv_template.txt, paths under /workspace/rsna). With more
# folds than GPUs, GPU g runs folds g, g+N, .. one after another ("lanes", remote/lane.sh, setsid nohup, survives
# ssh drops). Logs: /workspace/rsna/fleet_runs/fF_s1045_<RUNTAG>.log (trainer) and logs/lane_G.log.
# Idempotent: a lane that is alive is left alone; finished folds are skipped by lane.sh. Default FOLDS = state.env.
# The folds must have box placeholders (vast_launch.sh made them), so the box queue never trains them too.
SCRIPT_TAG=train; source "$(dirname "$0")/config.sh"
DRY=0; ARGS=()
for a in "$@"; do case $a in --dry-run) DRY=1;; [0-4]) ARGS+=("$a");; *) die "unknown argument $a";; esac; done
SF=$(st_get FOLDS); FOLDS=${ARGS[*]:-${SF:-0}}
for F in $FOLDS; do case " ${SF:-0} " in *" $F "*) ;; *) die "f$F is not in the Vast plan [${SF:-}] (no box placeholder)";; esac; done
IID=$(st_get INSTANCE_ID); NG=$(st_get NGPU); NG=${NG:-2}
set -- $FOLDS; NF=$#; LANES=$(( NF < NG ? NF : NG ))
declare -a LF; i=0; for F in $FOLDS; do LF[$((i % LANES))]="${LF[$((i % LANES))]:-} $F"; i=$((i + 1)); done
log "== vast_train | instance ${IID:-none} | folds [$FOLDS] on $LANES GPU lane(s):"
for g in $(seq 0 $((LANES - 1))); do log "   GPU $g -> folds [$(echo ${LF[$g]})]"; done
log "   argv (fold F): python fleet_train.py $(sed -e "s#@R@#$R#g" "$KIT/remote/argv_template.txt")"
if [ $DRY = 1 ] || [ -z "$IID" ]; then log "DRY: no instance touched"; exit 0; fi
[ "$(st_get SETUP_OK)" = 1 ] && ix "test -e $R/status/SETUP_OK" || die "setup not complete: run vast_setup.sh first"
for F in $FOLDS; do bops ph-status "$F" | grep -q 'EMPTY placeholder\|files)' || die "box placeholder for f$F is missing: the box could train it too"; done
ix "sed -i 's/^FOLDS=.*/FOLDS=\"$FOLDS\"/' $R/kit/plan.env"
for g in $(seq 0 $((LANES - 1))); do
  r=$(ix "p=\$(cat $R/status/lane_$g.pid 2>/dev/null); if [ -n \"\$p\" ] && kill -0 \$p 2>/dev/null; then echo alive \$p; else
         ( cd $R && exec setsid nohup bash $R/kit/lane.sh $g '$(echo ${LF[$g]})' ) >> $R/logs/lane_$g.out 2>&1 < /dev/null & sleep 2; echo started \$(cat $R/status/lane_$g.pid 2>/dev/null); fi" < /dev/null)
  log "lane GPU $g: $r"
done
[ -n "$(st_get TRAIN_STARTED)" ] || st_set TRAIN_STARTED "$(date +%s)"
log "waiting 3 min for the first iterations..."; sleep 180
ix "bash $R/kit/status.sh" | sed 's/^/  /'
ix "tail -n 3 $R/logs/lane_*.log" | sed 's/^/  /'
log "TRAINING STARTED. vast_watch.sh polls every 10 min, fetches and destroys at the end."
