#!/bin/bash
# sv_watch.sh "43 44 45 46" (instance, detached): as soon as a seed's lane writes status/s<seed>.rc = 0, score its swa.pt on gold58
# in the SERVING geometry (serve_check.py: 320 crop, 94 windows, mirror, fp16) on that seed's own GPU (lane i = GPU i), write
# runs/sv_s<seed>/ (+ SHA256SUMS) and status/sv_s<seed>.done (also written, without scoring, when the seed failed).
R=/workspace/rsna; K=$R/kit; S=$R/status; SEEDS=($1); PY=$(command -v python); NG=$(nvidia-smi -L | wc -l)
while :; do
  left=0
  for i in "${!SEEDS[@]}"; do s=${SEEDS[$i]}
    [ -e $S/sv_s$s.done ] && continue
    if [ -e $S/s$s.rc ]; then
      if [ "$(cat $S/s$s.rc)" = 0 ] && [ -f $R/runs/s$s/swa.pt ]; then
        ( cd $K && CUDA_VISIBLE_DEVICES=$(( i % NG )) HF_HUB_OFFLINE=1 $PY serve_check.py --index $R/rsna_index.csv --ck $R/runs/s$s/swa.pt \
            --gold $K/gold74_eval_set.csv --out $R/runs/sv_s$s ) > $R/runs/sv_s$s.log 2>&1
        echo $? > $S/sv_s$s.rc; ( cd $R/runs/sv_s$s 2>/dev/null && sha256sum *.npy > SHA256SUMS )
      fi
      touch $S/sv_s$s.done
    else left=$((left + 1)); fi
  done
  [ $left = 0 ] && break; sleep 60
done
