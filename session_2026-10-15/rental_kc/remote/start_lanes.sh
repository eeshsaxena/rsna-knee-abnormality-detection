#!/bin/bash
# start_lanes.sh STAGE "LANESPEC" (rental_kc instance; called over ssh by the box driver). STAGE smoke | real. LANESPEC = space
# separated "G:KEY" items (one member per lane, G = GPU index). Starts one detached kc_lane.sh per item (lane ids continue
# from status/lane_seq) and, once, the GPU monitor (nvidia-smi every 30 s -> logs/gpu_mon.csv). Prints "lane <id> <G:KEY>
# pid <pid>" per item.
set -u
R=${RSNA_ROOT:-/workspace/rsna}; S=$R/status; mkdir -p $S $R/logs
[ -e $S/setup.ok ] || { echo "setup not OK: refusing to start lanes"; exit 1; }
STAGE=$1; case $STAGE in smoke|real) ;; *) echo "STAGE must be smoke or real"; exit 1;; esac
n=$(cat $S/lane_seq 2>/dev/null || echo 0)
for it in $2; do
  g=${it%%:*}; k=${it#*:}
  ( cd $R && exec setsid nohup bash $R/kit/kc_lane.sh $n "$g" "$k" $STAGE ) >> $R/logs/lane_$n.out 2>&1 < /dev/null &
  sleep 1; echo "lane $n $it $STAGE pid $(cat $S/lane_$n.pid 2>/dev/null || echo ?)"; n=$(( n + 1 ))
done
echo $n > $S/lane_seq
if [ "${MOCK_CPU:-0}" != 1 ] && ! { [ -s $S/gpumon.pid ] && kill -0 "$(cat $S/gpumon.pid)" 2>/dev/null; }; then
  ( exec setsid nohup nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 30 ) >> $R/logs/gpu_mon.csv 2>/dev/null < /dev/null &
  echo $! > $S/gpumon.pid; echo "gpu monitor pid $!"
fi
