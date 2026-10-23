#!/bin/bash
# start_lanes.sh "LANESPEC" (instance; called over ssh by the box driver). LANESPEC = space separated "G:F1,F2" items, G = a
# GPU index or "any". Starts one detached rstud_lane.sh per item (lane ids continue from status/lane_seq) and, once, the GPU
# monitor (nvidia-smi every 30 s -> logs/gpu_mon.csv). Prints "lane <id> <spec> pid <pid>" per item.
set -u
R=${RSNA_ROOT:-/workspace/rsna}; S=$R/status; mkdir -p $S $R/logs
[ -e $S/setup.ok ] || { echo "setup not OK: refusing to start lanes"; exit 1; }
n=$(cat $S/lane_seq 2>/dev/null || echo 0)
for it in $1; do
  g=${it%%:*}; f=${it#*:}
  ( cd $R && exec setsid nohup bash $R/kit/rstud_lane.sh $n "$g" "$f" ) >> $R/logs/lane_$n.out 2>&1 < /dev/null &
  sleep 1; echo "lane $n $it pid $(cat $S/lane_$n.pid 2>/dev/null || echo ?)"; n=$(( n + 1 ))
done
echo $n > $S/lane_seq
if [ "${MOCK_CPU:-0}" != 1 ] && ! { [ -s $S/gpumon.pid ] && kill -0 "$(cat $S/gpumon.pid)" 2>/dev/null; }; then
  ( exec setsid nohup nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 30 ) >> $R/logs/gpu_mon.csv 2>/dev/null < /dev/null &
  echo $! > $S/gpumon.pid; echo "gpu monitor pid $!"
fi
