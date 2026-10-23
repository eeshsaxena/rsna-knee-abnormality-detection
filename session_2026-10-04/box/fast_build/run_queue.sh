#!/bin/bash
# run_queue.sh QUEUEFILE -- runs bench.py invocations one after another (never two at once; bench.py also flocks).
# each line of QUEUEFILE: NAME|bench.py args
cd ~/fast_build
export TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=$HOME/fast_build/cache/inductor \
       TRITON_CACHE_DIR=$HOME/fast_build/cache/triton HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/rsna/bin/python
while IFS='|' read -r NAME ARGS; do
  [ -z "$NAME" ] && continue; case "$NAME" in \#*) continue;; esac
  echo "=== $(date +%T) start $NAME :: $ARGS" >> logs/queue.log
  timeout -k 20 380 nice -n 10 $PY bench.py $ARGS --tag "$NAME" > "logs/$NAME.log" 2>&1
  echo "=== $(date +%T) end $NAME rc=$?" >> logs/queue.log
  sleep 5
done < "$1"
echo "=== $(date +%T) QUEUE DONE $1" >> logs/queue.log
