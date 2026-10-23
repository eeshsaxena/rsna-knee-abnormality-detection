#!/bin/bash
# poll eeshsaxena/rsna-jpeg-check until it leaves RUNNING/QUEUED, then fetch outputs to $1
OUT=$1
for i in $(seq 1 240); do
  S=$(kaggle kernels status eeshsaxena/rsna-jpeg-check 2>&1 | tail -1)
  echo "$(date +%H:%M:%S) $S"
  case "$S" in *RUNNING*|*QUEUED*|*running*|*queued*) sleep 30;; *) break;; esac
done
mkdir -p "$OUT" && kaggle kernels output eeshsaxena/rsna-jpeg-check -p "$OUT" 2>&1 | tail -3
echo WAIT_DONE
