#!/bin/bash
# poll eeshsaxena/rsna-v224f20j-check until it leaves RUNNING/QUEUED, then fetch outputs to $1
OUT=$1
for i in $(seq 1 200); do
  S=$(kaggle kernels status eeshsaxena/rsna-v224f20j-check 2>&1 | tail -1)
  echo "$(date +%H:%M:%S) $S"
  case "$S" in *RUNNING*|*QUEUED*|*running*|*queued*) sleep 45;; *) break;; esac
done
mkdir -p "$OUT" && kaggle kernels output eeshsaxena/rsna-v224f20j-check -p "$OUT" 2>&1 | tail -3
echo WAIT_DONE
