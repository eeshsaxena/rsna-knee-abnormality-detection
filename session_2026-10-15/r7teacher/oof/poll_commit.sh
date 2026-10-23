#!/bin/bash
# poll eeshsaxena/rsna-r7-oof until it leaves RUNNING/QUEUED, then download its output (read-only)
K=eeshsaxena/rsna-r7-oof
OUT=${1:-out_commit_v1}
for i in $(seq 1 80); do
  S=$(kaggle kernels status $K 2>&1 | tail -1)
  echo "$(date -u +%H:%M:%S) $S" >> poll_commit.log
  case "$S" in *RUNNING*|*QUEUED*) sleep 45;; *) break;; esac
done
mkdir -p $OUT && kaggle kernels output $K -p $OUT >> poll_commit.log 2>&1
echo "DONE $(date -u +%H:%M:%S) $S" >> poll_commit.log
