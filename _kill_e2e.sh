#!/bin/bash
pkill -9 -f run_e2e_after
pkill -9 -f "e2e_train.py"
sleep 2
echo "ALL E2E DONE (stopped - e2e 224px inferior, prioritizing 384px)" >> ~/rsna_train/e2e_after.log
echo "killed. GPU:"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
