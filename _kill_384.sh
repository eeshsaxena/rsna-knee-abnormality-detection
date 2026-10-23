#!/bin/bash
pkill -9 -f run_384_auto
pkill -9 -f run_384_box
pkill -9 -f run_384_ensemble
pkill -9 -f "e2e_train.py"
sleep 3
echo "cleared. GPU:"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
