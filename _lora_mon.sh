#!/bin/bash
prev=""
for i in $(seq 1 45); do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 training@192.168.10.30 'grep -E "=== ep[0-9]|phase2|BEST" ~/rsna_train/lora_v2.log 2>/dev/null | tail -1; pgrep -f box_lora_v2 >/dev/null && echo _RUN || echo _DONE' 2>/dev/null)
  latest=$(echo "$out" | grep -E "ep|BEST" | tail -1)
  if [ "$latest" != "$prev" ] && [ -n "$latest" ]; then echo "[$(date +%H:%M)] $latest"; prev="$latest"; fi
  echo "$out" | grep -q _DONE && { echo "=== TRAINING FINISHED ==="; ssh -o BatchMode=yes training@192.168.10.30 'grep BEST ~/rsna_train/lora_v2.log | tail -1'; break; }
  sleep 300
done
