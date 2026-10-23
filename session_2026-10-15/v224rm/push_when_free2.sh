#!/usr/bin/env bash
# push_when_free2.sh KERNEL_DIR LOG: push one private commit only while fewer than 2 of our kernels are RUNNING / QUEUED; a push
# Kaggle refuses ("Maximum batch GPU session count") does not count: it polls every 20 s until "successfully pushed" (or 2 h).
D="$1"; LOG="$2"
count_running() {
  local n=0
  for k in $(kaggle kernels list --mine --sort-by dateRun --page-size 6 2>/dev/null | awk 'NR>2 && $1 ~ /\//{print $1}'); do
    s=$(kaggle kernels status "$k" 2>/dev/null | tail -1)
    case "$s" in *RUNNING*|*QUEUED*) n=$((n+1));; esac
  done
  echo $n
}
end=$(( $(date +%s) + 7200 ))
while [ "$(date +%s)" -lt "$end" ]; do
  n=$(count_running)
  echo "$(date -u +%H:%M:%S) running=$n" >> "$LOG"
  if [ "$n" -lt 2 ]; then
    out=$(kaggle kernels push -p "$D" 2>&1)
    echo "$(date -u +%H:%M:%S) PUSH try: $out" >> "$LOG"
    case "$out" in *"successfully pushed"*) echo "$(date -u +%H:%M:%S) PUSHED OK" >> "$LOG"; exit 0;; esac
  fi
  sleep 20
done
echo "$(date -u +%H:%M:%S) GAVE UP (2 h)" >> "$LOG"
exit 1
