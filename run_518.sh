#!/bin/bash
cd 'C:\Users\eeshs\Downloads\rsna-knee'
PY312="/c/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe"
echo "[$(date +%H:%M)] waiting for exp518..."
for i in $(seq 1 90); do
  st=$($PY312 kaggle_do.py kstatus eeshsaxena/rsna-knee-exp518 2>&1 | grep -viE "requestsdep|warn" | tail -1)
  case "$st" in *COMPLETE*) echo "exp518 COMPLETE"; break;; *ERROR*|*CANCEL*) echo "exp518 FAILED: $st"; exit 1;; esac
  sleep 60
done
mkdir -p _exp518_out
$PY312 kaggle_do.py koutput eeshsaxena/rsna-knee-exp518 _exp518_out 2>&1 | grep -viE "requestsdep|warn" | tail -1
ls -la _exp518_out/*.npz 2>/dev/null
echo "=== 518px LOO ==="
$PY312 emb_res_loo.py _exp518_out/gold58_518.npz _dinov2_probe 2>&1 | grep -viE "Debugger|frozen|RequestsDep|warn|Convergence|n_iter|STOP|Increase|scale" | tail -3
echo "=== 392px LOO ==="
$PY312 emb_res_loo.py _exp518_out/gold58_392.npz _dinov2_probe 2>&1 | grep -viE "Debugger|frozen|RequestsDep|warn|Convergence|n_iter|STOP|Increase|scale" | tail -3
echo "[$(date +%H:%M)] 518 experiment DONE"
