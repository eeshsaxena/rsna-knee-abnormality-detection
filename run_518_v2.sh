#!/bin/bash
cd 'C:\Users\eeshs\Downloads\rsna-knee'
PY312="/c/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe"
echo "[$(date +%H:%M)] waiting for exp518 v2..."
for i in $(seq 1 45); do
  st=$($PY312 kaggle_do.py kstatus eeshsaxena/rsna-knee-exp518 2>&1 | grep -viE "requestsdep|warn" | tail -1)
  case "$st" in *COMPLETE*) echo "COMPLETE"; break;; *ERROR*|*CANCEL*) echo "FAILED: $st"; exit 1;; esac
  sleep 60
done
rm -rf _exp518_out; mkdir -p _exp518_out
$PY312 kaggle_do.py koutput eeshsaxena/rsna-knee-exp518 _exp518_out 2>&1 | grep -viE "requestsdep|warn" | tail -1
ls -la _exp518_out/*.npz 2>/dev/null
for RES in 518 392; do
  echo "=== ${RES}px LOO ==="
  $PY312 emb_res_loo.py _exp518_out/gold58_${RES}.npz _dinov2_probe 2>&1 | grep -viE "Debugger|frozen|RequestsDep|warn|Convergence|n_iter|STOP|Increase|scale" | tail -14
done
echo "[$(date +%H:%M)] 518 EXPERIMENT DONE"
