#!/bin/bash
# laptop CPU test suite for bigjump B1-B5 (64 px synthetic assets, convnext_atto; ~11 min). Writes tests/transcript.txt
cd "$(dirname "$0")"
PY=C:/Users/eeshs/AppData/Local/Programs/Python/Python312/python.exe
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=
: > transcript.txt
for t in t_b1_model t_b2_default_parity t_b2_heavy_equiv t_b2_ttr_train t_b4_pack_tag t_b5_tta_bench t_gt2_saveep; do
  s=$(date +%s)
  echo "=== $t $(date +%H:%M:%S)" >> transcript.txt
  $PY -W ignore $t.py > out/$t.log 2>&1; rc=$?
  grep -E '^[(]|^[[]|^T_|OK$|PASS|FAIL|Error|^model_sha|^fleet_(common|model)' out/$t.log | grep -v '^ ' | cut -c1-260 >> transcript.txt
  echo "rc $rc ($(( $(date +%s) - s )) s)" >> transcript.txt
done
grep -E "^=== |^rc |OK$|OK:" transcript.txt
