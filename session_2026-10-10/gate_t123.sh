#!/bin/bash
# gate_t123.sh (2026-10-10, Claude): CPU gate for the T123v00 probe (PROBE3_PREREG.md). Called ONCE by the queue runner
# when the first T123 fold line reaches the head (`needs PASS from ~/oai_build/out/t123.verdict wait 30 via ...`).
# 1) score_oai_probe2.py --tier T12 --report_only on the probe vs control D f0 (writes gate2_REPORT_T12_<run>.json,
#    runs gold_infer5 pass A on CPU for the probe's gold74 preds); 2) gate_t123.py applies the frozen bars and writes
#    ~/oai_build/out/t123.verdict. Never touches the GPU.
set -u
cd ~/oai_build || exit 1
RUN=~/fleet_runs/f0_s1043_ttr_frac75oaiT123v00
tail -n 1 $RUN.log 2>/dev/null | grep -q '^{"raw"' || { echo "probe not finished"; exit 3; }
CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python score_oai_probe2.py --tier T12 --report_only \
  --run $RUN --expect_labels labels_frac75oai_T123v00.csv > out/score_t123_$(date +%H%M).log 2>&1
echo "score_oai_probe2 rc $?"
~/miniconda3/envs/rsna/bin/python gate_t123.py
