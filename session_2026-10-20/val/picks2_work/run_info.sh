#!/bin/sh
# run_info.sh: INFO-only rows (pair with A = pw20 / eoa20 instead of the R1 best-id) on the rows most likely to flip it. picks2_mc.py, rep 2.
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/val/picks2_work
: > info.log
for a in "tag=info_mix family=mix" "tag=info_ALL abc=PFT_N01_logN_ALLruler" "tag=info_PAREA abc=PAREA_N01_logN" "tag=info_flatU abc=PFT_flat_logU" \
         "tag=info_shape2 shape_sd=2e-4" "tag=info_kpess k_mu=0.4 k_sd=0.4" "tag=info_noise0.7 noise=0.7" "tag=info_skepS6 our=skepS6"; do
  python picks2_mc.py info_a=1 $a >> info.log 2>&1
done
echo INFO_DONE >> info.log
