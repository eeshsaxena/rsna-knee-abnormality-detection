#!/bin/bash
# test of k3_pick_ttr.py's FINAL path on the real TTR best files (symlinked, read-only) with a fake closing log.
# Writes only /tmp/k3pick. usage: bash t_pick_ttr.sh
set -u
PYK=~/miniconda3/envs/kmimic/bin/python
R=~/fleet_runs/f0_s1042_ttr
D=/tmp/k3pick
rm -rf $D/run $D/dest; mkdir -p $D/run $D/dest
for f in $R/f0_s1042_*_best.pt $R/oof_f0_s1042_*.csv; do ln -s $f $D/run/; done
CUDA_VISIBLE_DEVICES= $PYK -W ignore - > $D/run.log <<'EOF'
import glob, json, torch
best = {}
for tag in ('raw', 'ema'):
    f = glob.glob(f'/tmp/k3pick/run/f0_s1042_{tag}_*_best.pt')[0]
    best[tag] = float(torch.load(f, map_location='cpu', weights_only=False)['cv'])
print('ep 23 raw | fake'); print('ep 23 ema | fake'); print(json.dumps(best))
EOF
CUDA_VISIBLE_DEVICES= nice -n 19 $PYK -W ignore ~/k3_test/k3/box/k3_pick_ttr.py --run $D/run --log $D/run.log --copy --dest $D/dest
echo "EXIT $?"
ls --time-style=full-iso -l $D/dest/ckpt $D/dest/oof
