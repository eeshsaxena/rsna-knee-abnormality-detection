#!/bin/bash
# k4_rehearsal5.sh (box, CPU, nice 19; read-only on ~/fleet_runs; writes only ~/k4_test/reh5 + ~/k4_test/runs):
# a 5-TTR-member rehearsal of the BUILT K4-main notebook (strict W .20) before folds 1-4 exist. The 5 members are real
# TTR checkpoints that exist today (J1 raw + ema, TTR v4 raw + ema, J1 epoch-20 ema snapshot): NOT a shippable set
# (two tags of one run), only a test that the K3 cell + bigjump worker serve 5 TTR members with a 5-name member filter,
# (the copies are RE-KEYED f0..f4_s1043_<tag>_ttr224-320x24_reh<i>_best.pt, the final K4 name pattern: the worker
# refuses two members with one f<fold>_s<seed> prefix, which is exactly the one-tag-per-run rule),
# a 5-entry manifest 'sources', per-member OOF parity, the mean-of-OOFs parity and the npz-mean identity.
#   bash ~/k4_test/k4/box/k4_rehearsal5.sh      -> runs/REH5_main_<HHMM>.log  (BOXSTAGE PASS n/n)
set -euo pipefail
PYK=~/miniconda3/envs/kmimic/bin/python
R=~/k4_test/reh5; J1=~/fleet_runs/f0_s1043_ttr_mix50; TV=~/fleet_runs/f0_s1042_ttr
rm -rf $R; mkdir -p $R/ds $R/kernel
cd ~/k4_test/ds/rsna-knee-cnxt-fleet && cp -p fleet_*.py dataset-metadata.json parity_*.npz $R/ds/
M=( $(ls $J1/f0_s1043_raw_*_best.pt) $(ls $J1/f0_s1043_ema_*_best.pt) $(ls $TV/f0_s1042_raw_*_best.pt) $(ls $TV/f0_s1042_ema_*_best.pt) $J1/epochs/f0_s1043_ema_ttr224-320x24_ep20_best.pt )
O=( $J1/oof_f0_s1043_raw.csv $J1/oof_f0_s1043_ema.csv $TV/oof_f0_s1042_raw.csv $TV/oof_f0_s1042_ema.csv $J1/epochs/oof_f0_s1043_ema_ep20.csv )
ARGS=""
N=()
for i in 0 1 2 3 4; do tag=$(basename ${M[$i]} | cut -d_ -f3); n=f${i}_s1043_${tag}_ttr224-320x24_reh${i}_best.pt; N+=($n)
  cp -p ${M[$i]} $R/ds/$n; ARGS="$ARGS --oof_member $n=${O[$i]}"; done
CUDA_VISIBLE_DEVICES= $PYK - $R "${M[@]}" <<'EOF'
import hashlib, json, sys, os, torch
R, paths = sys.argv[1], sys.argv[2:]
names = [f'f{i}_s1043_' + os.path.basename(p).split('_')[2] + f'_ttr224-320x24_reh{i}_best.pt' for i, p in enumerate(paths)]
src = {}
for i, p in enumerate(paths):
    a = torch.load(p, map_location='cpu', weights_only=False)['args']
    src[names[i]] = dict(original=os.path.basename(p), run_dir='members',
                                    sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16],
                                    model_mod=a.get('model_mod'), model_sha=a.get('model_sha'))
json.dump({'checkpoints': names, 'sources': src}, open(f'{R}/ds/fleet_manifest.json', 'w'), indent=1)
nb = json.load(open(os.path.expanduser('~/k4_test/kernel/rsna_sub_k4_fleet.ipynb')))
src31 = ''.join(nb['cells'][31]['source']); hdr, body = src31.split('\n', 1)
cfg = eval(hdr.split('=', 1)[1].split('   #')[0])
assert cfg['MAIN_MODE'] == 'strict' and abs(cfg['W_MAIN'] - 0.2) < 1e-9, cfg
cfg['MEMBERS'] = names
nb['cells'][31]['source'] = (f'K2_CFG = {cfg!r}   # REHEARSAL (k4_rehearsal5.sh): 5 TTR members\n' + body).splitlines(keepends=True)
json.dump(nb, open(f'{R}/kernel/reh5_main.ipynb', 'w'), indent=1)
print('manifest', names); print('K2_CFG', cfg)
EOF
cd ~/k4_test
T=$(date +%H%M)
CUDA_VISIBLE_DEVICES= nohup setsid nice -n 19 $PYK -W ignore k4/box/k2_box_stage.py --cpu --nb $R/kernel/reh5_main.ipynb \
  --ids ids/a0.csv --fleet_ds $R/ds --age_h 7.9 --threads 4 $ARGS --out runs/REH5_main_$T > runs/REH5_main_$T.log 2>&1 < /dev/null &
echo "started REH5_main pid $! log ~/k4_test/runs/REH5_main_$T.log"
