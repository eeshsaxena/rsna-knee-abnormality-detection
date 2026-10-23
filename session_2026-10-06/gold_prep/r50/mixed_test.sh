#!/bin/bash
# mixed_test.sh (gold_prep/r50, CPU, nice 19): the K7 pack tool + K5/K7 worker on a MIXED manifest = K5 TTR member f0_s1043
# (read-only from ~/k5_test/ckpt) + the R50 dry-run raw checkpoint (copy with args amp fp32 -> fp16, nothing else changed).
# Proves: pack_fleet_ds.py accepts an f2 member next to TTR, keys do not collide, both families load + serve (TTA AB) together,
# decode probes of both match the shipped parity npz, and the R50 pass-A rows still equal its trainer OOF.
G=$HOME/gold_prep/r50; PYK=~/miniconda3/envs/kmimic/bin/python; M=$G/mixed
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
rm -rf $M; mkdir -p $M/in; : > $G/mixed_test.log
nice -n 19 $PYK -W ignore - >> $G/mixed_test.log 2>&1 <<'PY'
import glob, os, torch
G = os.path.expanduser('~/gold_prep/r50'); p = glob.glob(f'{G}/dryA/f0_s243_raw_*_best.pt')[0]
ck = torch.load(p, map_location='cpu', weights_only=False); ck['args']['amp'] = 'fp16'
torch.save(ck, f'{G}/mixed/in/{os.path.basename(p)}'); print('patched copy', os.path.basename(p))
PY
R50=$(ls $M/in/f0_s243_raw_*_best.pt)
nice -n 19 $PYK -W ignore $G/k5code/pack_fleet_ds.py --out $M/ds --store $HOME/fleet_store --tta AB,AB \
  $HOME/k5_test/ckpt/f0_s1043_raw_ttr224-320x24_cv8835_best.pt $R50 >> $G/mixed_test.log 2>&1; echo "PACK rc=$?" >> $G/mixed_test.log
(cd $M && FLEET_ALLOW_CPU=1 FLEET_DIR=$M/ds FLEET_OUT=$M/out.npz RSNA_COMP_ROOT=$G/wk/comp \
  nice -n 19 $PYK -W ignore $M/ds/fleet_worker.py >> $G/mixed_test.log 2>&1); echo "WORKER rc=$?" >> $G/mixed_test.log
nice -n 19 $PYK -W ignore - >> $G/mixed_test.log 2>&1 <<'PY'
import os, numpy as np, pandas as pd
G = os.path.expanduser('~/gold_prep/r50')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
z = np.load(f'{G}/mixed/out.npz'); sids = [str(s) for s in z['sids']]
oofs = {'f0_s243': f'{G}/dryA/oof_f0_s243_raw.csv', 'f0_s1043': os.path.expanduser('~/fleet_runs/f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv')}
print('parity', str(z['parity'])[:300])
for k, n in enumerate(z['names']):
    key = '_'.join(str(n).split('_')[:2]); o = pd.read_csv(oofs[key], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    O = o.loc[sids, T].to_numpy(np.float32); A = z['PA'][k]
    print(f'{n}: |PA - trainer OOF| max {np.abs(A - O).max():.2e} | finite {np.isfinite(z["P"][k]).all()}')
PY
echo "MIXED DONE" >> $G/mixed_test.log
