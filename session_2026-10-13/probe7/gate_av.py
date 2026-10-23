#!/usr/bin/env python
# gate_av.py (2026-10-13, Claude): PROBE7_PREREG.md gate for the T123allv probe (box CPU). Exit 0 PASS, 1 FAIL, 2 integrity, 3 not ready.
import os, sys, glob, hashlib, subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
H = os.path.expanduser
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
SHA = open(H('~/oai_build/PROBE7_PREREG.sha256')).read().split()[0]
if hashlib.sha256(open(H('~/oai_build/PROBE7_PREREG.md'), 'rb').read()).hexdigest() != SHA:
    print('INTEGRITY: PROBE7_PREREG.md changed'); sys.exit(2)
RUN = H('~/fleet_runs/f0_s1043_ttr_frac75oaiT123allv'); DT = H('~/fleet_runs/f0_s1043_ttr_frac75oaiT123v00')
GP_AV = H('~/oai_build/gold74_preds/f0_s1043_ttr_frac75oaiT123allv__raw__A.csv')
if not os.path.exists(RUN + '.log') or not open(RUN + '.log', errors='replace').read().rstrip().splitlines()[-1].startswith('{"raw"'):
    print('NOT READY'); sys.exit(3)
import torch
ck = sorted(glob.glob(RUN + '/f0_s1043_raw_*_best.pt'))
if len(ck) != 1: print('NOT READY ckpt', ck); sys.exit(3)
a = torch.load(ck[0], map_location='cpu', weights_only=False)['args']
if a.get('train_labels') != 'labels_frac75oai_T123allv.csv': print('INTEGRITY: train_labels', a.get('train_labels')); sys.exit(2)
if not os.path.exists(GP_AV):
    r = subprocess.run(['nice', '-n', '19', H('~/miniconda3/envs/rsna/bin/python'), H('~/novel/gold_infer5.py'), '--passes', 'A',
                        '--out', H('~/oai_build/gold74_preds'), ck[0]], env=dict(os.environ, NTHR='6', OMP_NUM_THREADS='6'),
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(GP_AV): print('INTEGRITY: gold_infer5 failed', (r.stdout + r.stderr)[-500:]); sys.exit(2)
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = rd(H('~/rsna_train/llm_labels_v4_blend.csv')); gold = rd(H('~/rsna_train/gold74_eval_set.csv'))
def auc(y, p):
    ok = np.isfinite(y) & np.isfinite(p); yb = y[ok] >= .5
    return roc_auc_score(yb, p[ok]) if 0 < yb.mean() < 1 else np.nan
oa, od = rd(RUN + '/oof_f0_s1043_raw.csv'), rd(DT + '/oof_f0_s1043_raw.csv')
ids = sorted(od.index)
if sorted(oa.index) != ids or oa.index.str.startswith('oai_').any(): print('INTEGRITY: OOF ids'); sys.exit(2)
Y = v4.loc[ids, T].to_numpy(float)
per = lambda o: np.array([auc(Y[:, j], o.loc[ids, T].to_numpy(float)[:, j]) for j in range(12)])
pa, pd_ = per(oa), per(od)
g58 = gold[gold.source == 'original_gold58'].index; YG = gold.loc[g58, T].to_numpy(float)
G = rd(GP_AV).reindex(g58)[T].astype(float).to_numpy()
gav = float(np.nanmean([auc(YG[:, j], G[:, j]) for j in range(12)]))
g1, g2 = pa.mean() >= 0.889160, gav >= 0.914434
lines = [f'gate_av | prereg sha {SHA[:8]}', f'  G1 OOF12(AV) {pa.mean():.6f} >= .889160 (DT f0 {pd_.mean():.6f}) -> {"PASS" if g1 else "FAIL"}',
         f'  G2 gold58(AV) {gav:.6f} >= .914434 -> {"PASS" if g2 else "FAIL"}',
         '  per-label OOF AV - DT: ' + ' '.join(f'{T[j]} {pa[j] - pd_[j]:+.4f}' for j in range(12)),
         f'AV GATE: {"PASS" if g1 and g2 else "FAIL"}']
print('\n'.join(lines)); open(H('~/oai_build/out/gate_av.txt'), 'w').write('\n'.join(lines) + '\n')
tmp = H('~/oai_build/out/av.verdict.tmp'); open(tmp, 'w').write(('PASS' if g1 and g2 else 'FAIL') + '\n'); os.replace(tmp, H('~/oai_build/out/av.verdict'))
sys.exit(0 if g1 and g2 else 1)
