#!/usr/bin/env python
# gate_rr.py (2026-10-13, Claude): PROBE6_PREREG.md gate for the RadImageNet ResNet-50 T123 probe (box CPU only).
# Exit 0 PASS, 1 FAIL, 2 integrity, 3 not ready. --selftest: runs the arithmetic with D f0 in place of RR (no verdict).
import os, sys, json, glob, hashlib, argparse, subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
H = os.path.expanduser
ap = argparse.ArgumentParser(); ap.add_argument('--selftest', action='store_true'); A = ap.parse_args()
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
PREREG_SHA = open(H('~/oai_build/PROBE6_PREREG.sha256')).read().split()[0]
if hashlib.sha256(open(H('~/oai_build/PROBE6_PREREG.md'), 'rb').read()).hexdigest() != PREREG_SHA:
    print('INTEGRITY: PROBE6_PREREG.md changed'); sys.exit(2)
RUNS = dict(DT=H('~/fleet_runs/f0_s1043_ttr_frac75oaiT123v00'), D=H('~/fleet_runs/f0_s1043_ttr_frac75'),
            RR=H('~/fleet_runs/f0_s1043_r50rad_frac75oaiT123'))
GP = dict(DT=H('~/oai_build/gold74_preds/f0_s1043_ttr_frac75oaiT123v00__raw__A.csv'),
          D=H('~/novel/frac75/gold74_preds/f0_s1043_ttr_frac75__raw__A.csv'),
          RR=H('~/oai_build/gold74_preds/f0_s1043_r50rad_frac75oaiT123__raw__A.csv'))
if A.selftest:
    RUNS['RR'], GP['RR'] = RUNS['D'], GP['D']
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = rd(H('~/rsna_train/llm_labels_v4_blend.csv')); gold = rd(H('~/rsna_train/gold74_eval_set.csv'))
log = RUNS['RR'] + '.log'
if not A.selftest:
    if not os.path.exists(log) or not open(log, errors='replace').read().rstrip().splitlines()[-1].startswith('{"raw"'):
        print('NOT READY: RR probe not finished'); sys.exit(3)
    import torch
    ck = sorted(glob.glob(RUNS['RR'] + '/f0_s1043_raw_*_best.pt'))
    if len(ck) != 1: print('NOT READY: RR best ckpt', ck); sys.exit(3)
    a = torch.load(ck[0], map_location='cpu', weights_only=False)['args']
    if a.get('backbone') != 'resnet50.a1_in1k' or 'radimagenet' not in str(a.get('init_ckpt', '')) or a.get('train_labels') != 'labels_frac75oai_T123v00.csv':
        print('INTEGRITY: RR ckpt args', a.get('backbone'), a.get('train_labels')); sys.exit(2)
    if not os.path.exists(GP['RR']):
        r = subprocess.run(['nice', '-n', '19', H('~/miniconda3/envs/rsna/bin/python'), H('~/novel/gold_infer5.py'), '--passes', 'A',
                            '--out', H('~/oai_build/gold74_preds'), ck[0]], env=dict(os.environ, NTHR='6', OMP_NUM_THREADS='6'),
                           capture_output=True, text=True)
        print('\n'.join((r.stdout + r.stderr).splitlines()[-6:]))
        if r.returncode != 0 or not os.path.exists(GP['RR']): print('INTEGRITY: gold_infer5 failed'); sys.exit(2)
O = {k: rd(f'{v}/oof_f0_s1043_raw.csv')[T].astype(float) for k, v in RUNS.items()}
ids = sorted(O['DT'].index)
for k in O:
    if sorted(O[k].index) != ids or O[k].index.str.startswith('oai_').any(): print('INTEGRITY: OOF ids', k); sys.exit(2)
    O[k] = O[k].loc[ids]
Y = v4.loc[ids, T].to_numpy(float)


def auc_m(y, p):
    ok = np.isfinite(y) & np.isfinite(p); yb = y[ok] >= 0.5
    return roc_auc_score(yb, p[ok]) if ok.sum() and 0 < yb.mean() < 1 else np.nan


m12 = lambda P, Yy: float(np.nanmean([auc_m(Yy[:, j], P[:, j]) for j in range(12)]))
oof = {k: m12(O[k].to_numpy(), Y) for k in O}
oof_dtrr = m12((O['DT'].to_numpy() + O['RR'].to_numpy()) / 2, Y); oof_dtd = m12((O['DT'].to_numpy() + O['D'].to_numpy()) / 2, Y)
g58 = gold[gold.source == 'original_gold58'].index
G = {k: rd(p).reindex(g58)[T].astype(float).to_numpy() for k, p in GP.items()}
YG = gold.loc[g58, T].to_numpy(float)
g_dtrr = m12((G['DT'] + G['RR']) / 2, YG); g_dtd = m12((G['DT'] + G['D']) / 2, YG)
sp = lambda a, b: float(np.mean([pd.Series(O[a][c].values).corr(pd.Series(O[b][c].values), method='spearman') for c in T]))
res = dict(G1=(oof['RR'], oof['RR'] >= 0.881660), G2=(oof_dtrr - oof_dtd, oof_dtrr - oof_dtd >= 0.0010),
           G3=(g_dtrr - g_dtd, g_dtrr - g_dtd >= -0.005))
ok = all(v[1] for v in res.values())
lines = [f'gate_rr | prereg sha {PREREG_SHA[:8]} | selftest {A.selftest}',
         f'  OOF12: RR {oof["RR"]:.6f} DT {oof["DT"]:.6f} D {oof["D"]:.6f} | mean(DT,RR) {oof_dtrr:.6f} mean(DT,D) {oof_dtd:.6f}',
         f'  G1 strength  OOF12(RR) {oof["RR"]:.6f} >= .881660 -> {"PASS" if res["G1"][1] else "FAIL"}',
         f'  G2 diversity {res["G2"][0]:+.6f} >= +.0010 -> {"PASS" if res["G2"][1] else "FAIL"}',
         f'  G3 gold58    mean(DT,RR) {g_dtrr:.4f} - mean(DT,D) {g_dtd:.4f} = {res["G3"][0]:+.4f} >= -.005 -> {"PASS" if res["G3"][1] else "FAIL"}',
         f'  report: Spearman RR~DT {sp("RR", "DT"):.4f} RR~D {sp("RR", "D"):.4f} DT~D {sp("DT", "D"):.4f}',
         f'RR GATE: {"PASS" if ok else "FAIL"}' + (' (SELFTEST: RR := D, no verdict)' if A.selftest else '')]
print('\n'.join(lines))
if A.selftest: sys.exit(0)
os.makedirs(H('~/oai_build/out'), exist_ok=True)
open(H('~/oai_build/out/gate_rr.txt'), 'w').write('\n'.join(lines) + '\n')
tmp = H('~/oai_build/out/rr.verdict.tmp'); open(tmp, 'w').write(('PASS' if ok else 'FAIL') + '\n'); os.replace(tmp, H('~/oai_build/out/rr.verdict'))
sys.exit(0 if ok else 1)
