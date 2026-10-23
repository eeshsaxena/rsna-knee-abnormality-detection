#!/usr/bin/env python3
"""gold58 read of the K3 members (box, CPU, read-only): run after the K3-main notebook ran on the 58 gold studies
(k2_box_stage.py --ids gold58.csv). Per member from the worker npz (M1, M2), the fleet solo = raw-prob mean (M1, M2),
and a rank mean for reference. Macro AUC vs the gold labels in train.csv; per-label Spearman (mean over 12 labels) vs
the parent's out-of-sample half raptor_coat_side_rank.csv; blend lift into that OOS half (W .15-.50, info only).
gold58 is IN-SAMPLE for the parent's DINO / RadImageNet arms, so the full P+F blend is NOT judged here.
  python gold58_k3.py <run dir (has working/)> [--parts ~/kmimic/runs/gold58_gpu/localLB_parts]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--parts', default=os.path.expanduser('~/kmimic/runs/gold58_gpu/localLB_parts'))
ap.add_argument('--train', default=os.path.expanduser('~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv'))
ap.add_argument('--json', default=None)
a = ap.parse_args()
W = f'{a.run}/working'
tr = pd.read_csv(a.train, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
F = pd.read_csv(f'{W}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = [s for s in F.index if s in tr.index and not np.isnan(tr.loc[s, 'ACL'])]
Y = tr.loc[ids, T].values.astype(float)
members = {}
for f in sorted(glob.glob(f'{W}/_fleet_*.npz')):
    z = np.load(f)
    names = [str(x) for x in z['names']]
    for k, sid in enumerate(z['sids']):
        for i, nm in enumerate(names):
            members.setdefault(nm, {})[str(sid)] = z['P'][i, k].astype(np.float64)
M = {nm: np.stack([d[s] for s in ids]) for nm, d in members.items()}


def rp(x): return np.column_stack([rankdata(x[:, j]) / len(x) for j in range(x.shape[1])])
def per(x): return np.array([roc_auc_score(Y[:, j], x[:, j]) for j in range(12)])
def rho(x, b): return float(np.mean([spearmanr(x[:, j], b[:, j]).correlation for j in range(12)]))
def load(name): return pd.read_csv(f'{a.parts}/{name}', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, T].values.astype(float)


B = load('raptor_coat_side_rank.csv')
names = sorted(M, key=lambda n: (not n.startswith('f0_s42_'), n))
cand = {f'M{i + 1} {nm}': M[nm] for i, nm in enumerate(names)}
if len(names) >= 2:
    cand['mean(M1,M2) raw-prob (= fleetsolo)'] = F.loc[ids, T].values.astype(float)
    cand['mean(M1,M2) rank'] = rp(np.mean([rp(M[n]) for n in names], axis=0))
    chk = float(np.abs(np.mean([M[n] for n in names], axis=0) - F.loc[ids, T].values).max())
else:
    chk = float('nan')
b0 = per(B).mean()
out = {'n': len(ids), 'positives': Y.sum(0).astype(int).tolist(), 'raptor_coat_side_auc': b0,
       'fleetsolo_vs_member_mean_maxabs': chk, 'rows': {}}
print(f'gold58 n {len(ids)} | positives {Y.sum(0).astype(int).tolist()} | fleetsolo == raw mean of members: max|diff| {chk:.2e}')
print(f'raptor_coat_side (parent OOS half) gold58 macro AUC {b0:.4f}')
print(f'{"candidate":58s} gold58   rho_vs_RCside  | lift into RC side at W .15 / .25 / .35 / .50 (info only)')
for k, X in cand.items():
    auc = per(X); r = rho(X, B)
    lifts = [per(rp((1 - w) * rp(B) + w * rp(X))).mean() - b0 for w in (0.15, 0.25, 0.35, 0.5)]
    out['rows'][k] = dict(auc=float(auc.mean()), per_label=[round(float(v), 4) for v in auc], rho_rcside=r,
                          lift_rcside=dict(zip(['w15', 'w25', 'w35', 'w50'], [round(float(v), 4) for v in lifts])))
    print(f'{k:58s} {auc.mean():.4f}   {r:.3f}          | ' + ' / '.join(f'{v:+.4f}' for v in lifts))
if len(names) >= 2:
    r12 = rho(M[names[0]], M[names[1]])
    out['rho_M1_M2'] = r12
    print(f'rho(M1, M2) per-label mean {r12:.3f}')
print('per-label AUC: ' + ' | '.join(T))
for k, X in cand.items():
    print(f'  {k[:40]:40s} ' + ' '.join(f'{v:.3f}' for v in per(X)))
if a.json:
    json.dump(out, open(a.json, 'w'), indent=1)
