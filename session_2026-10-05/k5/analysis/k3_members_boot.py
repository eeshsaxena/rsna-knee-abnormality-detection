#!/usr/bin/env python3
"""K3 member read (box, CPU, read-only): paired bootstrap of macro AUC differences between M1, M2 and their raw-prob mean
on (a) gold58 (gold labels, from the K3-main gold58 stage run's worker npz) and (b) the 867 fold-0 val studies (the
trainers' own OOF csvs vs steven_v4 llm_labels_v4_blend.csv, the selection target of both runs).
  python k3_members_boot.py <gold58 run dir> <M1 oof csv> <M2 oof csv>
"""
import glob, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
run, o1, o2 = sys.argv[1:4]
rng = np.random.default_rng(0)


def mauc(Y, X, idx=None):
    if idx is not None:
        Y, X = Y[idx], X[idx]
    v = [roc_auc_score(Y[:, j], X[:, j]) for j in range(12) if 0 < Y[:, j].sum() < len(Y)]
    return float(np.mean(v))


def report(tag, Y, C, nb=1000):
    n = len(Y)
    base = {k: mauc(Y, X) for k, X in C.items()}
    print(f'[{tag}] n {n}: ' + ' | '.join(f'{k} {v:.4f}' for k, v in base.items()))
    pairs = [('M2', 'M1'), ('mean', 'M1'), ('mean', 'M2')]
    boots = {p: [] for p in pairs}
    for _ in range(nb):
        i = rng.integers(0, n, n)
        m = {k: mauc(Y, X, i) for k, X in C.items()}
        for a, b in pairs:
            boots[(a, b)].append(m[a] - m[b])
    for a, b in pairs:
        d = np.array(boots[(a, b)])
        print(f'[{tag}]   {a} - {b}: {base[a] - base[b]:+.4f}  90% [{np.quantile(d, .05):+.4f}, {np.quantile(d, .95):+.4f}]  P>0 {(d > 0).mean():.2f}')
    r = np.mean([spearmanr(C['M1'][:, j], C['M2'][:, j]).correlation for j in range(12)])
    print(f'[{tag}]   rho(M1, M2) per-label mean {r:.3f}')


# (a) gold58
tr = pd.read_csv(os.path.expanduser('~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv'),
                 dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
rows = {}
for f in sorted(glob.glob(f'{run}/working/_fleet_*.npz')):
    z = np.load(f)
    names = [str(x) for x in z['names']]
    for k, sid in enumerate(z['sids']):
        for i, nm in enumerate(names):
            rows.setdefault(nm, {})[str(sid)] = z['P'][i, k].astype(np.float64)
n1 = [n for n in rows if n.startswith('f0_s42_')][0]; n2 = [n for n in rows if n.startswith('f0_s1042_')][0]
ids = [s for s in rows[n1] if s in tr.index and not np.isnan(tr.loc[s, 'ACL'])]
Y = tr.loc[ids, T].values.astype(float)
M1 = np.stack([rows[n1][s] for s in ids]); M2 = np.stack([rows[n2][s] for s in ids])
report('gold58', Y, {'M1': M1, 'M2': M2, 'mean': (M1 + M2) / 2})
# (b) 867 fold-0 val vs steven_v4 (continuous labels -> AUC vs label > .5, the trainer's own metric convention?)
lab = pd.read_csv(os.path.expanduser('~/rsna_train/llm_labels_v4_blend.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
A = pd.read_csv(o1, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
B = pd.read_csv(o2, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = [s for s in A.index if s in B.index and s in lab.index]
L = lab.loc[ids, T].values.astype(float)
Yb = (L >= 0.5).astype(float)
X1, X2 = A.loc[ids, T].values.astype(float), B.loc[ids, T].values.astype(float)
report('val867 vs v4>=.5', Yb, {'M1': X1, 'M2': X2, 'mean': (X1 + X2) / 2}, nb=300)
