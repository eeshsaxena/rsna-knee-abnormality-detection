"""vfy_proxy_b.py : is the gold58 'effective r = latent r + .062' offset a small-sample property?
Uses the fold-0 val OOF (867 studies, labels = steven_v4 >= .5, all members OOS). For each ordered pair of fleet members:
  full sample: within-class latent r (Spearman -> Pearson, pooled Fisher z) and the macro binormal r_fit of the measured
               per-column rank-blend curve (W .10/.15/.25/.35/.50)
  gold58-sized case-control subsamples: per label, draw n1 positives and n0 negatives with the gold58 counts, repeat 200x,
               same two quantities -> offset r_fit - r_latent
Box, CPU, read-only inputs; prints only.
"""
import os, numpy as np, pandas as pd
from scipy.stats import rankdata, norm, spearmanr
from scipy.optimize import minimize_scalar

H = os.path.expanduser
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
G58_POS = [24, 9, 26, 23, 15, 11, 21, 35, 27, 12, 19, 18]
M = {'mix50': '~/fleet_runs/f0_s42/oof_f0_s42_raw.csv', 'v4': '~/fleet_runs/f0_s42_v4/oof_f0_s42_raw.csv',
     'ttr': '~/fleet_runs/f0_s1042_ttr/oof_f0_s1042_raw.csv'}
D = {k: pd.read_csv(H(v), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID') for k, v in M.items()}
ids = sorted(set.intersection(*[set(d.index) for d in D.values()]))
lab = pd.read_csv(H('~/rsna_train/llm_labels_v4_blend.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
Y = lab.loc[ids, T].values >= 0.5
X = {k: d.loc[ids, T].values.astype(float) for k, d in D.items()}
print('n', len(ids), '| val positives per label', Y.sum(0).tolist())
WS = np.array([0.10, 0.15, 0.25, 0.35, 0.50])


def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def wc_r(a, b, y):
    zs, ws = [], []
    for c in (False, True):
        m = y == c
        if m.sum() >= 5:
            rs = spearmanr(a[m], b[m]).correlation
            zs.append(np.arctanh(np.clip(2 * np.sin(np.pi * rs / 6), -.999, .999))); ws.append(m.sum() - 3)
    return np.tanh(np.average(zs, weights=ws))


def dv(a): return np.sqrt(2) * norm.ppf(a)


def bl(aB, aM, r, w):
    v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * r
    return norm.cdf(((1 - w) * dv(aB) + w * dv(aM)) / np.sqrt(2 * v))


def one(b, m, idx_per_label):
    aB, aM, rl, curve = [], [], [], np.zeros(len(WS))
    for j, idx in enumerate(idx_per_label):
        y = Y[idx, j]; sb = X[b][idx, j]; sm = X[m][idx, j]
        ab = auc(y, sb); am = auc(y, sm); aB.append(ab); aM.append(am)
        rb, rm = rankdata(sb), rankdata(sm)
        curve += np.array([auc(y, (1 - w) * rb + w * rm) - ab for w in WS])
        rl.append(wc_r(sb, sm, y))
    aB, aM, curve, rlat = np.mean(aB), np.mean(aM), curve / 12, float(np.mean(rl))
    f = lambda r: float(np.sum((bl(aB, aM, r, WS) - aB - curve) ** 2))
    o = minimize_scalar(f, bounds=(0, 0.999), method='bounded')
    return aB, aM, rlat, float(o.x), curve


full_idx = [np.arange(len(ids))] * 12
pos = [np.where(Y[:, j])[0] for j in range(12)]; neg = [np.where(~Y[:, j])[0] for j in range(12)]
rng = np.random.default_rng(2026)
pairs = [('mix50', 'v4'), ('v4', 'mix50'), ('mix50', 'ttr'), ('ttr', 'mix50'), ('v4', 'ttr'), ('ttr', 'v4')]
print('pair (base<-member) | FULL: aB aM r_lat r_fit offset | 58-subsample x200: median r_lat, r_fit, offset (interior fits), mean offset, frac interior')
for b, m in pairs:
    aB, aM, rl, rf, _ = one(b, m, full_idx)
    offs, rls, rfs = [], [], []
    for _ in range(200):
        idx = [np.concatenate([rng.choice(pos[j], min(G58_POS[j], len(pos[j])), replace=False),
                               rng.choice(neg[j], 58 - G58_POS[j], replace=False)]) for j in range(12)]
        _, _, rl_s, rf_s, _ = one(b, m, idx)
        rls.append(rl_s); rfs.append(rf_s)
        if 0.005 < rf_s < 0.994: offs.append(rf_s - rl_s)
    print(f'{b:5s}<-{m:5s} | FULL aB {aB:.4f} aM {aM:.4f} r_lat {rl:.3f} r_fit {rf:.3f} off {rf - rl:+.3f}'
          f' | SUB med r_lat {np.median(rls):.3f} r_fit {np.median(rfs):.3f} off med {np.median(offs):+.3f} mean {np.mean(offs):+.3f}'
          f' sd {np.std(offs):.3f} interior {len(offs) / 200:.2f}')
