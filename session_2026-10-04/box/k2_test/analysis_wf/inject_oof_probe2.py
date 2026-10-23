"""Rad heads: do e10 / e13 / e13-on-e11-layout share a fold split (argmax-loss head agreement)?"""
import numpy as np
from collections import Counter
from inject_common import *
D = load_all(); Y = D['Y']; n = len(Y)
def loss(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean(1)
am = {}
for k in ('rad_e10_raw', 'rad_e13_raw', 'rad_legacy_second_raw'):
    L = np.stack([loss(h) for h in D[k]]); am[k] = np.argmax(L, 0)
    # rank-based alternative: per-head AUC-style margin; also which head is most 'different' from the others
    dev = np.stack([np.abs(D[k][h] - np.delete(D[k], h, 0).mean(0)).mean(1) for h in range(5)])
    am[k + '_dev'] = np.argmax(dev, 0)
ks = list(am)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print(f'{ks[i]:26s} vs {ks[j]:26s} agreement {(am[ks[i]] == am[ks[j]]).mean():.2f}')
maj = np.load('oof_dino_fold_majority.npy')[:, 0]
for k in ks:
    print(k, 'agree DINO fold', round(float((am[k] == maj).mean()), 2), 'cross-tab rows=DINO fold:')
    print(pd.crosstab(maj, am[k]).to_string())
