# nested_oof.py: the c4-shaped NESTED structure on the OOF ruler: within-family first (s44 + s45, like r384 + r224), then
# cross-family with R7 (like stage 3), transform T applied at every stage as in g58_nested.py. Plus per-label sign agreement
# between this OOF ruler and gold58 nested c4 for pow2 / invgeo, and Spearman of each variant vs c4 on gold58.
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from load import *
d7 = r7(); s44, _ = s6seed(1044); s45, _ = s6seed(1045)
ii = [i for i in s45.index if i in s44.index and i in d7.index]
A = d7.loc[ii, ['p|' + c for c in L]].values; B = s44.loc[ii, L].values; C = s45.loc[ii, L].values
TF = {'rank': lambda v: pct(v), 'pow2': lambda v: pct(v) ** 2, 'invgeo': lambda v: -np.log(1 - pct(v) + .5 / len(v))}
def nest(T, wR=.5):
    out = np.zeros((len(ii), 12))
    for j in range(12):
        fam = .5 * T(B[:, j]) + .5 * T(C[:, j]); out[:, j] = wR * T(A[:, j]) + (1 - wR) * T(fam)
    return out
res = {}
for src in ('v4', 'v2'):
    lab = labels(src).loc[ii, L].values
    def per(S): return np.array([auc(S[~np.isnan(lab[:, j]), j], lab[~np.isnan(lab[:, j]), j].astype(int)) for j in range(12)])
    for wR in (.5, .7):
        b = per(nest(TF['rank'], wR))
        for nm in ('pow2', 'invgeo'):
            d = per(nest(TF[nm], wR)) - b; res[(src, wR, nm)] = d
            print('OOF %s nested (R7 %.1f) %-6s macro %+.5f labels +%d/-%d' % (src, wR, nm, d.mean(), (d > 0).sum(), (d < 0).sum()))
# gold58 nested c4 per-label deltas (recomputed as in g58_nested.py)
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
def c4(T):
    P = np.zeros((n, 12))
    for j, c in enumerate(L):
        w = C4.get(c, .5); rr = .75 * T(R384[:, j]) + .25 * T(R224[:, j])
        b = w * T(rr) + (1 - w) * T(R6[:, j]) if T is not TF['rank'] else w * rr + (1 - w) * T(R6[:, j])
        P[:, j] = .8 * T(b) + .2 * T(S6[:, j])
    return P
P0 = c4(TF['rank'])
for nm in ('pow2', 'invgeo'):
    P = c4(TF[nm]); dg = np.array([auc(P[:, j], Y[:, j]) - auc(P0[:, j], Y[:, j]) for j in range(12)])
    sp = np.mean([spearmanr(P[:, j], P0[:, j])[0] for j in range(12)])
    do = res[('v4', .5, nm)]
    nz = (dg != 0) & (do != 0)
    print('%-6s gold58 per-label %s | sign agreement with OOF v4 nested on %d nonzero labels: %d | mean Spearman vs c4 on gold58 %.4f' % (
        nm, np.round(dg, 4).tolist(), nz.sum(), int((np.sign(dg[nz]) == np.sign(do[nz])).sum()), sp))
    print('       OOF v4 per-label %s' % np.round(do, 4).tolist())
