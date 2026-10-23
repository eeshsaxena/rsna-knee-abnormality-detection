# local_gapboot.py (10-19 ex): paired bootstrap of the gold58 coupling gap, c4 Spearman minus gold58 expert tetrachoric, on the
# same 58 resampled studies (B 300), for the 36 OAI-covered pairs. Prints sd and P(gap > 0). Local CPU, aggregates only.
import itertools, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm, multivariate_normal
from scipy.optimize import brentq
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
L9 = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"]
AB = {"ACL": "ACL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA", "Lateral OA": "LOA", "PF OA": "PFOA",
      "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak"}
pct = lambda v: rankdata(v) / len(v)


def tetra(a, b):
    c11 = ((a == 1) & (b == 1)).sum() + .5; c10 = ((a == 1) & (b == 0)).sum() + .5
    c01 = ((a == 0) & (b == 1)).sum() + .5; c00 = ((a == 0) & (b == 0)).sum() + .5; N = c11 + c10 + c01 + c00
    h = norm.ppf(1 - (c11 + c10) / N); k = norm.ppf(1 - (c11 + c01) / N); p11 = c11 / N
    f = lambda r: multivariate_normal.cdf([-h, -k], mean=[0, 0], cov=[[1, r], [r, 1]]) - p11
    if f(-.995) > 0: return -.995
    if f(.995) < 0: return .995
    return brentq(f, -.995, .995, xtol=1e-3)


g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
C4 = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j])
    b = w * rr + (1 - w) * pct(R6[:, j]); C4[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
Y = g58[L].values
rng = np.random.default_rng(0); IDX = [rng.integers(0, n, n) for _ in range(300)]
print('| pair | gap g58 (c4 rho - gold58 tet) | boot sd | P(gap > 0) |')
print('|---|---|---|---|')
for a, b in itertools.combinations(L9, 2):
    ia, ib = L.index(a), L.index(b); ok = np.isfinite(Y[:, ia]) & np.isfinite(Y[:, ib])
    ya = Y[:, ia]; yb = Y[:, ib]
    g0 = spearmanr(C4[ok, ia], C4[ok, ib]).correlation - tetra(ya[ok].astype(int), yb[ok].astype(int))
    gs = []
    for ii in IDX:
        ii = ii[ok[ii]]; xa = ya[ii].astype(int); xb = yb[ii].astype(int)
        if xa.std() == 0 or xb.std() == 0: continue
        gs.append(spearmanr(C4[ii, ia], C4[ii, ib]).correlation - tetra(xa, xb))
    gs = np.array(gs)
    print('| %s-%s | %+.2f | %.2f | %.2f |' % (AB[a], AB[b], g0, gs.std(), (gs > 0).mean()), flush=True)
