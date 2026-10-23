# info_f32_ties.py (10-20, Claude): why the shipped float32 code path reads c4 .94113 / eoa20 .94360 on gold58 while the float64
# numpy screens (m14, e1_stages, local_mech) read .94102 / .94341, and what that does to E1's gold58 "+.00009". Info only.
import numpy as np, pandas as pd
from scipy.stats import rankdata
G = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
def run(e1, a, f32):
    V = np.zeros((n, 12))
    for j, c in enumerate(L):
        w = C4W.get(c, .5); V[:, j] = w * (.75 * pct(R384[:, j]) ** e1 + .25 * pct(R224[:, j]) ** e1) + (1 - w) * pct(R6[:, j]) ** e1
    if f32: V = V.astype(np.float32).astype(np.float64)
    F = np.stack([.8 * pct(V[:, j]) ** e1 + .2 * pct(S6[:, j]) ** e1 for j in range(12)], 1)
    if f32: F = F.astype(np.float32).astype(np.float64)
    if a:
        oi = [4, 5, 6]; U = {t: pct(F[:, t]) for t in oi}; F2 = F.copy()
        for t in oi:
            o = [m for m in oi if m != t]; F2[:, t] = pct(U[t] - a * (U[o[0]] + U[o[1]]) / 2.0)
        F = F2
    ties = sum(n - len(np.unique(F[:, j])) for j in range(12)); tV = sum(n - len(np.unique(V[:, j])) for j in range(12))
    return np.array([auc(F[:, j], Y[:, j]) for j in range(12)]), ties, tV
for nm, e1, a in (('c4', 1, 0), ('eoa20', 1, .2), ('E1', 2, 0), ('E1+eoa20', 2, .2)):
    p64, t64, v64 = run(e1, a, False); p32, t32, v32 = run(e1, a, True)
    print('%-9s float64 %.5f (tied rows: stage3 %d, final %d) | float32 kernel path %.5f (tied rows: stage3 %d, final %d) | per-label f32 - f64 %s' % (
        nm, p64.mean(), v64, t64, p32.mean(), v32, t32, [float(round(x, 4)) for x in p32 - p64]))
for f32 in (False, True):
    c, e = run(1, 0, f32)[0], run(2, 0, f32)[0]; c2, e2 = run(1, .2, f32)[0], run(2, .2, f32)[0]
    print('%s: E1 - c4 %+.5f | (E1+eoa20) - eoa20 %+.5f' % ('float32 kernel path' if f32 else 'float64 screen     ', e.mean() - c.mean(), e2.mean() - c2.mean()))
