# local_bone.py (10-20 more, Claude). RESEARCH ONLY, local CPU, aggregates only.
# Rubric exclusion check: Contusion = "bone marrow edema-like signal from impact, WITHOUT a discrete fracture line".
# Pair decorrelation Contusion <-> Fracture on gold58 c4 (theory rho vs q, symmetric decor at a .1 / .2 with bootstrap) and on
# report OOF (negative control). gold58 is not a selection ruler; OAI has no contusion / fracture labels.
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, r7, s6seed, labels
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
F = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); b = w * rr + (1 - w) * pct(R6[:, j])
    F[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
jc, jf = L.index('Contusion'), L.index('Fracture')
print('gold58 counts (Cont, Frac): 00 %d 01 %d 10 %d 11 %d' % tuple(((Y[:, jc] == a) & (Y[:, jf] == b)).sum() for a in (0, 1) for b in (0, 1)))
for t, o in ((jc, jf), (jf, jc)):
    zt, zo = ns(F[:, t]), ns(F[:, o]); y = Y[:, t]
    print('theory %s <- %s: rho %.3f q %.3f' % (L[t], L[o], np.corrcoef(zt, zo)[0, 1], np.corrcoef(zo, y)[0, 1] / np.corrcoef(zt, y)[0, 1]))
rng = np.random.default_rng(7)
def dec(M, a):
    O = M.copy(); U = pct(M[:, jc]), pct(M[:, jf]); O[:, jc] = U[0] - a * U[1]; O[:, jf] = U[1] - a * U[0]; return O
for a in (.1, .2):
    O = dec(F, a); d = [auc(O[:, j], Y[:, j]) - auc(F[:, j], Y[:, j]) for j in (jc, jf)]
    bs = []
    for _ in range(2000):
        ii = rng.integers(0, n, n)
        bs.append(sum(auc(O[ii, j], Y[ii, j]) - auc(F[ii, j], Y[ii, j]) for j in (jc, jf)) / 12)
    bs = np.array(bs); print('gold58 c4 bone decor a %.1f: Cont %+.4f Frac %+.4f macro %+.5f boot sd %.5f P>0 %.2f' % (a, d[0], d[1], sum(d) / 12, bs.std(), (bs > 0).mean()))
d7 = r7(); goldids = set(g.StudyInstanceUID)
for src in ('v2', 'v4'):
    lab = labels(src); tr = [i for i in d7.index if i in lab.index and i not in goldids]; M = d7.loc[tr, ['p|' + c for c in L]].values; Yr = lab.loc[tr, L].values
    for a in (.1, .2):
        O = dec(M, a); dd = []
        for j in (jc, jf):
            ok = ~np.isnan(Yr[:, j]); dd.append(auc(O[ok, j], Yr[ok, j].astype(int)) - auc(M[ok, j], Yr[ok, j].astype(int)))
        print('report OOF %s R7 bone decor a %.1f: Cont %+.4f Frac %+.4f' % (src, a, *dd))
