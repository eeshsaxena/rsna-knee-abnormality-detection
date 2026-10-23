# m3_c4proxy.py (10-19 trick lens, research only, CPU): what would a rubric-state transform of the R6 leg do INSIDE c4 on gold58?
# R6's own state read-out on gold58 does not exist yet (needs a --dump-state GPU serve), so R7 OOF's state heads stand in (same
# heliosli family, R6-R7 rho .94). Control: mixing R7's BINARY head into the R6 slot the same way (the "R7 add", LB-negative).
# The regime-specific part is (rubric variant) - (binary control).
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work')
from load import L, auc, pct, gold, G
import importlib.util
spec = importlib.util.spec_from_file_location('m2', 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work/m2_states.py')
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
# rubric weights + STRICT list copied from m2_states.py (pre-registered there)
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work/m2_states.py').read()
ns = {}; exec(src[src.index('W = {'):src.index("d = pd.read_csv")], ns); W, STRICT = ns['W'], ns['STRICT']
d = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids]
R7B = d[['p|' + c for c in L]].values
R7R = np.stack([sum(d['state|%s|%s' % (c, s)].values * w for s, w in W[c].items()) for c in L], 1)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}


def build(r6col, ix):
    P = np.zeros((len(ix), 12))
    for j, c in enumerate(L):
        w = C4.get(c, .5); rr = .75 * pct(R384[ix, j]) + .25 * pct(R224[ix, j])
        b = w * rr + (1 - w) * pct(r6col(ix, j)); P[:, j] = .8 * pct(b) + .2 * pct(S6[ix, j])
    return P


def variant(kind, labels, a=.5):
    def f(ix, j):
        if L[j] not in labels or kind == 'c4': return R6[ix, j]
        other = {'rub': R7R, 'bin': R7B}[kind][ix, j]
        return (1 - a) * pct(R6[ix, j]) + a * pct(other)
    return f


rng = np.random.default_rng(20261019); idxs = []
while len(idxs) < 2000:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
full = np.arange(n)
P0 = build(variant('c4', []), full)
A0 = np.array([auc(P0[:, j], Y[:, j]) for j in range(12)])
B0 = None
print('c4 gold58 macro %.5f' % A0.mean())
res = {}
for name, kind, labs in [('rub_STRICT', 'rub', STRICT), ('bin_STRICT', 'bin', STRICT), ('rub_ALL', 'rub', L), ('bin_ALL', 'bin', L)]:
    P = build(variant(kind, labs), full); A = np.array([auc(P[:, j], Y[:, j]) for j in range(12)])
    res[name] = (A - A0)
    print('%-11s macro d %+.5f | per-label %s' % (name, (A - A0).mean(), np.round(A - A0, 4).tolist()))
# bootstrap of the regime-specific contrast (rub - bin) and of rub alone, STRICT and ALL
for labs, tag in [(STRICT, 'STRICT'), (L, 'ALL')]:
    dr, dc = [], []
    for ix in idxs:
        Yb = Y[ix]
        p0 = build(variant('c4', []), ix); pr = build(variant('rub', labs), ix); pb = build(variant('bin', labs), ix)
        a0 = np.mean([auc(p0[:, j], Yb[:, j]) for j in range(12)]); ar = np.mean([auc(pr[:, j], Yb[:, j]) for j in range(12)])
        ab = np.mean([auc(pb[:, j], Yb[:, j]) for j in range(12)])
        dr.append(ar - a0); dc.append(ar - ab)
    dr, dc = np.array(dr), np.array(dc)
    print('%s boot: rub-c4 mean %+.5f sd %.5f P>0 %.3f | rub-bin (regime-specific) mean %+.5f sd %.5f P>0 %.3f' % (
        tag, dr.mean(), dr.std(), (dr > 0).mean(), dc.mean(), dc.std(), (dc > 0).mean()))
