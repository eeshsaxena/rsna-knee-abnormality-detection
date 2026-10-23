# m3b_slot.py (research only, CPU): the apples-to-apples proxy for "R6 + its own rubric state score" inside c4.
# Frame: c4 with R7 (same heliosli family) in R6's slot. Baseline = R7 binary head in the slot; transform = .5 pct(R7 bin) +
# .5 pct(R7 rubric state score) in the slot, on the pre-registered STRICT labels (m2_states.py) or on all 12.
# Also: "delta transfer" onto the real R6: slot = pct(R6) + .5 (pct(R7 rub) - pct(R7 bin)) re-ranked.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work/m2_states.py').read()
ns = {}; exec(src[src.index('W = {'):src.index("d = pd.read_csv")], ns); W, STRICT = ns['W'], ns['STRICT']
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
d = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids]
R7B = d[['p|' + c for c in L]].values
R7R = np.stack([sum(d['state|%s|%s' % (c, s)].values * w for s, w in W[c].items()) for c in L], 1)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
def build(slot, ix):
    P = np.zeros((len(ix), 12))
    for j, c in enumerate(L):
        w = C4.get(c, .5); rr = .75 * pct(R384[ix, j]) + .25 * pct(R224[ix, j])
        b = w * rr + (1 - w) * pct(slot(ix, j)); P[:, j] = .8 * pct(b) + .2 * pct(S6[ix, j])
    return P
V = {
 'c4': lambda ix, j: R6[ix, j],
 'R7slot_bin': lambda ix, j: R7B[ix, j],
 'R7slot_mix_STRICT': lambda ix, j: (.5 * pct(R7B[ix, j]) + .5 * pct(R7R[ix, j])) if L[j] in STRICT else R7B[ix, j],
 'R7slot_mix_ALL': lambda ix, j: .5 * pct(R7B[ix, j]) + .5 * pct(R7R[ix, j]),
 'R6xfer_STRICT': lambda ix, j: (pct(R6[ix, j]) + .5 * (pct(R7R[ix, j]) - pct(R7B[ix, j]))) if L[j] in STRICT else R6[ix, j],
 'R6xfer_ALL': lambda ix, j: pct(R6[ix, j]) + .5 * (pct(R7R[ix, j]) - pct(R7B[ix, j])),
}
mac = lambda P, yy: np.array([auc(P[:, j], yy[:, j]) for j in range(12)])
full = np.arange(n); A = {k: mac(build(f, full), Y) for k, f in V.items()}
for k in V: print('%-18s macro %.5f' % (k, A[k].mean()))
pairs = [('R7slot_mix_STRICT', 'R7slot_bin'), ('R7slot_mix_ALL', 'R7slot_bin'), ('R6xfer_STRICT', 'c4'), ('R6xfer_ALL', 'c4')]
for a, b in pairs: print('%s - %s: %+.5f per-label %s' % (a, b, (A[a] - A[b]).mean(), np.round(A[a] - A[b], 4).tolist()))
rng = np.random.default_rng(1019); idxs = []
while len(idxs) < 1500:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
B = {k: [] for k in V}
for ix in idxs:
    for k, f in V.items(): B[k].append(mac(build(f, ix), Y[ix]).mean())
B = {k: np.array(v) for k, v in B.items()}
for a, b in pairs:
    dd = B[a] - B[b]; print('boot %s - %s: mean %+.5f sd %.5f P>0 %.3f' % (a, b, dd.mean(), dd.std(), (dd > 0).mean()))
# split-half sign agreement of the per-study-resample effect (random halves, effect computed on each half separately)
agree = []; hs = []
for _ in range(400):
    perm = rng.permutation(n); h1, h2 = perm[:29], perm[29:]
    if any(Y[h, j].min() == Y[h, j].max() for h in (h1, h2) for j in range(12)): continue
    e = [mac(build(V['R6xfer_STRICT'], h), Y[h]).mean() - mac(build(V['c4'], h), Y[h]).mean() for h in (h1, h2)]
    agree.append(np.sign(e[0]) == np.sign(e[1])); hs.append(e)
hs = np.array(hs); print('R6xfer_STRICT split-half: n %d sign agreement %.3f corr %.3f mean half effect %+.5f' % (len(hs), np.mean(agree), np.corrcoef(hs.T)[0, 1], hs.mean()))
