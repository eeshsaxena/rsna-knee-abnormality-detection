# severity.py: a hierarchical prior in its simplest pre-specified form: every label borrows a little of a study-level
# "abnormality" score (mean of the 12 label percentiles, own label left out): s'_t = pct(s_t) + a * pct(mean_{k != t} pct(s_k)).
# Gold58 (c4 blend, Kaggle annotation) with paired bootstrap, and the OOF ruler (R7 + s44 .5/.5 rank, v4 + v2 report labels).
import numpy as np, pandas as pd
from load import *
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
P = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j])
    P[:, j] = .8 * pct(w * rr + (1 - w) * pct(R6[:, j])) + .2 * pct(S6[:, j])
def sev(Q, a):
    R = np.column_stack([pct(Q[:, j]) for j in range(12)]); out = np.zeros_like(R)
    for t in range(12):
        m = np.delete(R, t, axis=1).mean(1); out[:, t] = R[:, t] + a * pct(m)
    return out
rng = np.random.default_rng(3); idxs = []
while len(idxs) < 2000:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
B0 = np.array([[auc(P[ix, j], Y[ix, j]) for j in range(12)] for ix in idxs])
for a in (.05, .1, .2):
    S = sev(P, a); per = np.array([auc(S[:, j], Y[:, j]) - auc(P[:, j], Y[:, j]) for j in range(12)])
    Bb = np.array([[auc(S[ix, j], Y[ix, j]) for j in range(12)] for ix in idxs]); d = (Bb - B0).mean(1)
    print('gold58 c4 severity a=%.2f: macro %+.5f sd %.5f P>0 %.3f | labels +%d/-%d | %s' % (a, per.mean(), d.std(), (d > 0).mean(), (per > 0).sum(), (per < 0).sum(),
          sorted([(L[k], round(per[k], 4)) for k in range(12)], key=lambda t: -abs(t[1]))[:4]))
d7 = r7(); s44, _ = s6seed(1044); ii = [i for i in s44.index if i in d7.index]
Q = np.column_stack([.5 * pct(d7.loc[ii, 'p|' + c].values) + .5 * pct(s44.loc[ii, c].values) for c in L])
for src in ('v4', 'v2'):
    lab = labels(src).loc[ii, L].values
    def m_(S):
        return np.array([auc(S[~np.isnan(lab[:, j]), j], lab[~np.isnan(lab[:, j]), j].astype(int)) for j in range(12)])
    b = m_(Q)
    for a in (.05, .1, .2):
        per = m_(sev(Q, a)) - b
        print('OOF %s severity a=%.2f: macro %+.5f | labels +%d/-%d | %s' % (src, a, per.mean(), (per > 0).sum(), (per < 0).sum(),
              sorted([(L[k], round(per[k], 4)) for k in range(12)], key=lambda t: -abs(t[1]))[:4]))
# label-level correlation of the 12 labels with the leave-one-out abnormality count, gold58 vs report (v4)
lab4 = labels('v4').loc[ii, L].values
for nm, YY in (('gold58', Y.astype(float)), ('report v4', lab4)):
    cs = [np.corrcoef(YY[:, t], np.delete(YY, t, axis=1).sum(1))[0, 1] for t in range(12)]
    print('%-9s corr(label, count of other positives): %s' % (nm, {L[t]: round(cs[t], 2) for t in range(12)}))
