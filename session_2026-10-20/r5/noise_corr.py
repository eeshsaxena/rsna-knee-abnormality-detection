# noise_corr.py (10-20 r5, decision lens): joint public / private read noise of the E-OA deltas at a .1 and a .2 on the SAME
# subsample (m12_lbnoise.py method: R7 OOF, sol56 report labels, non-gold train; competition data only, CPU). Only the SD and the
# a.1-a.2 noise CORRELATION are used downstream (the mean is the report-regime effect, not the hidden one).
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R
g, _ = gold()
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
lab = pd.read_csv(R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L]
idx = [u for u in r7.index if u not in set(g.StudyInstanceUID) and u in lab.index]
X = r7.loc[idx, ['p|' + c for c in L]].values; Y = (lab.loc[idx].fillna(0).values >= .5).astype(int)
OA = [L.index(c) for c in ('Medial OA', 'Lateral OA', 'PF OA')]
def d(ix, a):
    s = 0.
    for t in OA:
        o = [h for h in OA if h != t]
        s += auc(pct(X[ix, t]) - a * .5 * (pct(X[ix, o[0]]) + pct(X[ix, o[1]])), Y[ix, t]) - auc(X[ix, t], Y[ix, t])
    return s / 12
rng = np.random.default_rng(1020)
print('n train studies', len(idx))
for size in (390, 910):
    v = []
    for _ in range(600):
        ix = rng.choice(len(idx), size, replace=False); v.append((d(ix, .1), d(ix, .2)))
    v = np.array(v); c = np.corrcoef(v.T)[0, 1]
    dd = v[:, 1] - v[:, 0]
    print('size %d: sd a.1 %.5f sd a.2 %.5f corr %.3f | sd(a.2 - a.1) %.5f | means %+.5f %+.5f' % (size, v[:, 0].std(), v[:, 1].std(), c, dd.std(), v[:, 0].mean(), v[:, 1].mean()))
