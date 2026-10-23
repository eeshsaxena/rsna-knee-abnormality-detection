# m12_lbnoise.py: paired noise of the OA-decor change at public (390) and private (910) size, from subsamples of the report-label OOF
# (R7 OOF, sol56 labels, non-gold train). The MEAN is the report-regime effect (expected < 0); the SD is the read noise for a fixed truth.
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
rng = np.random.default_rng(3)
for size in (390, 910):
    for a in (.1, .2):
        v = np.array([d(rng.choice(len(idx), size, replace=False), a) for _ in range(600)])
        print('size %d a %.1f: mean %+.5f sd %.5f' % (size, a, v.mean(), v.std()))
