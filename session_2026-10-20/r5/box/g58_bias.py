# g58_bias.py (10-20 r5, Claude): RSNA gold58 only (no OAI). How much does the store-rebuild path move R6's OA coupling and its
# E-OA delta? Paired bootstrap over the 58 studies of (approx - exact): mean OA model rho (normal scores) and OA-mean E-OA dAUC
# at a .1 / .2 (gold58 expert labels). Aggregates only.
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/r5_oai'))
from r5_analyze import rd, ns, auc, decor, OA, LAB, H
pred = sys.argv[1]
g = pd.read_csv(H + '/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g = g[g.source == 'original_gold58'].set_index('StudyInstanceUID')[OA].astype(float).dropna()
ex = rd(sys.argv[2] if len(sys.argv) > 2 else H + '/r6g58/run_b/worker0/predictions.csv'); ap = rd(pred)
ids = g.index.intersection(ex.index).intersection(ap.index)
Y = g.loc[ids].values.astype(int); E = ex.loc[ids, OA].values; A = ap.loc[ids, OA].values
def stats(F, y):
    Z = np.stack([ns(F[:, k]) for k in range(3)], 1); r = np.corrcoef(Z.T)[np.triu_indices(3, 1)].mean()
    b = np.mean([auc(F[:, k], y[:, k]) for k in range(3)])
    d = [np.mean([auc(decor(F, a)[:, k], y[:, k]) for k in range(3)]) - b for a in (.1, .2)]
    qq = []
    for k in range(3):
        o = [m for m in range(3) if m != k]; ck = np.corrcoef(Z[:, k], y[:, k])[0, 1]
        qq.append(np.mean([np.corrcoef(Z[:, j], y[:, k])[0, 1] for j in o]) / ck)
    return np.array([r, b] + d + [np.mean(qq), r - np.mean(qq)])
pt = stats(A, Y) - stats(E, Y)
rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    ii = rng.integers(0, len(Y), len(Y)); y = Y[ii]
    if (y.sum(0) == 0).any() or (y.sum(0) == len(ii)).any(): continue
    bs.append(stats(A[ii], y) - stats(E[ii], y))
bs = np.array(bs)
print('exact :', np.round(stats(E, Y), 4).tolist(), '| approx:', np.round(stats(A, Y), 4).tolist(), '(rho, base, d.1, d.2, q, gap)')
for k, nm in enumerate(['OA rho mean', 'OA-mean base AUC', 'E-OA dAUC a.1', 'E-OA dAUC a.2', 'q', 'gap rho-q']):
    print(f'approx - exact {nm:18s} {pt[k]:+.4f} (boot sd {bs[:, k].std():.4f}, 90% [{np.percentile(bs[:, k], 5):+.4f}, {np.percentile(bs[:, k], 95):+.4f}])')
print('n', len(ids))
