# subsample_sd.py: MEASURED paired noise of operator changes at the public (390) and private (910) sizes, on the OOF ruler
# (R7 p + s44 + s45 at .5/.25/.25 rank = base; v4 report labels). Draw disjoint 390 / 910 subsets (as Kaggle's split), recompute
# percentiles INSIDE each subset (as the kernel's pct over the test set would), and record the macro delta of each variant vs base.
# Also a calibration row: a one-line weight tweak (R7 .5 -> .45) to compare with lb_private.md's MODEL sd (1.5-2.5e-4).
import numpy as np, pandas as pd
from load import *
d7 = r7(); s44, _ = s6seed(1044); s45, _ = s6seed(1045); lab = labels('v4')
ii = [i for i in s45.index if i in s44.index and i in d7.index]
A = d7.loc[ii, ['p|' + c for c in L]].values; B = s44.loc[ii, L].values; C = s45.loc[ii, L].values; Y = lab.loc[ii, L].values.astype(int)
N = len(ii); zst = lambda v: (v - v.mean()) / (v.std() + 1e-12)
def blend(ix, op, w=(.5, .25, .25)):
    out = np.zeros((len(ix), 12))
    for j in range(12):
        X = [A[ix, j], B[ix, j], C[ix, j]]
        if op == 'rank': out[:, j] = sum(w[k] * pct(X[k]) for k in range(3))
        elif op == 'pow2': out[:, j] = sum(w[k] * pct(X[k]) ** 2 for k in range(3))
        elif op == 'invgeo': out[:, j] = -sum(w[k] * np.log(1 - pct(X[k]) + .5 / len(ix)) for k in range(3))
        elif op == 'zlogit': out[:, j] = sum(w[k] * zst(lg(X[k])) for k in range(3))
    return out
def mac(P, ix):
    return np.mean([auc(P[:, j], Y[ix, j]) for j in range(12) if 0 < Y[ix, j].sum() < len(ix)])
rng = np.random.default_rng(1)
V = [('pow2', 'pow2', (.5, .25, .25)), ('invgeo', 'invgeo', (.5, .25, .25)), ('zlogit', 'zlogit', (.5, .25, .25)), ('w R7 .45', 'rank', (.45, .275, .275)),
     ('w R7 .40', 'rank', (.40, .30, .30))]
res = {v[0]: {390: [], 910: []} for v in V}
full = {v[0]: mac(blend(np.arange(N), v[1], v[2]), np.arange(N)) - mac(blend(np.arange(N), 'rank'), np.arange(N)) for v in V}
for r in range(150):
    perm = rng.permutation(N); pub, prv = perm[:390], perm[390:1300]
    for ix, k in ((pub, 390), (prv, 910)):
        b0 = mac(blend(ix, 'rank'), ix)
        for nm, op, w in V:
            res[nm][k].append(mac(blend(ix, op, w), ix) - b0)
print('variant | full-ruler delta (n=%d) | public-size mean, sd, P>0 | private-size mean, sd, P>0 | corr(pub, priv) over draws' % N)
for nm, _, _ in V:
    a, b = np.array(res[nm][390]), np.array(res[nm][910])
    print('%-9s %+.5f | %+.5f %.5f %.2f | %+.5f %.5f %.2f | %.2f' % (nm, full[nm], a.mean(), a.std(), (a > 0).mean(), b.mean(), b.std(), (b > 0).mean(), np.corrcoef(a, b)[0, 1]))
