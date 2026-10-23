# moacf_noise.py (10-21 levers, Claude). LOCAL ONLY. Public (n 390) and private (n 910) read noise of the MOACF line (MOA <- Cont .05, Frac .05)
# and of lomm05 for comparison: weighted resamples of the train OOF rows (UNK0 labels, EB regime weights on the raking set), paired macro delta sd.
import json
import numpy as np
from rg_lib import L, J, OA, eoa, pct, load_all, near_pd, copula_cells, weights_from_target, tet_xy_fast
from scipy.stats import rankdata
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p; return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
D = load_all(); X = D['X']; Y = np.nan_to_num(D['Y'], nan=0.)
cal = json.load(open('regime_cal.json')); dl = json.load(open('regime_dlt.json')); REL = dl['rel']['nontrio']
Trep = np.eye(12); Dm = np.zeros((12, 12))
for i in range(12):
    for j in range(i + 1, 12): Trep[i, j] = Trep[j, i] = tet_xy_fast(Y[:, i], Y[:, j])
for k, e in dl['pairs'].items():
    a, b = k.split(' | '); Dm[J[a], J[b]] = Dm[J[b], J[a]] = e['delta']
trio = lambda i, j: L[i] in OA and L[j] in OA
mD = np.mean([Dm[i, j] for i in range(12) for j in range(i + 1, 12) if not trio(i, j)])
De = np.array([[0 if i == j else (Dm[i, j] if trio(i, j) else mD + REL * (Dm[i, j] - mD)) for j in range(12)] for i in range(12)])
rng = np.random.default_rng(3)
for nm, t, parts in (('MOACF', 'Medial OA', [('Contusion', .05), ('Fracture', .05)]), ('lomm05', 'Lateral OA', [('Medial Meniscus', .05)])):
    cols = OA + [o for o, _ in parts]; jc = [J[c] for c in cols]; Ys = Y[:, jc].astype(int)
    w = weights_from_target(Ys, copula_cells(Ys.mean(0), near_pd(Trep[np.ix_(jc, jc)] + De[np.ix_(jc, jc)]))); p = w / w.sum()
    for mn in ('R7', '3mean'):
        B = eoa(X[mn], .3, 'pw'); col = B[:, J[t]].copy()
        for o, a in parts: col = pct(pct(col) - a * pct(B[:, J[o]]))
        y = Y[:, J[t]].astype(int); out = {}
        for n in (390, 910):
            d = []
            for _ in range(800):
                ii = rng.choice(len(y), n, p=p)
                if y[ii].sum() in (0, n): continue
                d.append(auc(col[ii], y[ii]) - auc(B[ii, J[t]], y[ii]))
            out[n] = (np.mean(d) / 12 * 1e4, np.std(d) / 12 * 1e4)
        print('%-7s %-6s mean %+.2fe-4 macro | paired sd public (390) %.2fe-4, private (910) %.2fe-4' % (nm, mn, out[390][0], out[390][1], out[910][1]))
