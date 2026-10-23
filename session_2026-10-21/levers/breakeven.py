# breakeven.py (10-21 levers, Claude). RESEARCH ONLY, local, NO OAI. For named directions: how much of the pair's OWN same-knee D (gold58
# expert - report-v2 tet) must be real for the transform to break even. Target = report train tet + EB D for every other pair, and for the
# pair itself meanD + f x (D - meanD), f in a grid (f = rel .127 is EB, f = 1 is RAW). Writes breakeven.log.
import json
import numpy as np
from rg_lib import L, J, AB, OA, eoa, near_pd, copula_cells, weights_from_target, load_all, pct, wauc_multi
Bd = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/'
sw = json.load(open(Bd + 'regime_dlt_sweep.json')); cal = json.load(open(Bd + 'regime_cal.json')); dl = json.load(open(Bd + 'regime_dlt.json'))
REL, meanD = sw['REL'], sw['meanD']; Trep = np.array(cal['Trep'])
Dm = np.zeros((12, 12))
for k, e in dl['pairs'].items():
    a, b = k.split(' | '); Dm[J[a], J[b]] = Dm[J[b], J[a]] = e['delta']
trio = lambda i, j: L[i] in OA and L[j] in OA
Deb = np.array([[0 if i == j else (Dm[i, j] if trio(i, j) else meanD + REL * (Dm[i, j] - meanD)) for j in range(12)] for i in range(12)])
D = load_all(); X, Y = D['X'], D['Y']; BASE = {m: eoa(F, .3, 'pw') for m, F in X.items()}
FS = [0., .127, .25, .4, .55, .7, .85, 1.]
CELLS = [('MCL', 'Medial Meniscus', .10), ('MCL', 'Medial Meniscus', .05), ('Lateral OA', 'Medial Meniscus', .05), ('Lateral OA', 'Medial Meniscus', .10),
         ('Medial OA', 'Contusion', .10), ('Medial OA', 'Fracture', .10), ('ACL', 'PF OA', -.10), ('ACL', 'Medial OA', -.10),
         ('Medial Meniscus', 'Lateral Meniscus', .10), ('Baker\'s', 'Fracture', .10), ('Fracture', 'Baker\'s', .10), ('Baker\'s', 'ACL', .10)]
out = open(Bd + 'breakeven.log', 'w')
def P(s): print(s); out.write(s + '\n')
P('label AUC x1e4, rulers R7 / s1044 / s1045 / 3mean, by f = share of the pair\'s own same-knee D taken as real (others at EB)')
for t, o, a in CELLS:
    cols = list(dict.fromkeys(([c for c in OA] if (t in OA or o in OA) else []) + [t, o])); jc = [J[c] for c in cols]
    m_ = ~np.isnan(Y[:, jc]).any(1); Ys = Y[m_][:, jc].astype(int); yt = Ys[:, cols.index(t)]
    i, j = J[t], J[o]; Ws = []
    for f in FS:
        M = Deb.copy(); M[i, j] = M[j, i] = meanD + f * (Dm[i, j] - meanD)
        Ws.append(weights_from_target(Ys, copula_cells(Ys.mean(0), near_pd(Trep[np.ix_(jc, jc)] + M[np.ix_(jc, jc)]))))
    rows = []
    for mn in X:
        Bm = BASE[mn][m_]; bt = Bm[:, i]
        rows.append(wauc_multi(pct(pct(bt) - a * pct(Bm[:, j])), yt, Ws) - wauc_multi(bt, yt, Ws))
    R = np.array(rows) * 1e4
    be = []
    for r in R:
        s = np.sign(r); k = np.where(np.diff(s) != 0)[0]
        be.append('%.2f' % (FS[k[0]] + (FS[k[0] + 1] - FS[k[0]]) * (-r[k[0]]) / (r[k[0] + 1] - r[k[0]])) if len(k) else ('always+' if r[0] > 0 else 'never+'))
    P('%s<-%s a %+.2f | D %+.2f | ' % (AB[t], AB[o], a, Dm[i, j]) + ' '.join('f%.2f %s' % (f, '/'.join('%+.0f' % v for v in R[:, q])) for q, f in enumerate(FS)) + ' | break-even f ' + ' '.join(be))
