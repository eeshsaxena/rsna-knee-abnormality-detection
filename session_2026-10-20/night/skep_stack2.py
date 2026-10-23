# skep_stack2.py (10-20 night, Claude, SKEPTIC). RESEARCH ONLY, local CPU, NO OAI. Writes skep_stack2.log.
# RSNA-population analog of recal's gap-matched OAI read: reweight the RSNA train knees (report v2 labels, OOF models R7 and the
# 3-model mean) so the report 2x2 of the pair has gold74's EXPERT log OR shifted by k sd (same OR-matched cell reweighting as
# recal_box_gapmatch.py), then read rho_ns, q, gap and dAUC(target) of the exact probe line pct(t) - a pct(o).
# Unlike OAI, this keeps the RSNA images and the RSNA co-firing (R7 train MM-LM rho .42, 3-mean .54), so it tests the transfer
# premise directly. Control: LOA <- PF (a trio direction, the E-OA family whose public gain is known to be positive).
import sys
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
from scipy.optimize import brentq
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import auc, pct, L
OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
log = open(OUTD + 'skep_stack2.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(777)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}
OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
G74 = {('Medial Meniscus', 'Lateral Meniscus'): [25, 13, 19, 16], ('Medial Meniscus', 'Lateral OA'): [30, 7, 26, 7],
       ('Lateral OA', 'PF OA'): [37, 20, 5, 9]}


def pw(F, a=.3):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([1 - p1 - p2 + p11, p2 - p11, p1 - p11, p11])


D = CL.data()
MODELS = {'R7': (D['F7'], D['Yv2']), '3mean': (D['M3'], D['Y3v2'])}
AV = (.05, .1); KS = (-1., 0., 1., 99.)  # 99 = no reweighting (raw report coupling)
B = 150
for t, o, base in (('Lateral Meniscus', 'Medial Meniscus', 'raw'), ('Lateral OA', 'Medial Meniscus', 'pw30'), ('Lateral OA', 'PF OA', 'raw')):
    pair = [p for p in G74 if set(p) == {t, o}][0]; a_, b_ = pair
    cs = np.array(G74[pair], float) + .5; lor = np.log(cs[3] * cs[0] / (cs[1] * cs[2])); lsd = np.sqrt((1 / cs).sum())
    for mn, (F0, Y) in MODELS.items():
        F = pw(F0) if base == 'pw30' else F0
        m = ~np.isnan(Y[:, J[a_]]) & ~np.isnan(Y[:, J[b_]]); ya = Y[m, J[a_]].astype(int); yb = Y[m, J[b_]].astype(int)
        key = ya * 2 + yb; po = np.bincount(key, minlength=4) / len(key)
        lor_rep = np.log(((key == 3).sum() + .5) * ((key == 0).sum() + .5) / (((key == 1).sum() + .5) * ((key == 2).sum() + .5)))
        yt = ya if t == a_ else yb; Ft = F[m, J[t]]; Fo = F[m, J[o]]
        outs = {x: pct(Ft) - x * pct(Fo) for x in AV}
        P('== %s <- %s (%s base) | %s | n %d | report log OR %+.2f | gold74 expert log OR %+.2f (sd %.2f)' % (t, o, base, mn, m.sum(), lor_rep, lor, lsd))
        for ks in KS:
            if ks == 99.: p = np.ones(len(key)) / len(key); lab = 'raw report coupling'
            else:
                tg = or_cells(ya.mean(), yb.mean(), lor + ks * lsd); w = (tg / np.maximum(po, 1e-12))[key]; p = w / w.sum(); lab = 'gold74 %+.0f sd' % ks
            ess = 1 / (p ** 2).sum(); R, Q, Dd = [], [], {x: [] for x in AV}
            for _ in range(B):
                ii = rng.choice(len(key), len(key), p=p); y = yt[ii]
                zt = ns(Ft[ii]); zo = ns(Fo[ii]); ct = np.cov(zt, y)[0, 1]
                R.append(np.corrcoef(zt, zo)[0, 1]); Q.append(np.cov(zo, y)[0, 1] / ct)
                b0 = auc(Ft[ii], y)
                for x in AV: Dd[x].append(auc(outs[x][ii], y) - b0)
            P('   %-20s ess %4d | rho %.3f q %.3f gap %+.3f | dAUC_t %s' % (lab, ess, np.mean(R), np.mean(Q), np.mean(R) - np.mean(Q),
              ' '.join('a%+.2f %+.4f (sd %.4f, P>0 %.2f)' % (x, np.mean(Dd[x]), np.std(Dd[x]), np.mean(np.array(Dd[x]) > 0)) for x in AV)))
P('SKEP_STACK2 DONE')
