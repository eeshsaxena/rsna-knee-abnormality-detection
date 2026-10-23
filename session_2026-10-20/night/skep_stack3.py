# skep_stack3.py (10-20 night, Claude, SKEPTIC). RESEARCH ONLY, local CPU, NO OAI. Writes skep_stack3.log.
# Calibrate the RSNA-population reweighted ruler of skep_stack2.py on the E-OA family (the only family with public reads):
# rake gold74's EXPERT 8-cell OA-trio joint (+.5) to each model's report-v2 trio marginals (IPF keeps gold74's odds-ratio structure),
# weight the train knees, and read eoa10 / eoa20 / eoa30 (sym) and pw20 / pw30 (pair-weighted) vs the unmodified model as macro e-4
# (sum of the 3 OA label deltas / 12). Public (CLAIMED val/implied2.md joint, vs c4): 5.9 / 10.3 / 9.6 / 10.4 / 11.3.
# Also: is c4 R7-like or ensemble-like in MM-LM / LOA-MM co-firing? (3-model mean on gold58 if the S6 seed OOFs cover gold58.)
import sys
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import auc, pct, L
import load as LD
from cbase import load
OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
log = open(OUTD + 'skep_stack3.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(31)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}
OA = ['Medial OA', 'Lateral OA', 'PF OA']; jOA = [J[c] for c in OA]
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}


def eoa(F, a, mode):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]
        w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o] if mode == 'pw' else [1., 1.]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G


def wauc(s, y, w):
    # weighted AUC (ties half) via sorting
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    wp = w * y; wn = w * (1 - y)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    gp = np.bincount(inv, wp); gn = np.bincount(inv, wn)
    cn = np.cumsum(gn) - gn
    return float((gp * (cn + .5 * gn)).sum() / (wp.sum() * wn.sum()))


def rake(target_joint, margins, it=200):
    T = target_joint.copy()
    for _ in range(it):
        for ax in range(3):
            cur = T.sum(axis=tuple(i for i in range(3) if i != ax))
            sh = [1, 1, 1]; sh[ax] = 2
            T = T * (np.array([1 - margins[ax], margins[ax]]) / cur).reshape(sh)
    return T / T.sum()


D = CL.data()
Y74 = D['Y74'][:, jOA]
ok = ~np.isnan(Y74).any(1); Yg = Y74[ok].astype(int)
cnt = np.zeros((2, 2, 2))
for r in Yg: cnt[r[0], r[1], r[2]] += 1
P('gold74 trio joint n %d; expert prevalences %s' % (ok.sum(), np.round(Yg.mean(0), 3)))
PUB = {'eoa10': 5.9, 'eoa20': 10.3, 'eoa30': 9.6, 'pw20': 10.4, 'pw30': 11.3}
SH = {'eoa10': (.1, 'sym'), 'eoa20': (.2, 'sym'), 'eoa30': (.3, 'sym'), 'pw20': (.2, 'pw'), 'pw30': (.3, 'pw')}
for mn, (F, Y) in {'R7': (D['F7'], D['Yv2']), '3mean': (D['M3'], D['Y3v2'])}.items():
    m = ~np.isnan(Y[:, jOA]).any(1); Fm = F[m]; Ym = Y[m][:, jOA].astype(int)
    marg = Ym.mean(0)
    obs = np.zeros((2, 2, 2))
    for r in Ym: obs[r[0], r[1], r[2]] += 1
    obs /= obs.sum()
    for regime, sm in (('raw report coupling', None), ('gold74 expert coupling (raked)', .5), ('halfway (geometric mix)', 'half')):
        if sm is None: w = np.ones(len(Ym))
        else:
            tgt = rake(cnt + .5, marg)
            if sm == 'half': tgt = rake(np.sqrt(tgt * obs), marg)
            w = (tgt / obs)[Ym[:, 0], Ym[:, 1], Ym[:, 2]]
        w = w / w.mean(); ess = w.sum() ** 2 / (w ** 2).sum()
        out = {}
        for k, (a, mode) in SH.items():
            G = eoa(Fm, a, mode)
            out[k] = sum(wauc(G[:, J[c]], Ym[:, i], w) - wauc(Fm[:, J[c]], Ym[:, i], w) for i, c in enumerate(OA)) / 12 * 1e4
        # bootstrap sd for pw30
        bs = []
        for _ in range(60):
            ii = rng.integers(0, len(Ym), len(Ym)); G = eoa(Fm[ii], .3, 'pw')
            bs.append(sum(wauc(G[:, J[c]], Ym[ii, i], w[ii]) - wauc(Fm[ii, J[c]], Ym[ii, i], w[ii]) for i, c in enumerate(OA)) / 12 * 1e4)
        rat = np.array([PUB[k] / out[k] if out[k] > 0 else np.nan for k in SH])
        P('%-6s %-32s ess %4d | %s | pw30 sd %.1f | public/ruler %s (mean %.2f)' % (mn, regime, ess, ' '.join('%s %+.1f' % (k, v) for k, v in out.items()),
          np.std(bs), np.round(rat, 2), np.nanmean(rat)))

P('')
s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); r7 = LD.r7(); d0 = load(); ids = d0['ids']
cov = [i for i in ids if i in s44.index and i in s45.index and i in r7.index]
P('gold58 ids covered by R7 + s1044 + s1045 OOF: %d of %d' % (len(cov), len(ids)))
if len(cov) >= 40:
    R7c = ['p|' + c for c in L]
    M58 = np.column_stack([np.mean([pct(X[:, j]) for X in (r7.loc[cov, R7c].values.astype(float), s44.loc[cov, L].values, s45.loc[cov, L].values)], 0) for j in range(12)])
    pos = [ids.index(i) for i in cov]; C4c = D['C4'][pos]; R7g = r7.loc[cov, R7c].values.astype(float)
    for a_, b_ in (('Lateral Meniscus', 'Medial Meniscus'), ('Lateral OA', 'Medial Meniscus')):
        f = lambda F: np.corrcoef(ns(F[:, J[a_]]), ns(F[:, J[b_]]))[0, 1]
        P('%-34s on the %d covered gold58 knees: c4 %.3f | R7 %.3f | 3-mean %.3f | (train: R7 %.3f, 3-mean %.3f)' % (
            a_ + ' ~ ' + b_, len(cov), f(C4c), f(R7g), f(M58), f(D['F7']), f(D['M3'])))
P('SKEP_STACK3 DONE')
