# adj_c12.py (10-20 night, Claude, NIGHT synthesis adjudication). RESEARCH ONLY, local CPU, NO OAI, RSNA train OOF + gold74 counts.
# Question: the stack skeptic found that the 3-model OOF mean (R7 + s1044 + s1045) on non-gold train, with report-v2 labels raked to the
# gold74 EXPERT OA-trio joint, reproduces the five public E-OA reads at ratio ~1.05 (skep_stack3.log), and that the same ruler gives
# C12' (LOA <- MM a .05 on pw30) +2.1e-4 macro (skep_stack4.log). Is that a property of the ruler family (each member, each seed) or of
# one averaged model? And is the LOA <- MM read specific (vs placebo partners) under the SAME raking?
# Per model on the same tr3 rows: (1) E-OA calibration (8-cell trio raking +.5): ruler vs public for eoa10/20/30, pw20/30;
# (2) C12' under the 16-cell (MOA, LOA, PF, MM) raking +.5: LOA <- MM a .05 / .10 on pw30, bootstrap sd; raw (unraked) read;
#     the same for 10 placebo partners, and the raking effect (raked - raw) for MM vs placebos;
# (3) C6' LM <- MM a .05 under the (MM, LM) 2x2 raking; (4) label-free co-firing rho_ns vs c4 on gold58.
# Writes adj_c12.log. Aggregates only.
import sys
import numpy as np
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
from cbase import load
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/adj_c12.log'
log = open(OUT, 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
L = LD.L; pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}
OA = ['Medial OA', 'Lateral OA', 'PF OA']; jOA = [J[c] for c in OA]
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
PUB = {'eoa10': 5.9, 'eoa20': 10.3, 'eoa30': 9.6, 'pw20': 10.4, 'pw30': 11.3}
SH = {'eoa10': (.1, 'sym'), 'eoa20': (.2, 'sym'), 'eoa30': (.3, 'sym'), 'pw20': (.2, 'pw'), 'pw30': (.3, 'pw')}
rng = np.random.default_rng(2110)


def eoa(F, a, mode):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]
        w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o] if mode == 'pw' else [1., 1.]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G


def line(F, t, o, a):
    G = F.copy(); G[:, J[t]] = pct(pct(F[:, J[t]]) - a * pct(F[:, J[o]])); return G


def wauc(s, y, w):
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    wp = w * y; wn = w * (1 - y)
    _, inv = np.unique(s, return_inverse=True)
    gp = np.bincount(inv, wp); gn = np.bincount(inv, wn); cn = np.cumsum(gn) - gn
    return float((gp * (cn + .5 * gn)).sum() / (wp.sum() * wn.sum()))


def rake(T, margins, it=300):
    T = T.copy(); k = T.ndim
    for _ in range(it):
        for ax in range(k):
            cur = T.sum(axis=tuple(i for i in range(k) if i != ax)); sh = [1] * k; sh[ax] = 2
            T = T * (np.array([1 - margins[ax], margins[ax]]) / cur).reshape(sh)
    return T / T.sum()


def weights(Ysub, cols, Y74):
    jc = [J[c] for c in cols]; ok = ~np.isnan(Y74[:, jc]).any(1); Yg = Y74[ok][:, jc].astype(int)
    T = np.zeros((2,) * len(cols)); obs = np.zeros((2,) * len(cols))
    for r in Yg: T[tuple(r)] += 1
    for r in Ysub: obs[tuple(r)] += 1
    obs /= obs.sum(); tgt = rake(T + .5, Ysub.mean(0)); w = (tgt / obs)[tuple(Ysub.T)]
    return w / w.mean()


d0 = load(); g74 = d0['g74']; allg = set(g74.StudyInstanceUID); Y74 = g74[L].values.astype(float)
r7 = LD.r7(); R7c = ['p|' + c for c in L]
s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2')
tr = [i for i in r7.index if i not in allg and i in lv2.index]
tr3 = [i for i in tr if i in s44.index and i in s45.index]
X = {'R7': r7.loc[tr3, R7c].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float), 's1045': s45.loc[tr3, L].values.astype(float)}
X['S6mean'] = np.column_stack([np.mean([pct(X[k][:, j]) for k in ('s1044', 's1045')], 0) for j in range(12)])
X['3mean'] = np.column_stack([np.mean([pct(X[k][:, j]) for k in ('R7', 's1044', 's1045')], 0) for j in range(12)])
Y = lv2.loc[tr3, L].values
P('rows tr3 %d (non-gold train with R7 + s1044 + s1045 OOF and v2 labels)' % len(tr3))

# c4 label-free references on gold58 (RSNA images, final model)
C4 = d0['c4']
def rho(F, a_, b_): return float(np.corrcoef(ns(F[:, J[a_]]), ns(F[:, J[b_]]))[0, 1])
P('c4 gold58 rho_ns: MOA~LOA %.3f | LOA~MM (pw30) %.3f | MM~LM %.3f' % (rho(C4, 'Medial OA', 'Lateral OA'), rho(eoa(C4, .3, 'pw'), 'Lateral OA', 'Medial Meniscus'),
                                                                   rho(C4, 'Medial Meniscus', 'Lateral Meniscus')))
PLAC = ['Lateral Meniscus', 'Medial OA', 'PF OA', 'ACL', 'MCL', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
for mn, F in X.items():
    P('')
    P('==== %s' % mn)
    P('  rho_ns train: MOA~LOA %.3f | LOA~MM (pw30) %.3f | MM~LM %.3f' % (rho(F, 'Medial OA', 'Lateral OA'), rho(eoa(F, .3, 'pw'), 'Lateral OA', 'Medial Meniscus'),
                                                                        rho(F, 'Medial Meniscus', 'Lateral Meniscus')))
    # (1) E-OA calibration
    m = ~np.isnan(Y[:, jOA]).any(1); Fm = F[m]; Ym = Y[m][:, jOA].astype(int)
    w = weights(Ym, OA, Y74); out = {}
    for k, (a, mode) in SH.items():
        G = eoa(Fm, a, mode)
        out[k] = sum(wauc(G[:, J[c]], Ym[:, i], w) - wauc(Fm[:, J[c]], Ym[:, i], w) for i, c in enumerate(OA)) / 12 * 1e4
    rat = [PUB[k] / out[k] if out[k] > 0 else np.nan for k in SH]
    P('  (1) E-OA trio-raked: %s | public/ruler %s' % (' '.join('%s %+.1f' % (k, v) for k, v in out.items()), np.round(rat, 2)))
    # (2) C12' under the 16-cell raking
    cols = OA + ['Medial Meniscus']; jc = [J[c] for c in cols]
    m = ~np.isnan(Y[:, jc]).any(1); Fm = F[m]; Ym = Y[m][:, jc].astype(int); w = weights(Ym, cols, Y74); w1 = np.ones(len(w))
    B30 = eoa(Fm, .3, 'pw'); jl = J['Lateral OA']; yl = Ym[:, 1]
    trio = sum(wauc(B30[:, J[c]], Ym[:, i], w) - wauc(Fm[:, J[c]], Ym[:, i], w) for i, c in enumerate(OA)) / 12 * 1e4
    res = []
    for a in (.05, .1):
        G = line(B30, 'Lateral OA', 'Medial Meniscus', a)
        dd = wauc(G[:, jl], yl, w) - wauc(B30[:, jl], yl, w); dr = wauc(G[:, jl], yl, w1) - wauc(B30[:, jl], yl, w1)
        bs = []
        for _ in range(150):
            ii = rng.integers(0, len(yl), len(yl)); bs.append(wauc(G[ii, jl], yl[ii], w[ii]) - wauc(B30[ii, jl], yl[ii], w[ii]))
        res.append('a %.2f raked %+.4f (sd %.4f, P>0 %.2f) = %+.2fe-4 macro | raw %+.4f' % (a, dd, np.std(bs), np.mean(np.array(bs) > 0), dd / 12 * 1e4, dr))
    P('  (2) 16-cell raked: pw30 trio vs raw %+.1fe-4 | LOA <- MM: %s' % (trio, ' || '.join(res)))
    G = line(B30, 'Lateral OA', 'Medial Meniscus', .05)
    mm_r = wauc(G[:, jl], yl, w) - wauc(B30[:, jl], yl, w); mm_0 = wauc(G[:, jl], yl, w1) - wauc(B30[:, jl], yl, w1)
    pr, p0 = [], []
    for Xp in PLAC:
        G = line(B30, 'Lateral OA', Xp, .05)
        pr.append(wauc(G[:, jl], yl, w) - wauc(B30[:, jl], yl, w)); p0.append(wauc(G[:, jl], yl, w1) - wauc(B30[:, jl], yl, w1))
    pr = np.array(pr); p0 = np.array(p0); eff = pr - p0
    P('      placebo LOA <- X a .05 raked: %s' % ' '.join('%s %+.4f' % (x[:4], v) for x, v in zip(PLAC, pr)))
    P('      MM raked %+.4f vs placebo mean %+.4f (max %+.4f, MM rank %d of 11) | raking effect MM %+.4f vs placebo mean %+.4f (max %+.4f)' % (
        mm_r, pr.mean(), pr.max(), 1 + (pr > mm_r).sum(), mm_r - mm_0, eff.mean(), eff.max()))
    # (3) C6' under the (MM, LM) raking
    cols2 = ['Medial Meniscus', 'Lateral Meniscus']; jc2 = [J[c] for c in cols2]
    m2 = ~np.isnan(Y[:, jc2]).any(1); F2 = F[m2]; Y2 = Y[m2][:, jc2].astype(int); w2 = weights(Y2, cols2, Y74)
    jm = J['Lateral Meniscus']; G = line(F2, 'Lateral Meniscus', 'Medial Meniscus', .05)
    dd = wauc(G[:, jm], Y2[:, 1], w2) - wauc(F2[:, jm], Y2[:, 1], w2)
    P('  (3) C6\' LM <- MM a .05, (MM, LM) raked: %+.4f = %+.2fe-4 macro' % (dd, dd / 12 * 1e4))
P('ADJ_C12 DONE')
