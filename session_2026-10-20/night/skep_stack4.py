# skep_stack4.py (10-20 night, Claude, SKEPTIC). RESEARCH ONLY, local CPU, NO OAI. Writes skep_stack4.log.
# (1) Which RSNA ruler model is c4-like for each pair? Paired bootstrap of c4 - R7 normal-score rho on gold58 (MM-LM raw; LOA-MM with
#     pw30 applied to both; MOA-LOA raw as the trio reference where the 3-mean ruler fits the public).
# (2) Joint check of the C12' line: rake gold74's 16-cell (MOA, LOA, PF, MM) expert joint (+.5) to report marginals, then read
#     pw30 vs raw (trio, macro e-4) and the LOA <- MM a .05 / .1 increment ON the pw30 base, R7 and 3-mean. Also LM <- MM with the
#     (MM, LM) pair raked, and the stack (both lines) as macro e-4.
import sys
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night')
import census_lib as CL
from census_lib import auc, pct, L
OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
log = open(OUTD + 'skep_stack4.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(4)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}
OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}


def pw(F, a=.3):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o]
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


D = CL.data(); C4 = D['C4']; R58 = D['R7_58']
P('== (1) paired c4 - R7 rho on gold58 (bootstrap 2000)')
n = len(C4); BI = [rng.integers(0, n, n) for _ in range(2000)]
for a_, b_, tf in (('Lateral Meniscus', 'Medial Meniscus', None), ('Lateral OA', 'Medial Meniscus', 'pw30'), ('Medial OA', 'Lateral OA', None)):
    Fc = pw(C4) if tf else C4; Fr = pw(R58) if tf else R58
    za, zb, ra, rb = ns(Fc[:, J[a_]]), ns(Fc[:, J[b_]]), ns(Fr[:, J[a_]]), ns(Fr[:, J[b_]])
    d0 = np.corrcoef(za, zb)[0, 1] - np.corrcoef(ra, rb)[0, 1]
    bb = np.array([np.corrcoef(za[ii], zb[ii])[0, 1] - np.corrcoef(ra[ii], rb[ii])[0, 1] for ii in BI])
    Ft = pw(D['F7']) if tf else D['F7']; Mt = pw(D['M3']) if tf else D['M3']
    rt = np.corrcoef(ns(Ft[:, J[a_]]), ns(Ft[:, J[b_]]))[0, 1]; mt = np.corrcoef(ns(Mt[:, J[a_]]), ns(Mt[:, J[b_]]))[0, 1]
    P('%-34s %s| gold58 c4 %.3f R7 %.3f diff %+.3f (sd %.3f, P(c4>R7) %.2f) | train: R7 %.3f 3-mean %.3f (3-mean - R7 %+.3f)' % (
        a_ + ' ~ ' + b_, '(pw30 both) ' if tf else '', np.corrcoef(za, zb)[0, 1], np.corrcoef(ra, rb)[0, 1], d0, bb.std(), (bb > 0).mean(), rt, mt, mt - rt))

P('')
P('== (2) joint raked RSNA ruler (report v2 train labels moved to gold74 expert joint)')
Y74 = D['Y74']
def g74_joint(cols):
    ok = ~np.isnan(Y74[:, [J[c] for c in cols]]).any(1); Yg = Y74[ok][:, [J[c] for c in cols]].astype(int)
    T = np.zeros((2,) * len(cols))
    for r in Yg: T[tuple(r)] += 1
    return T, ok.sum()
for mn, (F, Y) in {'R7': (D['F7'], D['Yv2']), '3mean': (D['M3'], D['Y3v2'])}.items():
    # C12' with (MOA, LOA, PF, MM) raked
    cols = OA + ['Medial Meniscus']; jc = [J[c] for c in cols]
    m = ~np.isnan(Y[:, jc]).any(1); Fm = F[m]; Ym = Y[m][:, jc].astype(int)
    T, n74 = g74_joint(cols); marg = Ym.mean(0)
    obs = np.zeros((2,) * 4)
    for r in Ym: obs[tuple(r)] += 1
    obs /= obs.sum()
    for sm in (.5, 1.):
        tgt = rake(T + sm, marg); w = (tgt / obs)[tuple(Ym.T)]; w = w / w.mean(); ess = w.sum() ** 2 / (w ** 2).sum()
        B30 = pw(Fm); jl = J['Lateral OA']; yl = Ym[:, 1]
        trio = sum(wauc(B30[:, J[c]], Ym[:, i], w) - wauc(Fm[:, J[c]], Ym[:, i], w) for i, c in enumerate(OA)) / 12 * 1e4
        res = []
        for a in (.05, .1):
            G = line(B30, 'Lateral OA', 'Medial Meniscus', a); dd = wauc(G[:, jl], yl, w) - wauc(B30[:, jl], yl, w)
            bs = []
            for _ in range(150):
                ii = rng.integers(0, len(yl), len(yl)); bs.append(wauc(G[ii, jl], yl[ii], w[ii]) - wauc(B30[ii, jl], yl[ii], w[ii]))
            res.append('a %.2f LOA d %+.4f (sd %.4f, P>0 %.2f) = %+.2fe-4 macro' % (a, dd, np.std(bs), np.mean(np.array(bs) > 0), dd / 12 * 1e4))
        P('%-6s C12 joint (MOA,LOA,PF,MM) raked to gold74 (n %d, +%.1f) ess %4d | pw30 trio vs raw %+.1fe-4 | %s' % (mn, n74, sm, ess, trio, ' | '.join(res)))
    # C6' with (MM, LM) raked (pairwise, same as skep_stack2 k 0) for the stack sum
    cols2 = ['Medial Meniscus', 'Lateral Meniscus']; jc2 = [J[c] for c in cols2]
    m2 = ~np.isnan(Y[:, jc2]).any(1); F2 = F[m2]; Y2 = Y[m2][:, jc2].astype(int)
    T2, n2 = g74_joint(cols2); obs2 = np.zeros((2, 2))
    for r in Y2: obs2[tuple(r)] += 1
    obs2 /= obs2.sum()
    tgt2 = rake(T2 + .5, Y2.mean(0)); w2 = (tgt2 / obs2)[tuple(Y2.T)]; w2 = w2 / w2.mean()
    jm = J['Lateral Meniscus']; ylm = Y2[:, 1]; G = line(F2, 'Lateral Meniscus', 'Medial Meniscus', .05)
    dd = wauc(G[:, jm], ylm, w2) - wauc(F2[:, jm], ylm, w2)
    bs = []
    for _ in range(150):
        ii = rng.integers(0, len(ylm), len(ylm)); bs.append(wauc(G[ii, jm], ylm[ii], w2[ii]) - wauc(F2[ii, jm], ylm[ii], w2[ii]))
    P('%-6s C6\' LM <- MM a .05, (MM, LM) raked to gold74 (n %d): LM d %+.4f (sd %.4f, P>0 %.2f) = %+.2fe-4 macro' % (mn, n2, dd, np.std(bs), np.mean(np.array(bs) > 0), dd / 12 * 1e4))
P('SKEP_STACK4 DONE')
