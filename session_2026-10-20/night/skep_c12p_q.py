# skep_c12p_q.py (10-20 night, Claude, SKEPTIC C12'): paired expert-vs-report q on the same knees and models, then a train-like
# population gap = rho_train - (q_report_train + dq_paired). pw30 base throughout. Local only, aggregates.
import sys
import numpy as np
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import pct, L
import load as LD
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/skep_c12p_q.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}; OA = ['Medial OA', 'Lateral OA', 'PF OA']
W = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
def pw(F, a=.3):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; w = [W[tuple(sorted((c, k), key=OA.index))] for k in o]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G
T, O = J['Lateral OA'], J['Medial Meniscus']
def rq(G, y):
    zt, zo = ns(G[:, T]), ns(G[:, O]); return np.corrcoef(zt, zo)[0, 1], np.corrcoef(zo, y)[0, 1] / np.corrcoef(zt, y)[0, 1]
D = CL.data(); g74, _ = LD.gold(); ids74 = g74.StudyInstanceUID.tolist(); ids58 = D['ids']
labs = {s: LD.labels(s) for s in ('v2', 'v4', 'sol56')}
rng = np.random.default_rng(3)
def paired(nm, F, ids, Yexp):
    G = pw(F); out = []
    for s, lab in labs.items():
        yr = lab.reindex(ids)[L[T]].values; m = ~np.isnan(yr) & ~np.isnan(Yexp) & ~np.isnan(G).any(1)
        ye = Yexp[m].astype(int); yr_ = yr[m].astype(int); Gm = G[m]
        r_, qe = rq(Gm, ye); _, qr = rq(Gm, yr_)
        bs = []
        for _ in range(1000):
            ii = rng.integers(0, m.sum(), m.sum())
            if ye[ii].min() == ye[ii].max() or yr_[ii].min() == yr_[ii].max(): continue
            bs.append(rq(Gm[ii], ye[ii])[1] - rq(Gm[ii], yr_[ii])[1])
        P('%-10s %-5s n %d | rho %.3f | q expert %+.3f q report %+.3f | dq (exp - rep) %+.3f (boot sd %.3f)' % (nm, s, m.sum(), r_, qe, qr, qe - qr, np.std(bs)))
        out.append((s, qe - qr, np.std(bs)))
    return out
dq7 = paired('R7 gold74', D['R7_74'], ids74, D['Y74'][:, T])
dqc = paired('c4 gold58', D['C4'], ids58, D['Y58'][:, T].astype(float))
F7 = D['F7']; r7 = LD.r7(); allg = set(ids74)
tr = [i for i in r7.index if i not in allg]
X = r7.loc[tr, ['p|' + c for c in L]].values.astype(float); ok = ~np.isnan(X).any(1); X = X[ok]; tr = [u for u, k in zip(tr, ok) if k]
G = pw(X)
for s, lab in labs.items():
    y = lab.reindex(tr)[L[T]].values; m = ~np.isnan(y)
    r_, qr = rq(G[m], y[m].astype(int))
    d7 = [d for ss, d, _ in dq7 if ss == s][0]; dc = [d for ss, d, _ in dqc if ss == s][0]
    P('R7 train   %-5s n %d | rho %.3f q report %.3f gap_report %+.3f | train-like expert gap via R7-gold74 dq %+.3f, via c4-gold58 dq %+.3f' % (
        s, m.sum(), r_, qr, r_ - qr, r_ - (qr + d7), r_ - (qr + dc)))
