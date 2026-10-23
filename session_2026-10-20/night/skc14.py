# skc14.py (10-20 night, Claude, skeptic pass on C14 LOA += .1 LM on pw30). RESEARCH ONLY, local CPU, no OAI. Writes skc14.log.
import sys
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import auc, pct, L, tet
OUT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/skc14.log'
log = open(OUT, 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(1414)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}; OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
def pw(F, a=.3):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G
def extra(F, t, o, a):
    G = F.copy(); G[:, J[t]] = pct(pct(F[:, J[t]]) - a * pct(F[:, J[o]])); return G
T, O = 'Lateral OA', 'Lateral Meniscus'; jt, jo = J[T], J[O]
def coup(F, y):
    zt, zo = ns(F[:, jt]), ns(F[:, jo]); ct = np.corrcoef(zt, y)[0, 1]
    return np.corrcoef(zt, zo)[0, 1], np.corrcoef(zo, y)[0, 1] / ct
def dlt(F, y, a=-.1):
    return auc(extra(F, T, O, a)[:, jt], y) - auc(F[:, jt], y)
D = CL.data()
# 1. label coupling LOA-LM by label source
Y74 = D['Y74']; Y58 = D['Y58']
P('label tet LOA-LM: gold74 %.3f | gold58 %.3f | report v2 (train) %.3f | report v4 %.3f' % (
    tet(Y74[:, jt], Y74[:, jo]), tet(Y58[:, jt], Y58[:, jo]),
    tet(D['Yv2'][~np.isnan(D['Yv2'][:, [jt, jo]]).any(1)][:, jt], D['Yv2'][~np.isnan(D['Yv2'][:, [jt, jo]]).any(1)][:, jo]),
    tet(D['Yv4'][~np.isnan(D['Yv4'][:, [jt, jo]]).any(1)][:, jt], D['Yv4'][~np.isnan(D['Yv4'][:, [jt, jo]]).any(1)][:, jo])))
P('gold58 LOA pos %d of %d; gold74 LOA pos %d of %d' % (Y58[:, jt].sum(), len(Y58), np.nansum(Y74[:, jt]), len(Y74)))
# 2. report-domain coupling and borrow delta, with bootstrap sd (in-scope ruler for this pair)
for nm, F, Yr in (('R7 x report v2', D['F7'], D['Yv2']), ('R7 x report v4', D['F7'], D['Yv4']), ('3-mean x report v2', D['M3'], D['Y3v2']), ('3-mean x report v4', D['M3'], D['Y3v4'])):
    ok = ~np.isnan(F[:, [J[c] for c in OA]]).any(1) & ~np.isnan(Yr[:, jt]) & ~np.isnan(F[:, jo])
    G = pw(F[ok]); y = Yr[ok, jt].astype(int); r, q = coup(G, y)
    d = {a: dlt(G, y, a) for a in (-.1, -.05)}
    bs = []; gb = []
    for _ in range(300):
        ii = rng.integers(0, len(y), len(y)); bs.append(dlt(G[ii], y[ii]))
        rr, qq = coup(G[ii], y[ii]); gb.append(rr - qq)
    P('%-20s n %d pos %d | LOA AUC %.4f | rho %.3f q %.3f gap %+.3f (sd %.3f) | borrow a-.10 %+.4f (boot sd %.4f, P>0 %.2f) a-.05 %+.4f' % (
        nm, len(y), y.sum(), auc(G[:, jt], y), r, q, r - q, np.std(gb), d[-.1], np.std(bs), np.mean(np.array(bs) > 0), d[-.05]))
# 3. gold58: c4 vs R7 vs each member, coupling and delta at a -.1 on the pw30-applied base
B30 = pw(D['C4']); y = Y58[:, jt]
r, q = coup(B30, y); P('c4 pw30 gold58: LOA AUC %.4f rho %.3f q %.3f gap %+.3f d(a-.1) %+.4f' % (auc(B30[:, jt], y), r, q, r - q, dlt(B30, y)))
R7_58 = D['R7_58']; ok = ~np.isnan(R7_58).any(1)
G = pw(R7_58[ok]); r, q = coup(G, y[ok]); P('R7 pw30 gold58 (n %d): LOA AUC %.4f rho %.3f q %.3f gap %+.3f d(a-.1) %+.4f' % (ok.sum(), auc(G[:, jt], y[ok]), r, q, r - q, dlt(G, y[ok])))
for k, F in D['members'].items():
    G = pw(np.asarray(F, float)); r, q = coup(G, y)
    P('member %-5s pw30 gold58: LOA AUC %.4f rho %.3f q %.3f gap %+.3f d(a-.1) %+.4f d(a-.05) %+.4f' % (k, auc(G[:, jt], y), r, q, r - q, dlt(G, y), dlt(G, y, -.05)))
# 4. gold74 minus gold58 (the 16 extra knees): independent part of the R7 x gold74 read
ids58 = set(D['ids']); g74 = D['ids'] and CL.load()['ids74']
m16 = np.array([u not in ids58 for u in g74]); R74 = D['R7_74']; okk = m16 & ~np.isnan(R74).any(1) & ~np.isnan(Y74[:, jt])
y16 = Y74[okk, jt].astype(int)
P('gold74 minus gold58: n %d, LOA pos %d' % (okk.sum(), y16.sum()))
if 0 < y16.sum() < len(y16):
    G = pw(R74[okk]); P('  R7 pw30 on the extra knees: d(a-.1) %+.4f' % dlt(G, y16))
okk58 = ~m16 & ~np.isnan(R74).any(1) & ~np.isnan(Y74[:, jt]); G = pw(R74[okk58]); P('  R7 pw30 on the gold58 part of gold74: d(a-.1) %+.4f' % dlt(G, Y74[okk58, jt].astype(int)))
# 5. gold58 leave-one-out and which pairs move
d0 = dlt(B30, y); loo = np.array([dlt(np.delete(B30, i, 0), np.delete(y, i)) for i in range(len(y))])
P('gold58 c4 d(a-.1) %+.4f; LOO range [%+.4f, %+.4f]; LOO sign flips to <=0 in %d of %d deletions' % (d0, loo.min(), loo.max(), (loo <= 0).sum(), len(y)))
G = extra(B30, T, O, -.1); s0 = B30[:, jt]; s1 = G[:, jt]; pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
gain = sum(1 for i in pos for k in neg if (s0[i] <= s0[k]) and (s1[i] > s1[k])); loss = sum(1 for i in pos for k in neg if (s0[i] > s0[k]) and (s1[i] <= s1[k]))
P('  pair flips: %d gained, %d lost (net %+d of %d)' % (gain, loss, gain - loss, len(pos) * len(neg)))
P('SKC14 DONE')
