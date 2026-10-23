# skep_c12p_rw.py (10-20 night, Claude, SKEPTIC C12'): gold58 / gold74 are trauma-enriched (report v4 ACL .50 vs .21 train, Frac .36 vs
# .07). Rake the gold knees to train-like composition on report-v4 labels (ACL, MCL, LM, Eff, Cont, Frac) and recompute LOA-MM rho_ns,
# expert q, gap and the LOA <- MM a .05 / .10 delta on the pw30 base, for c4 (gold58) and R7 (gold74; validation: does R7's raked rho
# move to its train value .326?). Local only.
import sys
import numpy as np
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import pct, L
import load as LD
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/skep_c12p_rw.log', 'w')
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
def wcorr(a, b, w):
    w = w / w.sum(); ma, mb = (w * a).sum(), (w * b).sum()
    return (w * (a - ma) * (b - mb)).sum() / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
def wauc(s, y, w):
    pos, neg = y == 1, y == 0; sp, sn, wp, wn = s[pos], s[neg], w[pos], w[neg]
    M = (sp[:, None] > sn[None, :]) + .5 * (sp[:, None] == sn[None, :])
    return (wp[:, None] * wn[None, :] * M).sum() / (wp.sum() * wn.sum())
def ext(G, a):
    H = G.copy(); H[:, T] = pct(pct(G[:, T]) - a * pct(G[:, O])); return H
RK = ['ACL', 'MCL', 'Lateral Meniscus', 'Effusion', 'Contusion', 'Fracture']
lv4 = LD.labels('v4'); g74, _ = LD.gold(); allg = set(g74.StudyInstanceUID)
tr = [u for u in lv4.index if u not in allg]; tgt = np.array([np.nanmean(lv4.loc[tr, c].values) for c in RK])
def rake(ids):
    X = lv4.reindex(ids)[RK].values; X = np.where(np.isnan(X), 0, X); w = np.ones(len(ids))
    for _ in range(300):
        for k in range(len(RK)):
            p = (w * X[:, k]).sum() / w.sum(); t = tgt[k]
            w = np.where(X[:, k] == 1, w * t / max(p, 1e-9), w * (1 - t) / max(1 - p, 1e-9))
    w = np.clip(w / w.mean(), 0, 6); return w / w.mean()
D = CL.data()
for nm, F, ids, y in (('c4 gold58', D['C4'], D['ids'], D['Y58'][:, T].astype(float)), ('R7 gold74', D['R7_74'], g74.StudyInstanceUID.tolist(), D['Y74'][:, T]),
                      ('R7 gold58', D['R7_58'], D['ids'], D['Y58'][:, T].astype(float))):
    m = ~np.isnan(y) & ~np.isnan(F).any(1); F = F[m]; y = y[m].astype(int); ids = [u for u, k in zip(ids, m) if k]
    G = pw(F); w1 = np.ones(len(ids)); w2 = rake(ids); ess = w2.sum() ** 2 / (w2 ** 2).sum()
    for wn, w in (('flat', w1), ('raked', w2)):
        zt, zo = ns(G[:, T]), ns(G[:, O]); r = wcorr(zt, zo, w); q = wcorr(zo, y, w) / wcorr(zt, y, w)
        b = wauc(G[:, T], y, w); d05 = wauc(ext(G, .05)[:, T], y, w) - b; d10 = wauc(ext(G, .10)[:, T], y, w) - b
        comp = ' '.join('%s %.2f' % (CL.AB[c], (w * np.nan_to_num(lv4.reindex(ids)[c].values)).sum() / w.sum()) for c in RK)
        P('%-10s %-5s n %d ess %.0f | LOA prev %.2f | rho_ns %.3f q_exp %+.3f gap %+.3f | LOA<-MM a.05 %+.4f a.10 %+.4f | %s' % (
            nm, wn, len(ids), ess if wn == 'raked' else len(ids), (w * y).sum() / w.sum(), r, q, r - q, d05, d10, comp))
P('train targets: ' + ' '.join('%s %.2f' % (CL.AB[c], t) for c, t in zip(RK, tgt)))
