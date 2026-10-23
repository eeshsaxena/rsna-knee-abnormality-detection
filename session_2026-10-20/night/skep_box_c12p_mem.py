# skep_box_c12p_mem.py (10-20 night, Claude, SKEPTIC on C12'): same as skep_box_c12p.py but with c4's own members R6 / S6fall served on OAI V00 (~/r5_oai) plus a c4-shaped blend (D5 in the Raptor slot). Read-only CPU; prints AGGREGATES ONLY (per model x regime means over
# weighted full-size resamples). Never writes a file, never prints an id, a row or a per-knee value. Piped via ssh stdin into python -I.
# Question (recal.md sec 5, item 2, not done there): is LOA <- MM worth the same ON TOP OF the pw30 E-OA block as standalone?
# 4-label IPF (MOA, LOA, PF, MM): OAI marginals; trio pairs at gold74 log OR + sh_t sd (sh_t -0.83 = the public gap-matched regime,
# CLAIMED recal.md / implied2; also 0); MM-LOA at gold74 log OR + k sd; MM-MOA, MM-PF at gold74. gold74 2x2 counts are local aggregates.
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
OA = ['Medial OA', 'Lateral OA', 'PF OA']; LB4 = OA + ['Medial Meniscus']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'], 'PF OA': ['mp', 'lp', 'amf', 'alf']}
MODELS = {"D5": H + "/teacher/teacher_D5_mean_v00.csv", "R6": os.path.expanduser("~") + "/r5_oai/run_r6/worker0/predictions.csv", "S6fall": os.path.expanduser("~") + "/r5_oai/s6_r5/out/predictions.csv"}
# (i, j): counts [i0j0, i0j1, i1j0, i1j1] on gold74, index into LB4; tag 't' = trio shift, 'k' = MM-LOA shift, '0' = fixed
G4 = {(0, 1): ([45, 5, 12, 9], 't'), (0, 2): ([36, 14, 6, 16], 't'), (1, 2): ([37, 20, 5, 9], 't'),
      (3, 1): ([30, 7, 26, 7], 'k'), (3, 0): ([33, 4, 16, 18], '0'), (3, 2): ([26, 12, 16, 18], '0')}
WPW = {(0, 1): .876, (0, 2): .708, (1, 2): 1.416}
B = int(os.environ.get('B', '80'))
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([[1 - p1 - p2 + p11, p2 - p11], [p1 - p11, p11]])


def weights(Y, sh_t, k):
    n = len(Y); key = Y[:, 0] * 8 + Y[:, 1] * 4 + Y[:, 2] * 2 + Y[:, 3]; p0 = np.bincount(key, minlength=16) / n
    if sh_t is None: return np.ones(n)
    cells = np.array([[(c >> (3 - i)) & 1 for i in range(4)] for c in range(16)]); marg = Y.mean(0); q = p0.copy(); tg = {}
    for (i, j), (c, tag) in G4.items():
        c = np.array(c, float) + .5; s = {'t': sh_t, 'k': k, '0': 0.}[tag]
        tg[(i, j)] = or_cells(marg[i], marg[j], np.log(c[3] * c[0] / (c[1] * c[2])) + s * np.sqrt((1 / c).sum()))
    for _ in range(400):
        for (i, j), t in tg.items():
            for a in (0, 1):
                for b in (0, 1):
                    m = (cells[:, i] == a) & (cells[:, j] == b); s = q[m].sum()
                    if s > 0: q[m] *= t[a, b] / s
        q /= q.sum()
    return np.where(p0 > 0, q / np.maximum(p0, 1e-12), 0)[key]


T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
Yd = pd.DataFrame(index=T.index)
for c in OA:
    cc = S.reindex(T.index)[['m_cm_' + r for r in COMP[c]]].astype(float).values
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
    y = ((ft >= 1) & fin).any(1).astype(float); y[~fin.any(1)] = np.nan; Yd[c] = y
Yd['Medial Meniscus'] = np.where(T['Medial Meniscus'].notna(), (T['Medial Meniscus'] >= 1 - 1e-6).astype(float), np.nan)
Yd = Yd.dropna()
Pm = {}
for k_, f in MODELS.items():
    d = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID'); Pm[k_] = d[LB4]
ids = Yd.index
for k_ in Pm: ids = ids.intersection(Pm[k_].dropna().index)
pc = lambda v: rankdata(v) / len(v)
C4W = {'Lateral OA': .2, 'PF OA': .5, 'Medial OA': .5, 'Medial Meniscus': .5}
Pm['c4D5'] = pd.DataFrame({c: .8 * pc(C4W[c] * pc(Pm['D5'].loc[ids, c].values) + (1 - C4W[c]) * pc(Pm['R6'].loc[ids, c].values)) + .2 * pc(Pm['S6fall'].loc[ids, c].values) for c in LB4}, index=ids)
Y = Yd.loc[ids, LB4].values.astype(int); n = len(Y)
print('n %d | prevalence MOA %.3f LOA %.3f PF %.3f MM %.3f | B %d' % ((n,) + tuple(Y.mean(0)) + (B,)), flush=True)
rng = np.random.default_rng(1021)
REGS = [('raw', None, None)] + [('t%+.2f k%+.1f' % (st, kk), st, kk) for st in (-0.83, 0.0) for kk in (-1.0, -0.5, 0.0, 0.5, 1.0)]
for rn, st, kk in REGS:
    w = weights(Y, st, kk); p = w / w.sum(); ess = 1 / (p ** 2).sum()
    for m, Pd in Pm.items():
        F = Pd.loc[ids, LB4].values.astype(float); acc = []
        for _ in range(B):
            ii = rng.choice(n, n, p=p); y = Y[ii, 1]
            if y.sum() < 10: continue
            U = np.column_stack([pct(F[ii, c]) for c in range(4)])
            GL = pct(U[:, 1] - .15 * (WPW[(0, 1)] * U[:, 0] + WPW[(1, 2)] * U[:, 2]))
            a0 = auc(U[:, 1], y); ap = auc(GL, y); r = [ap - a0]
            for a in (.05, .10):
                r.append(auc(pct(GL - a * U[:, 3]), y) - ap)
                r.append(auc(pct(U[:, 1] - a * U[:, 3]), y) - a0)
            zt, zo, zl = ns(GL), ns(U[:, 3]), ns(U[:, 1])
            rho_p = np.corrcoef(zt, zo)[0, 1]; q_p = np.corrcoef(zo, y)[0, 1] / np.corrcoef(zt, y)[0, 1]
            rho_s = np.corrcoef(zl, zo)[0, 1]; q_s = np.corrcoef(zo, y)[0, 1] / np.corrcoef(zl, y)[0, 1]
            phi = np.corrcoef(Y[ii, 1], Y[ii, 3])[0, 1]
            acc.append(r + [rho_p, q_p, rho_s, q_s, phi])
        X = np.array(acc).mean(0); sd = np.array(acc).std(0)
        print('%-14s %-7s ess %4d | LOA pw30 row %+.4f | +LOA<-MM on pw30: a.05 %+.4f (sd %.4f) a.10 %+.4f | standalone: a.05 %+.4f a.10 %+.4f | pw30 base rho %.3f q %.3f gap %+.3f | standalone rho %.3f q %.3f gap %+.3f | label phi %.3f' % (
            rn, m, ess, X[0], X[1], sd[1], X[3], X[2], X[4], X[5], X[6], X[5] - X[6], X[7], X[8], X[7] - X[8], X[9]), flush=True)
print('DONE')
