# box_eoa_ladder.py (10-20 more, Claude). Read-only, AGGREGATES ONLY. Robustness of the in-flight E-OA (OA trio decorrelation,
# out_t = pct(f_t) - a * (pct(f_o1) + pct(f_o2)) / 2) to the cartilage definition: raw OAI OA-mean AUC delta at a .1 / .2 for
# each MOAKS ladder level, per OAI-free model, with the trio's expert tetrachoric coupling at that level. No reweighting here
# (the skeptic's gold74-coupling transfer is the size estimate; this only checks the sign is not P_FT-specific).
import os, numpy as np, pandas as pd
from scipy.stats import rankdata, norm, multivariate_normal
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'], 'PF OA': ['mp', 'lp', 'amf', 'alf']}
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def tetra(a, b):
    c11 = ((a == 1) & (b == 1)).sum() + .5; c10 = ((a == 1) & (b == 0)).sum() + .5
    c01 = ((a == 0) & (b == 1)).sum() + .5; c00 = ((a == 0) & (b == 0)).sum() + .5; N = c11 + c10 + c01 + c00
    h = norm.ppf(1 - (c11 + c10) / N); k = norm.ppf(1 - (c11 + c01) / N); p11 = c11 / N
    f = lambda r: multivariate_normal.cdf([-h, -k], mean=[0, 0], cov=[[1, r], [r, 1]]) - p11
    if f(-.995) > 0: return -.995
    if f(.995) < 0: return .995
    return brentq(f, -.995, .995, xtol=1e-4)
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[OA]
P = {'D5': rd(H + '/teacher/teacher_D5_mean_v00.csv'), 's1044': rd(H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv'),
     's1045': rd(H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv')}
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
common = S.index
for k in P: common = common.intersection(P[k].dropna().index)
S = S.loc[common]
lev = {}
fin_all = np.ones(len(S), bool)
for c, regs in COMP.items():
    cc = S[['m_cm_' + r for r in regs]].astype(float).values; fin_all &= np.isfinite(cc).all(1)
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10)
    for nm, v in {'area1+': (area >= 1).any(1), 'area2+': (area >= 2).any(1), 'FT': (ft >= 1).any(1),
                  'area2&FT': ((area >= 2) & (ft >= 1)).any(1), 'FT2+': (ft >= 2).any(1)}.items():
        lev.setdefault(nm, {})[c] = v.astype(int)
ok = fin_all; print('knees with all OA subregions read and 3 model predictions: %d' % ok.sum())
rng = np.random.default_rng(3)
for nm, Yd in lev.items():
    Y = np.column_stack([Yd[c][ok] for c in OA]); n = len(Y)
    tets = [tetra(Y[:, i], Y[:, j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    line = '%-9s prev %s tet ML/MPF/LPF %s |' % (nm, '/'.join('%.2f' % v for v in Y.mean(0)), '/'.join('%+.2f' % v for v in tets))
    for k, Pk in P.items():
        Fm = Pk.loc[common[ok], OA].values; U = np.column_stack([pct(Fm[:, i]) for i in range(3)])
        res = []
        for a in (.1, .2):
            O = np.column_stack([U[:, i] - a * (U[:, [m for m in range(3) if m != i]].sum(1)) / 2 for i in range(3)])
            d = np.mean([auc(O[:, i], Y[:, i]) - auc(U[:, i], Y[:, i]) for i in range(3)])
            bs = []
            for _ in range(200):
                ii = rng.integers(0, n, n); bs.append(np.mean([auc(O[ii, i], Y[ii, i]) - auc(U[ii, i], Y[ii, i]) for i in range(3)]))
            res.append('a%.1f %+.4f (P %.2f)' % (a, d, np.mean(np.array(bs) > 0)))
        line += ' %s %s |' % (k, ' '.join(res))
    print(line)
