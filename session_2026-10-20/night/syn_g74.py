# syn_g74.py (10-20 night, Claude). LOCAL, aggregates. gold74 (58 expert + 16 our reads under the gold convention) member heads vs
# gold Synovitis, for the members that have all 74 rows out of their own training: R7 OOF, our fleetOF mean. Plus E2 a .1 / .2.
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
g = g[g.Synovitis.notna()]
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
M = {'R7 OOF': r7[['p|' + c for c in L]].set_axis(L, axis=1),
     'fleetOF mean': pd.read_csv(R + '/session_2026-10-19/think/models_work/gold_fleetOF_mean.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L]}
rng = np.random.default_rng(3)
for sub, msk in (('gold74', np.ones(len(g), bool)), ('extra16 only', (g.source != 'original_gold58').values)):
    y = g.Synovitis.values[msk].astype(int)
    for k, P in M.items():
        P = P.loc[g.index[msk]]
        s, e, b = P.Synovitis.values, P.Effusion.values, P["Baker's"].values
        out = []
        for a in (.1, .2):
            o = pct(s) + a * pct(e); dd = auc(o, y) - auc(s, y)
            bd = [auc(o[ii], y[ii]) - auc(s[ii], y[ii]) for ii in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if 0 < y[ii].sum() < len(y)]
            out.append('a %.1f %+.4f (P %.2f)' % (a, dd, np.mean(np.array(bd) > 0)))
        print('%-13s %-13s n %d pos %d | Syn head %.3f Eff head %.3f Bak head %.3f rho(S,E) %.2f | E2 %s' % (sub, k, len(y), y.sum(), auc(s, y), auc(e, y), auc(b, y), spearmanr(s, e).correlation, ' ; '.join(out)))
