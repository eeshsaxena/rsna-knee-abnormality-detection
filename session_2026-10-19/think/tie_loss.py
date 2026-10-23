# tie_loss.py (10-19, Claude, research only): how much ranking information does the float32-saturated probability output of the
# heliosli compact family (R6 / R7) throw away at the top? Measured on R7 OOF (4,349 non-gold train studies, same worker family,
# p = float32 sigmoid of the fold-mean logit z) against LLM report verdicts (report_labels_v2 YES / NO, UNK dropped).
# Single-leg AUC with p-ranks (average ties, as the kernel's pct) vs z-ranks, plus a 2-leg blend with a synthetic partner.
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
d = pd.read_csv(R + '/session_2026-10-18/next/g58/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
g = pd.read_csv(R + '/session_2026-10-18/next/g58/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
lab = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
d = d.loc[~d.index.isin(set(g.StudyInstanceUID))]
lab = lab.reindex(d.index)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


rng = np.random.default_rng(3)
tot = []
print('label | n_lab pos | cells in p-ties (all rows) | AUC p-rank | AUC z-rank | diff | sat group n (pos/neg) | within-sat AUC of z | '
      '2-leg blend diff (partner rho .85)')
for c in L:
    v = lab[c + '__verdict'].values; m = np.isin(v, ['YES', 'NO'])
    y = (v[m] == 'YES').astype(int)
    p = d['p|' + c].values[m]; z = d['z|' + c].values[m]
    u, cnt = np.unique(d['p|' + c].values, return_counts=True); tied = int(cnt[cnt > 1].sum())
    ap, az = auc(p, y), auc(z, y)
    sat = p >= 1 - 1e-6
    wa = auc(z[sat], y[sat]) if 0 < y[sat].sum() < sat.sum() else np.nan
    # partner: a second leg correlated with z at Spearman ~.85 (Gaussian copula on z-ranks), blended .5 / .5 by rank
    zr = (rankdata(z) - .5) / len(z); from scipy.stats import norm
    gz = norm.ppf(zr); part = .85 * gz + np.sqrt(1 - .85 ** 2) * rng.standard_normal(len(z))
    bp = .5 * rankdata(p) + .5 * rankdata(part); bz = .5 * rankdata(z) + .5 * rankdata(part)
    db = auc(bz, y) - auc(bp, y)
    tot.append((c, az - ap, db))
    print('%-17s | %5d %4d | %4d | %.5f | %.5f | %+.5f | %4d (%d/%d) | %s | %+.5f' % (
        c, m.sum(), y.sum(), tied, ap, az, az - ap, sat.sum(), y[sat].sum(), sat.sum() - y[sat].sum(),
        'nan' if np.isnan(wa) else '%.3f' % wa, db))
print('macro single-leg gain z over p: %+.5f ; macro 2-leg blend gain: %+.5f' % (np.mean([t[1] for t in tot]), np.mean([t[2] for t in tot])))
