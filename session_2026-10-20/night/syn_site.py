# syn_site.py (10-20 night, Claude). LOCAL, RSNA gold58 only, aggregates. Pre-specified mechanism check for E2 (written after
# syn_more.py part A, before looking at any gold58 split): if report-trained Synovitis heads carry a site prior from languages that
# rarely write "synovitis" (bg / hr / tr / nl: v2 Syn YES .005-.072 vs en / es / de / el / fr .12-.19), then on gold58
# (a) expert Syn prevalence should NOT be lower in those low-report sites, (b) the Syn head should under-rank their expert positives
# relative to the Eff head, and (c) E2's gain should be larger there. One fixed split, no tuning.
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import cbase
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'
L = cbase.L; J = {c: i for i, c in enumerate(L)}; SYN, EFF = J['Synovitis'], J['Effusion']
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); r = rankdata(s); p = y.sum(); q = len(y) - p
    return np.nan if p == 0 or q == 0 else (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
d = cbase.load(); ids = d['ids']; Y = d['Y58']; ys, ye = Y[:, SYN], Y[:, EFF]
lang = pd.read_csv(R + '/_data/_reports_lang.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['lang'].reindex(ids).values
low = np.isin(lang, ['bg', 'hr', 'tr', 'nl'])
print('gold58 low-report sites n %d (Syn+ %d, Eff+ %d) | high-report n %d (Syn+ %d, Eff+ %d)' % (low.sum(), ys[low].sum(), ye[low].sum(), (~low).sum(), ys[~low].sum(), ye[~low].sum()))
M = {'c4 final': d['c4'], 'R6': d['members']['R6'], 'S6fall': d['members']['S6'], 'Raptor r384': d['members']['R384'], 'Raptor r224': d['members']['R224'],
     'R7 OOF': pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').loc[ids, ['p|' + c for c in L]].values}
rng = np.random.default_rng(5)
for k, P in M.items():
    s, e = pct(P[:, SYN]), pct(P[:, EFF])
    # (b) pooled-rank position of expert Syn+ studies, low vs high sites, Syn head vs Eff head
    pos = ys == 1
    line = '%-12s mean pooled rank-pct of expert Syn+: low sites Syn head %.3f Eff head %.3f | high sites Syn head %.3f Eff head %.3f' % (
        k, s[pos & low].mean(), e[pos & low].mean(), s[pos & ~low].mean(), e[pos & ~low].mean())
    negs = ys == 0
    line += ' || Syn- : low Syn %.3f Eff %.3f | high Syn %.3f Eff %.3f' % (s[negs & low].mean(), e[negs & low].mean(), s[negs & ~low].mean(), e[negs & ~low].mean())
    print(line)
    # (c) E2 gain decomposed by which pairs change: pairs with the positive in low vs high sites (pooled ranking)
    for a in (.1, .2):
        o = s + a * e
        P_, N_ = np.where(pos)[0], np.where(negs)[0]
        b = (s[P_][:, None] > s[N_][None, :]) + .5 * (s[P_][:, None] == s[N_][None, :])
        f = (o[P_][:, None] > o[N_][None, :]) + .5 * (o[P_][:, None] == o[N_][None, :])
        dm = f - b
        lp = low[P_]
        print('    a %.1f: total d %+.4f | from pairs with positive in LOW sites %+.4f (%d pos) | HIGH sites %+.4f (%d pos) | per-positive mean LOW %+.4f HIGH %+.4f' % (
            a, dm.sum() / dm.size, dm[lp].sum() / dm.size, lp.sum(), dm[~lp].sum() / dm.size, (~lp).sum(), dm[lp].sum() / dm.size / max(1, lp.sum()), dm[~lp].sum() / dm.size / max(1, (~lp).sum())))
print('-- room check: pooled vs within-site-group AUC, and an ORACLE site-centred score (subtract the group mean rank; uses the report language, which the test set does not have; info only)')
for k, P in M.items():
    s, e = pct(P[:, SYN]), pct(P[:, EFF])
    cen = s.copy()
    for gm in (low, ~low):
        cen[gm] = s[gm] - s[gm].mean()
    o2 = s + .2 * e
    print('%-12s pooled %.4f | within low %.4f high %.4f | oracle site-centred pooled %.4f (%+.4f) | E2 a .2 pooled %.4f (%+.4f), within low %.4f high %.4f' % (
        k, auc(s, ys), auc(s[low], ys[low]), auc(s[~low], ys[~low]), auc(cen, ys), auc(cen, ys) - auc(s, ys), auc(o2, ys), auc(o2, ys) - auc(s, ys),
        auc(o2[low], ys[low]), auc(o2[~low], ys[~low])))
