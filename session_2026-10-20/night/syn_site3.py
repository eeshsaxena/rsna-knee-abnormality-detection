# syn_site3.py (10-20 night, Claude). LOCAL, gold58, aggregates. Specificity of the site fingerprint: for every borrow helper h of the
# c4 / pw30 Synovitis column (out = pct(Syn) + a pct(h), a .1 / .2), split the gold58 AUC gain into pairs whose positive is in a
# low-report site (bg / hr / tr / nl, split fixed in syn_site.py) vs a high-report site. Also each helper's expert tetrachoric with Syn
# is in census_table.md (CLAIMED).
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import cbase
R = 'C:/Users/eeshs/Downloads/rsna-knee'
L = cbase.L; J = {c: i for i, c in enumerate(L)}; SYN = J['Synovitis']
pct = lambda v: rankdata(v) / len(v)
d = cbase.load(); ids = d['ids']; Y = d['Y58']; ys = Y[:, SYN]; F = d['c4']
lang = pd.read_csv(R + '/_data/_reports_lang.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['lang'].reindex(ids).values
low = np.isin(lang, ['bg', 'hr', 'tr', 'nl'])
s = pct(F[:, SYN]); P_, N_ = np.where(ys == 1)[0], np.where(ys == 0)[0]; lp = low[P_]
b = (s[P_][:, None] > s[N_][None, :]) + .5 * (s[P_][:, None] == s[N_][None, :])
print('helper  | a .1: total  low-pos  high-pos | a .2: total  low-pos  high-pos | low-site helper mean rank minus high-site (Syn+ / Syn-)')
for c in L:
    if c == 'Synovitis':
        continue
    h = pct(F[:, J[c]]); row = '%-16s' % c
    for a in (.1, .2):
        o = s + a * h
        f = (o[P_][:, None] > o[N_][None, :]) + .5 * (o[P_][:, None] == o[N_][None, :]); dm = f - b
        row += ' | %+.4f %+.4f %+.4f' % (dm.sum() / dm.size, dm[lp].sum() / dm.size, dm[~lp].sum() / dm.size)
    row += ' | %+.3f / %+.3f' % (h[(ys == 1) & low].mean() - h[(ys == 1) & ~low].mean(), h[(ys == 0) & low].mean() - h[(ys == 0) & ~low].mean())
    print(row)
print('Syn head itself: low minus high mean rank, Syn+ %+.3f / Syn- %+.3f' % (s[(ys == 1) & low].mean() - s[(ys == 1) & ~low].mean(), s[(ys == 0) & low].mean() - s[(ys == 0) & ~low].mean()))
