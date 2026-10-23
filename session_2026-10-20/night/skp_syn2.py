# skp_syn2.py (10-20 night, Claude, SKEPTIC). LOCAL gold58 only, aggregates. Is c4's class-conditional low-site offset in
# (Syn pct - Eff pct) a site prior or the trauma (ACL) case mix of the low-site knees? OLS dd ~ class + low (+ expert ACL), and
# a within-class / within-ACL-stratum permutation of the site label.
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import cbase, io, contextlib
R = 'C:/Users/eeshs/Downloads/rsna-knee'
pct = lambda v: rankdata(v) / len(v)
with contextlib.redirect_stdout(io.StringIO()):
    d = cbase.load()
ids = d['ids']; Y = d['Y58']; L = cbase.L; J = {c: i for i, c in enumerate(L)}
ys = Y[:, J['Synovitis']]; acl = Y[:, J['ACL']]
lang = pd.read_csv(R + '/_data/_reports_lang.csv', dtype={'StudyInstanceUID': str}, usecols=['StudyInstanceUID', 'lang']).set_index('StudyInstanceUID')['lang']
lw = np.isin(lang.reindex(ids).values, ['bg', 'hr', 'tr', 'nl'])
rng = np.random.default_rng(13)
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
for k, P in (('c4', d['c4']), ('R6', d['members']['R6']), ('R384', d['members']['R384'])):
    dd = pct(P[:, J['Synovitis']]) - pct(P[:, J['Effusion']]); aclh = pct(P[:, J['ACL']])
    for nm, X in (('class+low', np.c_[np.ones(58), ys, lw]), ('class+low+expertACL', np.c_[np.ones(58), ys, lw, acl]),
                  ('class+low+modelACL', np.c_[np.ones(58), ys, lw, aclh])):
        b = np.linalg.lstsq(X, dd, rcond=None)[0]
        print('%-5s %-20s low coef %+.3f%s' % (k, nm, b[2], ('  ACL coef %+.3f' % b[3]) if X.shape[1] > 3 else ''))
    strata = [(ys == c) & (acl == a) for c in (0, 1) for a in (0, 1)]
    stat = lambda w: np.nanmean([dd[s & w].mean() - dd[s & ~w].mean() if (s & w).sum() and (s & ~w).sum() else np.nan for s in strata])
    obs = stat(lw); null = []
    for _ in range(20000):
        w = lw.copy()
        for s in strata:
            ii = np.where(s)[0]; w[ii] = rng.permutation(w[ii])
        null.append(stat(w))
    null = np.array(null)
    print('%-5s offset within class x expert-ACL strata %+.3f | one-sided perm p %.3f' % (k, obs, (null <= obs).mean()))
    hi = ~lw
    print('      high sites only: dd mean ACL+ %+.3f (n %d) vs ACL- %+.3f (n %d)' % (dd[hi & (acl == 1)].mean(), (hi & (acl == 1)).sum(), dd[hi & (acl == 0)].mean(), (hi & (acl == 0)).sum()))
# E2 gain on gold58 by expert-ACL status of the positive (c4, a .2)
s = pct(d['c4'][:, J['Synovitis']]); e = pct(d['c4'][:, J['Effusion']]); o = s + .2 * e
P_, N_ = np.where(ys == 1)[0], np.where(ys == 0)[0]
b = (s[P_][:, None] > s[N_][None, :]) + .5 * (s[P_][:, None] == s[N_][None, :]); f = (o[P_][:, None] > o[N_][None, :]) + .5 * (o[P_][:, None] == o[N_][None, :])
dm = f - b
for nm, mp in (('pos ACL+', acl[P_] == 1), ('pos ACL-', acl[P_] == 0)):
    print('c4 E2 a .2 gold58 d from pairs with %s positives: %+.4f (%d pos)' % (nm, dm[mp].sum() / dm.size, mp.sum()))
for nm, mn in (('neg ACL+', acl[N_] == 1), ('neg ACL-', acl[N_] == 0)):
    print('c4 E2 a .2 gold58 d from pairs with %s negatives: %+.4f (%d neg)' % (nm, dm[:, mn].sum() / dm.size, mn.sum()))
print('gold58 P(Syn | ACL+): low %d/%d high %d/%d ; P(Syn | ACL-): low %d/%d high %d/%d' % (
    ys[lw & (acl == 1)].sum(), (lw & (acl == 1)).sum(), ys[~lw & (acl == 1)].sum(), (~lw & (acl == 1)).sum(),
    ys[lw & (acl == 0)].sum(), (lw & (acl == 0)).sum(), ys[~lw & (acl == 0)].sum(), (~lw & (acl == 0)).sum()))
