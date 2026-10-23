# skp_syn.py (10-20 night, Claude, SKEPTIC of E2s). LOCAL RSNA only (no OAI), aggregates only. Tests:
# A. Is the OOF "site prior" in the quantity E2 acts on (Syn head minus its Eff-explained part) a reporting-rate gradient, or an
#    English-vs-rest effect? Per-language residual (Syn pct regressed on Eff pct), Spearman vs v2 Syn YES rate, leave-en-out gap.
# B. Case mix by site group from v2 report labels (labels reported consistently across languages: ACL, OA, Eff, Bak, Cont, Frac).
# C. Does the low-site residual survive controlling for v2 trauma / degenerative indicators?
# D. gold58 permutation test of the c4 site offset in (Syn pct - Eff pct), class-stratified.
# E. MODEL: pooled-AUC effect of a group offset as a function of the low-vs-high expert prevalence difference.
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
from load import s6seed
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'
pct = lambda v: rankdata(v) / len(v)
LOW = ['bg', 'hr', 'tr', 'nl']
g74 = set(pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
lang = pd.read_csv(R + '/_data/_reports_lang.csv', dtype={'StudyInstanceUID': str}, usecols=['StudyInstanceUID', 'lang']).set_index('StudyInstanceUID')['lang']
v2 = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
yes = lambda c, idx: (v2[c + '__verdict'].reindex(idx) == 'YES').values.astype(float)
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
mods = {'R7': r7[['p|Synovitis', 'p|Effusion']].set_axis(['Synovitis', 'Effusion'], axis=1)}
for s_ in (1044, 1045):
    mods['s%d' % s_] = s6seed(s_)[0][['Synovitis', 'Effusion']]
print('=' * 30, 'A. per-language residual of the Syn head on the Eff head (OOF, non-gold)')
for k, P in mods.items():
    P = P[~P.index.isin(g74)].dropna()
    df = pd.DataFrame({'syn': pct(P.Synovitis.values), 'eff': pct(P.Effusion.values)}, index=P.index)
    df['lang'] = lang.reindex(df.index).values; df['effyes'] = yes('Effusion', df.index); df['synyes'] = yes('Synovitis', df.index)
    b = np.polyfit(df.eff, df.syn, 1); df['res'] = df.syn - np.polyval(b, df.eff)
    for nm, m in (('all', np.ones(len(df), bool)), ('v2 Eff YES', df.effyes.values == 1)):
        t = df[m].groupby('lang').agg(n=('res', 'size'), res=('res', 'mean'), rate=('synyes', 'mean'))
        t = t[t.n >= 50]
        rho = spearmanr(t.res, t.rate)[0]
        lo = df[m & df.lang.isin(LOW).values].res.mean(); hi_ne = df[m & ~df.lang.isin(LOW).values & (df.lang != 'en').values].res.mean()
        en = df[m & (df.lang == 'en').values].res.mean()
        print('[%s] %-10s Spearman(res, v2 Syn rate) over %d langs %+.2f | res: LOW %+.3f, HIGH-non-en %+.3f, en %+.3f | gap LOW - HIGH-non-en %+.3f' % (
            k, nm, len(t), rho, lo, hi_ne, en, lo - hi_ne))
        if nm == 'all':
            print('      ' + ' '.join('%s %+.3f' % (l, r) for l, r in t.res.sort_values().items()))
print('=' * 30, 'B. case mix by site group (v2 YES rates, non-gold train)')
idx = lang.index[~lang.index.isin(g74)]; low = lang.reindex(idx).isin(LOW).values; en_ = (lang.reindex(idx) == 'en').values
for c in ['ACL', 'MCL', 'Contusion', 'Fracture', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', "Baker's", 'Medial Meniscus', 'Synovitis']:
    y = yes(c, idx)
    print('%-16s LOW %.3f | HIGH %.3f | HIGH-non-en %.3f | en %.3f' % (c, y[low].mean(), y[~low].mean(), y[~low & ~en_].mean(), y[en_].mean()))
y_acute = np.maximum.reduce([yes(c, idx) for c in ('ACL', 'Contusion', 'Fracture')]); y_oa = np.maximum.reduce([yes(c, idx) for c in ('Medial OA', 'Lateral OA', 'PF OA')])
print('any acute (ACL/Cont/Frac) LOW %.3f HIGH %.3f | any OA LOW %.3f HIGH %.3f' % (y_acute[low].mean(), y_acute[~low].mean(), y_oa[low].mean(), y_oa[~low].mean()))
print('=' * 30, 'C. low-site residual gap, raw vs controlling v2 case mix (OLS on res, non-gold OOF)')
for k, P in mods.items():
    P = P[~P.index.isin(g74)].dropna()
    s, e = pct(P.Synovitis.values), pct(P.Effusion.values)
    b = np.polyfit(e, s, 1); res = s - np.polyval(b, e)
    lw = lang.reindex(P.index).isin(LOW).values.astype(float)
    X0 = np.c_[np.ones(len(P)), lw]
    cov = np.c_[[yes(c, P.index) for c in ('ACL', 'MCL', 'Contusion', 'Fracture', 'Medial OA', 'Lateral OA', 'PF OA', "Baker's", 'Medial Meniscus', 'Lateral Meniscus')]].T
    X1 = np.c_[X0, cov]
    b0 = np.linalg.lstsq(X0, res, rcond=None)[0]; b1 = np.linalg.lstsq(X1, res, rcond=None)[0]
    print('[%s] low-site coef raw %+.4f | with 10 v2 case-mix indicators %+.4f' % (k, b0[1], b1[1]))
print('=' * 30, 'D. gold58 c4: site offset in (Syn pct - Eff pct), class-stratified permutation (site labels permuted within class)')
import cbase
d = cbase.load(); ids = d['ids']; Y = d['Y58']; L = cbase.L; J = {c: i for i, c in enumerate(L)}
ys = Y[:, J['Synovitis']]; lg = lang.reindex(ids).values; lw = np.isin(lg, LOW)
rng = np.random.default_rng(11)
for k, P in (('c4', d['c4']), ('R6', d['members']['R6']), ('S6fall', d['members']['S6']), ('R384', d['members']['R384'])):
    dd = pct(P[:, J['Synovitis']]) - pct(P[:, J['Effusion']])
    stat = lambda w: np.mean([dd[(ys == c) & w].mean() - dd[(ys == c) & ~w].mean() for c in (0, 1)])
    obs = stat(lw); null = []
    for _ in range(20000):
        w = lw.copy()
        for c in (0, 1):
            ii = np.where(ys == c)[0]; w[ii] = rng.permutation(w[ii])
        null.append(stat(w))
    null = np.array(null)
    print('%-7s offset (low - high, mean over classes) %+.3f | one-sided perm p %.3f | null sd %.3f' % (k, obs, (null <= obs).mean(), null.std()))
# expert ACL / acute among the low-site Syn+ (counts only)
for c in ('ACL', 'MCL', 'Contusion', 'Fracture', 'Effusion', 'PF OA', 'Medial OA'):
    yc = Y[:, J[c]]
    print('gold58 expert %-10s among Syn+: LOW %d/%d  HIGH %d/%d | among Syn-: LOW %d/%d HIGH %d/%d' % (
        c, yc[(ys == 1) & lw].sum(), ((ys == 1) & lw).sum(), yc[(ys == 1) & ~lw].sum(), ((ys == 1) & ~lw).sum(),
        yc[(ys == 0) & lw].sum(), ((ys == 0) & lw).sum(), yc[(ys == 0) & ~lw].sum(), ((ys == 0) & ~lw).sum()))
print('=' * 30, 'E. MODEL: binormal, AUC .85 within group, low share w .30, latent offset on low group; pooled AUC loss vs no offset (oracle-centring room)')
def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
mu = norm.ppf(.85) * np.sqrt(2); N = 400000; rs = np.random.default_rng(3)
for dpct in (.02, .05, .10):
    for pL, pH in ((.47, .47), (.44, .48), (.50, .46), (.53, .44)):
        g = rs.random(N) < .30; y = np.where(g, rs.random(N) < pL, rs.random(N) < pH).astype(int)
        z = y * mu + rs.normal(0, 1, N)
        sd_ = z.std(); off = dpct / (norm.pdf(0) / sd_)  # rank-pct offset -> latent units near the centre
        a0 = auc(z, y); a1 = auc(z - off * g, y)
        print('offset %.2f pct  pL %.2f pH %.2f | room (oracle gain) %+.4f label = %+.2fe-4 macro' % (dpct, pL, pH, a0 - a1, (a0 - a1) / 12 * 1e4))
