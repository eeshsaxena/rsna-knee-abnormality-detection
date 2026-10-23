# m2_member_site.py (10-19 trick lens, Claude, research only, CPU): per-site behaviour of the members we hold OOF for.
# R7 OOF (R6 family, 4 random folds) and our S6-recipe seeds s1044 (5 folds) / s1045 (4 folds) (S6fall family).
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
from scipy.stats import rankdata
L = LD.L; R = LD.R
lang = pd.read_csv(R+'/_data/_reports_lang.csv', usecols=['StudyInstanceUID','lang']).set_index('StudyInstanceUID').lang
r7 = LD.r7()[['p|'+c for c in L]]; r7.columns = L
s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045)
g, g58 = LD.gold(); gold_ids = set(g.StudyInstanceUID)
mem = {'R7': r7, 'S6s44': s44, 'S6s45': s45}
for src in ['v2', 'v4']:
    lab = LD.labels(src)
    ids = sorted(set(r7.index) & set(s44.index) & set(s45.index) & set(lab.index) - gold_ids)
    print('\n######## labels', src, 'common non-gold studies', len(ids))
    lg_ = lang.reindex(ids).values
    langs = ['en','es','tr','hr','el','de','bg','nl','fr']
    res = {}
    for m, d in mem.items():
        P = d.loc[ids, L].values.astype(float)
        rows = []
        for j, c in enumerate(L):
            y = lab.loc[ids, c].values; ok = ~np.isnan(y); yy = y[ok].astype(int); pp = P[ok, j]; ll = lg_[ok]
            pr = rankdata(pp) / len(pp)
            pooled = LD.auc(pp, yy)
            # within-site AUC: weighted by pairs
            num = 0.; den = 0.; per = {}
            for s in langs:
                k = ll == s; npos = yy[k].sum(); nneg = k.sum() - npos
                if npos >= 3 and nneg >= 3:
                    a = LD.auc(pp[k], yy[k]); per[s] = a; num += a * npos * nneg; den += npos * nneg
            within = num / den
            # site offset spread among negatives / positives (pooled pct rank), weighted SD across sites
            offn = []; offp = []; wn = []; wp = []
            for s in langs:
                k = ll == s
                kn = k & (yy == 0); kp = k & (yy == 1)
                if kn.sum() >= 10: offn.append(pr[kn].mean()); wn.append(kn.sum())
                if kp.sum() >= 10: offp.append(pr[kp].mean()); wp.append(kp.sum())
            def wsd(v, w):
                v = np.array(v); w = np.array(w, float); mu = (v*w).sum()/w.sum(); return np.sqrt(((v-mu)**2*w).sum()/w.sum())
            rows.append(dict(label=c, pooled=pooled, within=within, gap=pooled-within, sd_neg=wsd(offn, wn), sd_pos=wsd(offp, wp), **{'a_'+s: per.get(s, np.nan) for s in langs}))
        res[m] = pd.DataFrame(rows).set_index('label')
    for m, df in res.items():
        print('\n==', m, 'macro pooled %.4f within %.4f gap %.4f | mean sd_neg %.4f sd_pos %.4f' % (df.pooled.mean(), df.within.mean(), df.gap.mean(), df.sd_neg.mean(), df.sd_pos.mean()))
        print('  per-site macro AUC:', {s: round(df['a_'+s].mean(), 4) for s in langs})
    print('\nper label sd_neg (site offset among negatives, pct-rank units):')
    print(pd.DataFrame({m: df.sd_neg for m, df in res.items()}).round(4).to_string())
    print('\nper label gap (pooled - within):')
    print(pd.DataFrame({m: df.gap for m, df in res.items()}).round(4).to_string())
    # site-size dependence: per-site macro AUC relative to member's pooled macro, vs site n
    n_site = pd.Series(lg_).value_counts()
    print('\nper-site macro AUC minus member macro within (vs site n):')
    tab = pd.DataFrame({m: {s: df['a_'+s].mean() - df.within.mean() for s in langs} for m, df in res.items()})
    tab['n'] = n_site.reindex(langs).values
    print(tab.round(4).to_string())
    pd.concat(res, names=['member']).to_csv('m2_%s.csv' % src)
