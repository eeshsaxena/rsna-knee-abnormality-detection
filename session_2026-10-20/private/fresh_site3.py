# fresh_site3.py (10-20 ~23:20 IST, Claude, fresh-eyes track). RESEARCH ONLY, local CPU, NO OAI. Q-profile 90 % upper bounds on the
# between-site sd (tau) of the E-OA dose response and of lomm05 on the calibrated ruler (3-mean, report v2 raked to the gold74 expert
# joint), 9 language sites, plus the pooled within-site / between-site pair split for the v4 reader. Writes fresh_site3.log.
import sys
import numpy as np, pandas as pd
from scipy.stats import chi2
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
R = 'C:/Users/eeshs/Downloads/rsna-knee'
src = open(R + '/session_2026-10-20/private/fresh_site.py', encoding='utf-8').read()
exec(src.split("g74, _ = LD.gold()")[0].replace("log = open(OUT, 'w', encoding='utf-8')", "log = None"), globals())
log = open(R + '/session_2026-10-20/private/fresh_site3.log', 'w', encoding='utf-8')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(2320)
g74, _ = LD.gold(); allg = set(g74.StudyInstanceUID); Y74 = g74[L].values.astype(float)
r7 = LD.r7(); R7c = ['p|' + c for c in L]; s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045)
lang = pd.read_csv(R + '/_data/_reports_lang.csv', usecols=['StudyInstanceUID', 'lang'], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['lang']


def qprof_upper(est, var, level=.90):
    est = np.asarray(est); var = np.asarray(var); k = len(est)
    def Q(t2):
        w = 1 / (var + t2); m = (w * est).sum() / w.sum(); return (w * (est - m) ** 2).sum()
    crit = chi2.ppf(1 - level, k - 1)  # upper bound: smallest t2 with Q(t2) <= lower chi2 quantile
    if Q(0) <= crit: return 0.0
    lo, hi = 0., 1e4
    for _ in range(80):
        mid = (lo + hi) / 2
        if Q(mid) > crit: lo = mid
        else: hi = mid
    return np.sqrt(hi)


for reader in ('v2', 'v4'):
    lv = LD.labels(reader)
    tr3 = [i for i in r7.index if i not in allg and i in lv.index and i in s44.index and i in s45.index]
    X3 = np.column_stack([np.mean([pct(A[:, j]) for A in (r7.loc[tr3, R7c].values.astype(float), s44.loc[tr3, L].values.astype(float),
                                                          s45.loc[tr3, L].values.astype(float))], 0) for j in range(12)])
    Y = lv.loc[tr3, L].values; LG = lang.reindex(tr3).fillna('na').values
    m = ~np.isnan(Y[:, jOA]).any(1); F = X3[m]; Ym = Y[m][:, jOA].astype(int); LGm = LG[m]
    groups = list(pd.Series(LGm).value_counts().index)
    wg = np.zeros(len(Ym))
    for g in groups:
        ii = np.where(LGm == g)[0]; wg[ii] = weights(Ym[ii], OA, Y74)
    Gs = {k: eoa(F, *SH[k]) for k in ('pw20', 'pw25', 'pw30')}
    E = {k: [] for k in ('pw30', 'pw30-pw20', 'pw30-pw25')}; V = {k: [] for k in E}
    for g in groups:
        ii = np.where(LGm == g)[0]
        b = {k: trio_gain(Gs[k], F, Ym, wg, ii) for k in Gs}; bs = {k: [] for k in Gs}
        for _ in range(250):
            jj = ii[rng.integers(0, len(ii), len(ii))]
            if (Ym[jj].sum(0) == 0).any() or (Ym[jj].sum(0) == len(jj)).any(): continue
            for k in Gs: bs[k].append(trio_gain(Gs[k], F, Ym, wg, jj))
        bs = {k: np.array(v) for k, v in bs.items()}
        E['pw30'].append(b['pw30']); V['pw30'].append(bs['pw30'].var())
        E['pw30-pw20'].append(b['pw30'] - b['pw20']); V['pw30-pw20'].append((bs['pw30'] - bs['pw20']).var())
        E['pw30-pw25'].append(b['pw30'] - b['pw25']); V['pw30-pw25'].append((bs['pw30'] - bs['pw25']).var())
    for k in E:
        P('reader %s 3mean %-10s per-site %s | tau 90%% upper %.1fe-4' % (reader, k, ' '.join('%+.1f' % x for x in E[k]), qprof_upper(E[k], V[k])))
    # lomm05
    cols = OA + ['Medial Meniscus']; jc = [J[c] for c in cols]
    m2 = ~np.isnan(Y[:, jc]).any(1); F2 = X3[m2]; Y2 = Y[m2][:, jc].astype(int); LG2 = LG[m2]
    B30 = eoa(F2, .3, 'pw'); G = line(B30, 'Lateral OA', 'Medial Meniscus', .05); jl = J['Lateral OA']; yl = Y2[:, 1]
    e_, v_ = [], []
    for g in list(pd.Series(LG2).value_counts().index):
        ii = np.where(LG2 == g)[0]; w = weights(Y2[ii], cols, Y74)
        e_.append((wauc(G[ii, jl], yl[ii], w) - wauc(B30[ii, jl], yl[ii], w)) / 12 * 1e4); bs = []
        for _ in range(250):
            k_ = rng.integers(0, len(ii), len(ii))
            if yl[ii[k_]].sum() in (0, len(k_)): continue
            bs.append((wauc(G[ii[k_], jl], yl[ii[k_]], w[k_]) - wauc(B30[ii[k_], jl], yl[ii[k_]], w[k_])) / 12 * 1e4)
        v_.append(np.var(bs))
    P('reader %s 3mean lomm05     per-site %s | tau 90%% upper %.2fe-4' % (reader, ' '.join('%+.1f' % x for x in e_), qprof_upper(e_, v_)))
    if reader == 'v4':
        wpool = weights(Ym, OA, Y74); out = []
        for k, G_ in Gs.items():
            dw = db = 0.
            for i, c in enumerate(OA):
                r_ = {}
                for tag, S_ in (('G', G_), ('F', F)):
                    Np, Mp = wauc_parts(S_[:, J[c]], Ym[:, i], wpool); Nw = Mw = 0.
                    for g in groups:
                        ii = LGm == g
                        if Ym[ii, i].sum() in (0, ii.sum()): continue
                        a_, b_ = wauc_parts(S_[ii, J[c]], Ym[ii, i], wpool[ii]); Nw += a_; Mw += b_
                    r_[tag] = (Nw / Mw, (Np - Nw) / (Mp - Mw))
                dw += r_['G'][0] - r_['F'][0]; db += r_['G'][1] - r_['F'][1]
            out.append('%s within %+.1f between %+.1f' % (k, dw / 12 * 1e4, db / 12 * 1e4))
        P('reader v4 3mean pooled pair split: ' + ' | '.join(out))
P('FRESH_SITE3 DONE')
