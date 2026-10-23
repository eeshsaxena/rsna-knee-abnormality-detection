# fresh_site2.py (10-20 ~22:50 IST, Claude, fresh-eyes track). RESEARCH ONLY, local CPU, NO OAI. Follow-up to fresh_site.py:
# is the per-site spread of the E-OA dose response larger than sampling noise? Random-effects (DerSimonian-Laird) over the 9
# report-language sites, per dose and per dose difference, on the calibrated 3-mean ruler (and S6mean / R7), for two report
# readers (v2 = the calibrated one, v4 = a second LLM reader of the same reports). Also lomm05 (LOA <- MM a .05 on pw30).
# Transform on the pooled set, AUC inside each site with that site's raking (site margins, gold74 expert association).
# Writes fresh_site2.log. Aggregates only.
import sys
import numpy as np, pandas as pd
from scipy.stats import chi2
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/private')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
R = 'C:/Users/eeshs/Downloads/rsna-knee'
src = open(R + '/session_2026-10-20/private/fresh_site.py', encoding='utf-8').read()
# reuse the helper defs only (everything above the data-loading line); its own log handle is disabled
exec(src.split("g74, _ = LD.gold()")[0].replace("log = open(OUT, 'w', encoding='utf-8')", "log = None"), globals())
OUT = R + '/session_2026-10-20/private/fresh_site2.log'
log = open(OUT, 'w', encoding='utf-8')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(2250)
g74, _ = LD.gold(); allg = set(g74.StudyInstanceUID); Y74 = g74[L].values.astype(float)
r7 = LD.r7(); R7c = ['p|' + c for c in L]
s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045)
lang = pd.read_csv(R + '/_data/_reports_lang.csv', usecols=['StudyInstanceUID', 'lang'], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['lang']
B = 300
KEYS = ['eoa10', 'eoa20', 'pw20', 'pw25', 'pw30', 'pw35']
DIFFS = [('pw30', 'pw20'), ('pw30', 'pw25'), ('pw25', 'pw20'), ('pw20', 'eoa20')]


def dl(est, var):
    est = np.asarray(est); var = np.asarray(var); w = 1 / var; fe = (w * est).sum() / w.sum()
    Q = (w * (est - fe) ** 2).sum(); k = len(est); c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0., (Q - (k - 1)) / c); wr = 1 / (var + tau2); re = (wr * est).sum() / wr.sum()
    return fe, np.sqrt(1 / w.sum()), Q, 1 - chi2.cdf(Q, k - 1), np.sqrt(tau2), re, np.sqrt(1 / wr.sum())


for reader in ('v2', 'v4'):
    lv = LD.labels(reader)
    tr = [i for i in r7.index if i not in allg and i in lv.index]
    tr3 = [i for i in tr if i in s44.index and i in s45.index]
    X = {'R7': r7.loc[tr3, R7c].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float), 's1045': s45.loc[tr3, L].values.astype(float)}
    X['S6mean'] = np.column_stack([np.mean([pct(X[k][:, j]) for k in ('s1044', 's1045')], 0) for j in range(12)])
    X['3mean'] = np.column_stack([np.mean([pct(X[k][:, j]) for k in ('R7', 's1044', 's1045')], 0) for j in range(12)])
    Y = lv.loc[tr3, L].values; LG = lang.reindex(tr3).fillna('na').values
    m = ~np.isnan(Y[:, jOA]).any(1); Ym = Y[m][:, jOA].astype(int); LGm = LG[m]
    groups = list(pd.Series(LGm).value_counts().index)
    wgrp = np.zeros(len(Ym)); wpool = weights(Ym, OA, Y74)
    for g in groups:
        ii = np.where(LGm == g)[0]; wgrp[ii] = weights(Ym[ii], OA, Y74)
    for mn in ('3mean', 'S6mean', 'R7'):
        F = X[mn][m]; Gs = {k: eoa(F, *SH[k]) for k in KEYS}
        P('')
        P('==== reader %s | model %s | trio rows %d | pooled raked: %s' % (reader, mn, len(Ym), ' '.join('%s %+.1f' % (k, trio_gain(Gs[k], F, Ym, wpool)) for k in KEYS)))
        est = {k: [] for k in KEYS + ['%s-%s' % d for d in DIFFS]}; var = {k: [] for k in est}
        for g in groups:
            ii = np.where(LGm == g)[0]
            base = {k: trio_gain(Gs[k], F, Ym, wgrp, ii) for k in KEYS}
            bs = {k: [] for k in KEYS}
            for _ in range(B):
                jj = ii[rng.integers(0, len(ii), len(ii))]
                if (Ym[jj].sum(0) == 0).any() or (Ym[jj].sum(0) == len(jj)).any(): continue
                for k in KEYS: bs[k].append(trio_gain(Gs[k], F, Ym, wgrp, jj))
            bs = {k: np.array(v) for k, v in bs.items()}
            for k in KEYS: est[k].append(base[k]); var[k].append(bs[k].var())
            for a_, b_ in DIFFS:
                kk = '%s-%s' % (a_, b_); est[kk].append(base[a_] - base[b_]); var[kk].append((bs[a_] - bs[b_]).var())
        P('  per-site estimates (sd): ' + ' | '.join('%s pw30 %+.1f (%.1f)' % (g, est['pw30'][i], np.sqrt(var['pw30'][i])) for i, g in enumerate(groups)))
        P('  %-10s %8s %6s %7s %6s %6s %8s %6s' % ('quantity', 'FE mean', 'FE sd', 'Q', 'p(Q)', 'tau', 'RE mean', 'RE sd'))
        for k in est:
            fe, fsd, Q, pq, tau, re, rsd = dl(est[k], var[k])
            P('  %-10s %+8.1f %6.1f %7.1f %6.3f %6.1f %+8.1f %6.1f' % (k, fe, fsd, Q, pq, tau, re, rsd))
        # leave-one-site-out sensitivity of the FE pw30-pw20 / pw30-pw25 means
        for kk in ('pw30-pw20', 'pw30-pw25'):
            lo = []
            for i in range(len(groups)):
                e_ = [x for j, x in enumerate(est[kk]) if j != i]; v_ = [x for j, x in enumerate(var[kk]) if j != i]
                lo.append(dl(e_, v_)[0])
            P('  leave-one-site-out FE %s: min %+.1f (drop %s) max %+.1f (drop %s)' % (kk, min(lo), groups[int(np.argmin(lo))], max(lo), groups[int(np.argmax(lo))]))
    # lomm05 per site, 3-mean
    cols = OA + ['Medial Meniscus']; jc = [J[c] for c in cols]
    m2 = ~np.isnan(Y[:, jc]).any(1); F = X['3mean'][m2]; Y2 = Y[m2][:, jc].astype(int); LG2 = LG[m2]
    B30 = eoa(F, .3, 'pw'); G = line(B30, 'Lateral OA', 'Medial Meniscus', .05); jl = J['Lateral OA']; yl = Y2[:, 1]
    e_, v_ = [], []
    for g in list(pd.Series(LG2).value_counts().index):
        ii = np.where(LG2 == g)[0]; w = weights(Y2[ii], cols, Y74)
        d = (wauc(G[ii, jl], yl[ii], w) - wauc(B30[ii, jl], yl[ii], w)) / 12 * 1e4
        bs = []
        for _ in range(B):
            k_ = rng.integers(0, len(ii), len(ii))
            if yl[ii[k_]].sum() in (0, len(k_)): continue
            bs.append((wauc(G[ii[k_], jl], yl[ii[k_]], w[k_]) - wauc(B30[ii[k_], jl], yl[ii[k_]], w[k_])) / 12 * 1e4)
        e_.append(d); v_.append(np.var(bs))
    fe, fsd, Q, pq, tau, re, rsd = dl(e_, v_)
    P('  lomm05 3mean reader %s: FE %+.2f (sd %.2f) Q %.1f p %.3f tau %.2f RE %+.2f (sd %.2f)' % (reader, fe, fsd, Q, pq, tau, re, rsd))
P('FRESH_SITE2 DONE')
