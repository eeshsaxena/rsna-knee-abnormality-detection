# fresh_site.py (10-20 ~22:30 IST, Claude, fresh-eyes track). RESEARCH ONLY, local CPU, NO OAI, RSNA train OOF + gold74 counts.
# Question (not examined by any prior round): is the E-OA dose response SITE-STABLE? The shipped A (pw30) and the slate (pw25,
# r6pw20 at .2, lomm05) all rest on a transform whose sign is steep in model trio co-firing (adj_c12: R7 co-fires .824 and reads
# pw30 -10.3e-4, the 3-mean co-fires .854 and reads +11.0e-4). If co-firing or the raked gain varies by site, a private set with
# a different site mix (or unseen sites) prefers a different dose than public.
# Ruler = the adj_c12 / skep_stack3 ruler that reproduced the 5 public E-OA reads at ratio 1.01-1.12 (3-mean of R7 + s1044 + s1045
# OOF, report v2 labels, raked to the gold74 EXPERT trio joint). Site proxy = report language (9 groups) and acquisition
# atypicality (trick/work/atypicality.csv). Transforms are applied on the POOLED set (as the kernel does over the whole hidden set)
# and scored inside each group with that group's own raking (group margins, gold74 association). Also: pooled gain split into
# within-site pairs vs between-site pairs. Writes fresh_site.log. Aggregates only.
import sys
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
R = 'C:/Users/eeshs/Downloads/rsna-knee'
OUT = R + '/session_2026-10-20/private/fresh_site.log'
log = open(OUT, 'w', encoding='utf-8')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
L = LD.L; pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}
OA = ['Medial OA', 'Lateral OA', 'PF OA']; jOA = [J[c] for c in OA]
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
SH = {'eoa10': (.1, 'sym'), 'eoa20': (.2, 'sym'), 'eoa30': (.3, 'sym'), 'pw20': (.2, 'pw'), 'pw25': (.25, 'pw'), 'pw30': (.3, 'pw'), 'pw35': (.35, 'pw')}
PUB = {'eoa10': 5.9, 'eoa20': 10.3, 'eoa30': 9.6, 'pw20': 10.4, 'pw30': 11.3}
rng = np.random.default_rng(2210)


def eoa(F, a, mode):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]
        w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o] if mode == 'pw' else [1., 1.]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G


def line(F, t, o, a):
    G = F.copy(); G[:, J[t]] = pct(pct(F[:, J[t]]) - a * pct(F[:, J[o]])); return G


def wauc_parts(s, y, w):
    """weighted AUC numerator and pair mass."""
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    wp = w * y; wn = w * (1 - y)
    _, inv = np.unique(s, return_inverse=True)
    gp = np.bincount(inv, wp); gn = np.bincount(inv, wn); cn = np.cumsum(gn) - gn
    return float((gp * (cn + .5 * gn)).sum()), float(wp.sum() * wn.sum())


def wauc(s, y, w):
    a, b = wauc_parts(s, y, w); return a / b


def rake(T, margins, it=300):
    T = T.copy(); k = T.ndim
    for _ in range(it):
        for ax in range(k):
            cur = T.sum(axis=tuple(i for i in range(k) if i != ax)); sh = [1] * k; sh[ax] = 2
            T = T * (np.array([1 - margins[ax], margins[ax]]) / cur).reshape(sh)
    return T / T.sum()


def weights(Ysub, cols, Y74):
    jc = [J[c] for c in cols]; ok = ~np.isnan(Y74[:, jc]).any(1); Yg = Y74[ok][:, jc].astype(int)
    T = np.zeros((2,) * len(cols)); obs = np.zeros((2,) * len(cols))
    for r in Yg: T[tuple(r)] += 1
    for r in Ysub: obs[tuple(r)] += 1
    obs /= obs.sum(); tgt = rake(T + .5, Ysub.mean(0)); w = (tgt / obs)[tuple(Ysub.T)]
    return w / w.mean()


def trio_gain(G, F, Ym, w, idx=None):
    if idx is None: idx = np.arange(len(Ym))
    return sum(wauc(G[idx, J[c]], Ym[idx, i], w[idx]) - wauc(F[idx, J[c]], Ym[idx, i], w[idx]) for i, c in enumerate(OA)) / 12 * 1e4


def rho(F, a_, b_, idx):
    return float(np.corrcoef(ns(F[idx, J[a_]]), ns(F[idx, J[b_]]))[0, 1])


g74, _ = LD.gold(); allg = set(g74.StudyInstanceUID); Y74 = g74[L].values.astype(float)
r7 = LD.r7(); R7c = ['p|' + c for c in L]
s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2')
tr = [i for i in r7.index if i not in allg and i in lv2.index]
tr3 = [i for i in tr if i in s44.index and i in s45.index]
X = {'R7': r7.loc[tr3, R7c].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float), 's1045': s45.loc[tr3, L].values.astype(float)}
X['S6mean'] = np.column_stack([np.mean([pct(X[k][:, j]) for k in ('s1044', 's1045')], 0) for j in range(12)])
X['3mean'] = np.column_stack([np.mean([pct(X[k][:, j]) for k in ('R7', 's1044', 's1045')], 0) for j in range(12)])
Y = lv2.loc[tr3, L].values
lang = pd.read_csv(R + '/_data/_reports_lang.csv', usecols=['StudyInstanceUID', 'lang'], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['lang']
atyp = pd.read_csv(R + '/session_2026-10-19/trick/work/atypicality.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['atyp']
LG = lang.reindex(tr3).fillna('na').values; AT = atyp.reindex(tr3).values
P('rows tr3 %d | lang counts %s | atyp missing %d' % (len(tr3), dict(pd.Series(LG).value_counts()), int(np.isnan(AT).sum())))

m = ~np.isnan(Y[:, jOA]).any(1); Ym = Y[m][:, jOA].astype(int); LGm = LG[m]; ATm = AT[m]
P('rows with trio labels %d' % m.sum())
groups = [g for g in pd.Series(LGm).value_counts().index]
wpool = weights(Ym, OA, Y74)
wgrp = np.zeros(len(Ym))
for g in groups:
    ii = np.where(LGm == g)[0]; wgrp[ii] = weights(Ym[ii], OA, Y74)
# report-label trio coupling per group (phi, raw) for info
P('')
P('report v2 trio prevalence / phi(MOA,LOA) per group (raw, before raking):')
for g in groups:
    ii = LGm == g; yy = Ym[ii]
    P('  %-3s n %4d prev MOA %.2f LOA %.2f PF %.2f | phi MOA-LOA %.2f MOA-PF %.2f LOA-PF %.2f' % (
        g, ii.sum(), *yy.mean(0), np.corrcoef(yy[:, 0], yy[:, 1])[0, 1], np.corrcoef(yy[:, 0], yy[:, 2])[0, 1], np.corrcoef(yy[:, 1], yy[:, 2])[0, 1]))

for mn in ('3mean', 'S6mean', 'R7'):
    F = X[mn][m]
    Gs = {k: eoa(F, a, md) for k, (a, md) in SH.items()}
    P('')
    P('==== %s' % mn)
    pooled = {k: trio_gain(G, F, Ym, wpool) for k, G in Gs.items()}
    P('  pooled raked (pooled transform, pooled AUC): %s' % ' '.join('%s %+.1f' % (k, v) for k, v in pooled.items()))
    # within-site vs between-site pair decomposition (pooled raking weights)
    wi, bt = {}, {}
    for k, G in Gs.items():
        dw, db = 0., 0.
        for i, c in enumerate(OA):
            num_p = {}; mass_p = {}
            for tag, S_ in (('G', G), ('F', F)):
                Np, Mp = wauc_parts(S_[:, J[c]], Ym[:, i], wpool)
                Nw, Mw = 0., 0.
                for g in groups:
                    ii = LGm == g
                    if Ym[ii, i].sum() == 0 or Ym[ii, i].sum() == ii.sum(): continue
                    a_, b_ = wauc_parts(S_[ii, J[c]], Ym[ii, i], wpool[ii]); Nw += a_; Mw += b_
                num_p[tag] = (Nw / Mw, (Np - Nw) / (Mp - Mw)); mass_p[tag] = (Mw, Mp - Mw)
            dw += num_p['G'][0] - num_p['F'][0]; db += num_p['G'][1] - num_p['F'][1]
        wi[k] = dw / 12 * 1e4; bt[k] = db / 12 * 1e4
    P('  within-site pairs only : %s' % ' '.join('%s %+.1f' % (k, v) for k, v in wi.items()))
    P('  between-site pairs only: %s' % ' '.join('%s %+.1f' % (k, v) for k, v in bt.items()))
    P('  within-site share of pair mass (last label, PF OA): %.2f' % (mass_p['F'][0] / sum(mass_p['F'])))
    # per-group: co-firing and raked gain (group raking), bootstrap sd for pw30 - pw20 and pw30 - pw25
    P('  per group (transform pooled, AUC in group, group-raked):')
    P('    grp    n  coMOA~LOA coMOA~PF coLOA~PF | ' + ' '.join('%6s' % k for k in SH) + ' | pw30-pw20 (sd)  pw30-pw25 (sd) | best')
    rows = []
    for g in groups + ['ALL']:
        ii = np.arange(len(Ym)) if g == 'ALL' else np.where(LGm == g)[0]
        w = wpool if g == 'ALL' else wgrp
        co = [rho(F, 'Medial OA', 'Lateral OA', ii), rho(F, 'Medial OA', 'PF OA', ii), rho(F, 'Lateral OA', 'PF OA', ii)]
        gv = {k: trio_gain(G, F, Ym, w, ii) for k, G in Gs.items()}
        bs1, bs2 = [], []
        for _ in range(120):
            jj = ii[rng.integers(0, len(ii), len(ii))]
            if (Ym[jj].sum(0) == 0).any(): continue
            a30 = trio_gain(Gs['pw30'], F, Ym, w, jj); a20 = trio_gain(Gs['pw20'], F, Ym, w, jj); a25 = trio_gain(Gs['pw25'], F, Ym, w, jj)
            bs1.append(a30 - a20); bs2.append(a30 - a25)
        best = max(gv, key=gv.get)
        rows.append((g, len(ii), co, gv))
        P('    %-4s %4d  %.3f     %.3f    %.3f    | ' % (g, len(ii), *co) + ' '.join('%+6.1f' % gv[k] for k in SH) +
          ' | %+5.1f (%.1f)    %+5.1f (%.1f)   | %s' % (gv['pw30'] - gv['pw20'], np.std(bs1), gv['pw30'] - gv['pw25'], np.std(bs2), best))
    # across groups: does the pw30 - pw20 difference track co-firing?
    gr = [r for r in rows if r[0] != 'ALL']
    co_mean = np.array([np.mean(r[2]) for r in gr]); d3020 = np.array([r[3]['pw30'] - r[3]['pw20'] for r in gr])
    g30 = np.array([r[3]['pw30'] for r in gr]); nn = np.array([r[1] for r in gr])
    P('  across %d groups: mean trio co-firing range %.3f-%.3f (sd %.3f); corr(co-firing, pw30-pw20) %+.2f; corr(co-firing, pw30 gain) %+.2f; '
      'n-weighted sd of pw30 gain across groups %.1f' % (len(gr), co_mean.min(), co_mean.max(), co_mean.std(), np.corrcoef(co_mean, d3020)[0, 1],
                                                       np.corrcoef(co_mean, g30)[0, 1], np.sqrt(np.cov(g30, aweights=nn))))
    # atypicality strata (acquisition-atypical = unseen-site stand-in)
    ok = ~np.isnan(ATm); q80 = np.nanquantile(ATm, .8); q90 = np.nanquantile(ATm, .9)
    for nm, sel in (('atyp top20', ok & (ATm >= q80)), ('atyp top10', ok & (ATm >= q90)), ('typ bottom80', ok & (ATm < q80))):
        ii = np.where(sel)[0]; w = weights(Ym[ii], OA, Y74); ww = np.zeros(len(Ym)); ww[ii] = w
        co = [rho(F, 'Medial OA', 'Lateral OA', ii), rho(F, 'Medial OA', 'PF OA', ii), rho(F, 'Lateral OA', 'PF OA', ii)]
        gv = {k: trio_gain(G, F, Ym, ww, ii) for k, G in Gs.items()}
        bs1 = []
        for _ in range(120):
            jj = ii[rng.integers(0, len(ii), len(ii))]
            bs1.append(trio_gain(Gs['pw30'], F, Ym, ww, jj) - trio_gain(Gs['pw20'], F, Ym, ww, jj))
        P('  %-12s n %4d co %.3f/%.3f/%.3f | %s | pw30-pw20 %+.1f (sd %.1f)' % (nm, len(ii), *co, ' '.join('%s %+.1f' % (k, v) for k, v in gv.items()),
                                                                            gv['pw30'] - gv['pw20'], np.std(bs1)))

# lomm05 per group on the 3-mean (16-cell raking within group), pw30 base
P('')
P('==== lomm05 (LOA <- MM a .05 on pw30), 3mean, per group, 16-cell raking within group')
cols = OA + ['Medial Meniscus']; jc = [J[c] for c in cols]
m2 = ~np.isnan(Y[:, jc]).any(1); F = X['3mean'][m2]; Y2 = Y[m2][:, jc].astype(int); LG2 = LG[m2]
B30 = eoa(F, .3, 'pw'); G = line(B30, 'Lateral OA', 'Medial Meniscus', .05); jl = J['Lateral OA']; yl = Y2[:, 1]
for g in [x for x in pd.Series(LG2).value_counts().index] + ['ALL']:
    ii = np.arange(len(Y2)) if g == 'ALL' else np.where(LG2 == g)[0]
    w = weights(Y2[ii], cols, Y74)
    d = wauc(G[ii, jl], yl[ii], w) - wauc(B30[ii, jl], yl[ii], w)
    co = rho(B30, 'Lateral OA', 'Medial Meniscus', ii)
    bs = []
    for _ in range(150):
        k_ = rng.integers(0, len(ii), len(ii)); bs.append(wauc(G[ii[k_], jl], yl[ii[k_]], w[k_]) - wauc(B30[ii[k_], jl], yl[ii[k_]], w[k_]))
    P('  %-4s n %4d LOA~MM co-firing (pw30) %.3f | lomm05 %+.2fe-4 macro (sd %.2f, P>0 %.2f)' % (g, len(ii), co, d / 12 * 1e4, np.std(bs) / 12 * 1e4, np.mean(np.array(bs) > 0)))
P('FRESH_SITE DONE')
