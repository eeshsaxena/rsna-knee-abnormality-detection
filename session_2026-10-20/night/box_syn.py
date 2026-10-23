# box_syn.py (10-20 night, Claude). Box CPU only, read-only, piped via ssh stdin into python -I. Prints AGGREGATES ONLY
# (n, prevalence, grade cross-tab counts, AUC, AUC deltas, sd, P, Spearman, tetrachoric). Never prints an id, a row or a per-knee
# value; writes nothing.
# Question: is "Synovitis borrows Effusion" (E2: out = pct(Syn) + a pct(Eff)) supported on OAI MOAKS expert reads for OAI-FREE
# models, under every plausible definition of a synovitis target (MOAKS Hoffa-synovitis m_syic, MOAKS effusion-synovitis m_eff_wk,
# composites), unweighted and reweighted so the (target, effusion>=2) joint matches the RSNA gold58 expert (Syn, Eff) joint?
# Models (all OAI-free): R6 and S6fall served on the store-rebuilt OAI DICOMs (~/r5_oai), their c4-style .8/.2 rank blend, the
# fleet teacher D5 (s1043 5-fold mean), s1044 mix50 f0, s1045 288 f0. Raptor (c4's main Synovitis member) is OAI-trained on
# Hoffa >= 1 and cannot be measured here.
import os
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm, multivariate_normal
H = os.path.expanduser('~')
B = int(os.environ.get('BOOT', '500'))
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
       "Baker's", 'Contusion', 'Fracture']
pct = lambda v: rankdata(v) / len(v)


def auc(s, y, w=None):
    s = np.asarray(s, float); y = np.asarray(y, int)
    if w is None:
        r = rankdata(s); p = y.sum(); q = len(y) - p
        return np.nan if p == 0 or q == 0 else (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
    # weighted Mann-Whitney
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    wp = w * (y == 1); wn = w * (y == 0)
    _, inv = np.unique(s, return_inverse=True)
    cn = np.bincount(inv, wn); cp = np.bincount(inv, wp)
    below = np.cumsum(cn) - cn
    return (cp * (below + .5 * cn)).sum() / (wp.sum() * wn.sum())


def tetra(a, b):
    pa, pb = a.mean(), b.mean(); p11 = (a * b).mean(); ta, tb = norm.ppf(1 - pa), norm.ppf(1 - pb)
    best, br = 9, 0
    for r in np.linspace(-.98, .98, 99):
        p = 1 - norm.cdf(ta) - norm.cdf(tb) + multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([ta, tb])
        if abs(p - p11) < best:
            best, br = abs(p - p11), r
    return br


def rd(p):
    d = pd.read_csv(H + p, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    d = d[[c for c in LAB if c in d.columns]].astype(float)
    return d[d.index.str.startswith('oai_')]


S = pd.read_csv(H + '/oai_build/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
M = {'R6': rd('/r5_oai/run_r6/worker0/predictions.csv'), 'S6fall': rd('/r5_oai/s6_r5/out/predictions.csv'),
     'D5': rd('/oai_build/teacher/teacher_D5_mean_v00.csv'), 's1044': rd('/oai_build/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv'),
     's1045': rd('/oai_build/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv')}
ok = S[['m_eff_wk', 'm_syic']].notna().all(axis=1)
common = S.index[ok]
for k in M:
    common = common.intersection(M[k][['Synovitis', 'Effusion']].dropna().index)
common = common.sort_values()
n = len(common)
E = S.loc[common, 'm_eff_wk'].values.astype(int); Hf = S.loc[common, 'm_syic'].values.astype(int)
print('knees with both MOAKS grades and every model scored: %d' % n)
print('effusion-synovitis grade (rows 0-3) x Hoffa-synovitis grade (cols 0-3) counts:')
print(pd.crosstab(E, Hf).to_string())
for th in (1, 2):
    h = (Hf >= th).astype(int); e2 = (E >= 2).astype(int); e1 = (E >= 1).astype(int)
    print('Hoffa>=%d prev %.3f | P(H|eff>=2) %.3f P(H|eff<2) %.3f tet(H, eff>=2) %.2f | P(H|eff>=1) %.3f P(H|eff=0) %.3f tet(H, eff>=1) %.2f' % (
        th, h.mean(), h[e2 == 1].mean(), h[e2 == 0].mean(), tetra(h, e2), h[e1 == 1].mean(), h[e1 == 0].mean(), tetra(h, e1)))
print('RSNA gold58 reference (CLAIMED syn_local.log): Syn prev .466, P(Syn|Eff) .629, P(Syn|no Eff) .217, tet .61')
# c4-shaped OAI-free blends for the Synovitis column: c4 Syn = .8 pct(.8 Raptor + .2 R6) + .2 pct(S6fall); with Raptor absent
# this is .8 pct(R6) + .2 pct(S6fall) (and a .5/.5 sensitivity).
for w, nm in ((.2, 'R6.8S6.2'), (.5, 'R6.5S6.5')):
    M[nm] = pd.DataFrame({c: (1 - w) * pct(M['R6'].loc[common, c].values) + w * pct(M['S6fall'].loc[common, c].values) for c in LAB},
                         index=common)
TG = {'Hoffa>=1 (Raptor map)': (Hf >= 1), 'Hoffa>=2': (Hf >= 2), 'effsyn>=1': (E >= 1), 'effsyn>=2 (=RSNA Eff rubric)': (E >= 2),
      'effsyn>=2 OR Hoffa>=2': (E >= 2) | (Hf >= 2), 'Hoffa>=1 AND effsyn>=1': (Hf >= 1) & (E >= 1)}
TG = {k: v.astype(int) for k, v in TG.items()}
rng = np.random.default_rng(1020)
IDX = [rng.integers(0, n, n) for _ in range(B)]
# gold58 expert (Eff, Syn) joint: Eff0Syn0 18, Eff0Syn1 5, Eff1Syn0 13, Eff1Syn1 22 (MEASURED syn_local.log)
G58 = {(0, 0): 18, (0, 1): 5, (1, 0): 13, (1, 1): 22}
ALPHAS = (.05, .1, .2, .3)
print('per model: AUC of the Syn / Eff / Bak heads for each target; E2 delta = AUC(pct Syn + a pct Eff) - AUC(Syn head)')
for tname, y in TG.items():
    e2 = (E >= 2).astype(int)
    cell = np.array([G58[(a, b)] for a, b in zip(e2, y)], float)
    po = pd.Series(list(zip(e2, y))).value_counts(normalize=True)
    tot = sum(G58.values())
    wg = np.array([G58[(a, b)] / tot / po[(a, b)] for a, b in zip(e2, y)])
    print('== target %s: n %d prev %.3f | tet(target, eff>=2) %.2f | reweighted to the gold58 (Eff, Syn) joint: ESS %.0f' % (
        tname, n, y.mean(), tetra(y, e2), wg.sum() ** 2 / (wg ** 2).sum()))
    for k, P in M.items():
        s = P.loc[common, 'Synovitis'].values; e = P.loc[common, 'Effusion'].values; b = P.loc[common, "Baker's"].values
        us, ue = pct(s), pct(e)
        base = auc(us, y)
        line = '  %-9s Syn %.3f Eff %.3f Bak %.3f rho(S,E) %.3f | in eff>=2: Syn %.3f Eff %.3f | in eff<2: Syn %.3f Eff %.3f |' % (
            k, base, auc(e, y), auc(b, y), spearmanr(s, e).correlation, auc(s[e2 == 1], y[e2 == 1]), auc(e[e2 == 1], y[e2 == 1]),
            auc(s[e2 == 0], y[e2 == 0]), auc(e[e2 == 0], y[e2 == 0]))
        parts = []
        for a in ALPHAS:
            o = us + a * ue
            dd = auc(o, y) - base
            bd = np.array([auc(o[ii], y[ii]) - auc(us[ii], y[ii]) for ii in IDX])
            dw = auc(o, y, wg) - auc(us, y, wg)
            parts.append('a%.2f %+.4f (P %.2f) w %+.4f' % (a, dd, (bd > 0).mean(), dw))
        print(line, ' ; '.join(parts))
# weighted bootstrap P for the c4-shaped blend under the gold58 reweighting, main targets only
print('-- reweighted (gold58 joint) bootstrap for the c4-shaped blend R6.8S6.2 and for D5, a .1 / .2 (B %d)' % B)
for tname in ('Hoffa>=1 (Raptor map)', 'Hoffa>=2', 'effsyn>=2 OR Hoffa>=2'):
    y = TG[tname]; e2 = (E >= 2).astype(int)
    po = pd.Series(list(zip(e2, y))).value_counts(normalize=True); tot = sum(G58.values())
    wg = np.array([G58[(a, b)] / tot / po[(a, b)] for a, b in zip(e2, y)]); p = wg / wg.sum()
    for k in ('R6.8S6.2', 'D5'):
        us = pct(M[k].loc[common, 'Synovitis'].values); ue = pct(M[k].loc[common, 'Effusion'].values)
        out = []
        for a in (.1, .2):
            bd = []
            for _ in range(B):
                ii = rng.choice(n, n, p=p)
                if 0 < y[ii].sum() < n:
                    bd.append(auc(us[ii] + a * ue[ii], y[ii]) - auc(us[ii], y[ii]))
            bd = np.array(bd); out.append('a %.1f %+.4f (sd %.4f, P %.2f)' % (a, bd.mean(), bd.std(), (bd > 0).mean()))
        print('  %-24s %-9s %s' % (tname, k, ' | '.join(out)))
