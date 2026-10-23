# recal_box_gapmatch (10-20 night, Claude, TRACK 1). Read-only CPU; prints AGGREGATES ONLY (per model x regime means of rho_ns, q, gap and
# AUC deltas over weighted resamples). Never writes a file, never prints an id, a row or a per-knee value. Piped via ssh stdin into python -I.
# Question: the E-OA public read says OAI proxies must be read at the regime where the PROXY's own gap equals c4's gap (the trio's
# coupling-matched rulers underestimated because proxy OA rho < c4 OA rho). Do the killed meniscus-type directions change sign when read
# the same way? For each direction: pairwise OR-matched reweighting (OAI marginals, gold74 log OR shifted by k sd, k in -2..+1), and at
# each regime the proxy's rho_ns, q, gap and dAUC(target) for a in (-.1, -.05, .05, .1, .15, .2). The gold74 2x2 counts below are local
# aggregates (session_2026-10-19/ex/local_coupling.json).
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
G74 = {'Medial Meniscus | Lateral Meniscus': [25, 13, 19, 16], 'Medial Meniscus | Lateral OA': [30, 7, 26, 7],
       'Lateral Meniscus | Lateral OA': [39, 3, 18, 11], 'Medial Meniscus | Medial OA': [33, 4, 16, 18], 'Lateral OA | PF OA': [37, 20, 5, 9]}
DIRS = [('Lateral Meniscus', 'Medial Meniscus'), ('Medial Meniscus', 'Lateral Meniscus'), ('Lateral OA', 'Medial Meniscus'),
        ('Lateral OA', 'Lateral Meniscus'), ('Medial OA', 'Medial Meniscus'), ('Lateral OA', 'PF OA')]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'], 'PF OA': ['mp', 'lp', 'amf', 'alf']}
MODELS = {'D5': H + '/teacher/teacher_D5_mean_v00.csv', 's1044': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv',
          's1045': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv'}
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
B = int(os.environ.get('B', '150'))
AV = (-.1, -.05, .05, .1, .15, .2); KS = (-2., -1.5, -1., -.5, 0., .5, 1.)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def or_cells(p1, p2, lor):
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([1 - p1 - p2 + p11, p2 - p11, p1 - p11, p11])


T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
B1 = pd.DataFrame(index=T.index)
thr = lambda c, x: np.where(T[c].notna(), (T[c] >= x - 1e-6).astype(float), np.nan)
for c in ('Medial Meniscus', 'Lateral Meniscus'): B1[c] = thr(c, 1.)
for c in OA:
    cc = S.reindex(T.index)[['m_cm_' + r for r in COMP[c]]].astype(float).values
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
    y = ((ft >= 1) & fin).any(1).astype(float); y[~fin.any(1)] = np.nan; B1[c] = y
LBL = ['Medial Meniscus', 'Lateral Meniscus'] + OA
Pm = {}
for k, f in MODELS.items():
    d = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID'); Pm[k] = d[LBL]
common = T.index
for k in Pm: common = common.intersection(Pm[k].dropna().index)
print('OAI knees with all 3 models: %d' % len(common)); sys.stdout.flush()
rng = np.random.default_rng(20261020)
for t, o in DIRS:
    pair = [p for p in G74 if set(p.split(' | ')) == {t, o}][0]; a_, b_ = pair.split(' | ')
    m = B1.loc[common, [a_, b_]].dropna(); ids = m.index; ya = m[a_].values.astype(int); yb = m[b_].values.astype(int)
    key = (ya * 2 + yb).astype(int); po = np.bincount(key, minlength=4) / len(key)
    cs = np.array(G74[pair], float) + .5; lor = np.log(cs[3] * cs[0] / (cs[1] * cs[2])); lsd = np.sqrt((1 / cs).sum())
    yt = ya if t == a_ else yb
    print('== %s <- %s | n %d | OAI log OR %+.2f | gold74 log OR %+.2f (sd %.2f)' % (t, o, len(ids), np.log(((key == 3).sum() + .5) * ((key == 0).sum() + .5) / (((key == 1).sum() + .5) * ((key == 2).sum() + .5))), lor, lsd))
    for k_, P_ in Pm.items():
        Ft = P_.loc[ids, t].values.astype(float); Fo = P_.loc[ids, o].values.astype(float)
        outs = {x: pct(Ft) - x * pct(Fo) for x in AV}
        for ks in KS:
            tg = or_cells(ya.mean(), yb.mean(), lor + ks * lsd); w = (tg / np.maximum(po, 1e-12))[key]; p = w / w.sum(); ess = 1 / (p ** 2).sum()
            R, Q, Dd = [], [], {x: [] for x in AV}
            for _ in range(B):
                ii = rng.choice(len(ids), len(ids), p=p); y = yt[ii]
                if y.sum() < 5 or y.sum() > len(y) - 5: continue
                zt = ns(Ft[ii]); zo = ns(Fo[ii]); ct = np.cov(zt, y)[0, 1]
                R.append(np.corrcoef(zt, zo)[0, 1]); Q.append(np.cov(zo, y)[0, 1] / ct)
                b0 = auc(Ft[ii], y)
                for x in AV: Dd[x].append(auc(outs[x][ii], y) - b0)
            print('  %-6s k %+.1f ess %4d | rho %.3f q %.3f gap %+.3f | dAUC_t ' % (k_, ks, ess, np.mean(R), np.mean(Q), np.mean(R) - np.mean(Q)) +
                  ' '.join('a%+.2f %+.4f' % (x, np.mean(Dd[x])) for x in AV))
            sys.stdout.flush()
print('DONE')
