# box_coupling (10-19 ex, Claude). Read-only; prints AGGREGATES ONLY (n, prevalence, phi, tetrachoric, rho, AUC deltas, P,
# ESS). Never writes a file, never prints an id, a row or a per-knee value. Piped via ssh stdin into python -I.
# The line "G74 = {...}" at the top is filled in locally by make_box_script.py from gold74 2x2 counts (local aggregates).
#
# For every pair of OAI-covered labels (ACL, both menisci, OA trio, Effusion, Synovitis, Baker's):
#   expert coupling : phi + tetrachoric on binarised targets (B1 = rubric-like primary, B2 = alternative), Spearman on graded T123
#   model coupling  : Spearman of each OAI-free model's predictions on the same knees
#   directional pair decorrelation, per model: out_t = pct(f_t) - a * pct(f_o), dAUC(t) for a in (-.1, .1, .2), paired
#   bootstrap P>0 (B 400); theory q = corr(ns f_o, Y_t) / corr(ns f_t, Y_t) vs rho_ns
#   coupling-matched reweighting (skeptic method, by the 2-label cell): R1 = gold74 4-cell joint (+.5 smoothed);
#   R2 = OAI marginals with gold74 odds ratio (coupling only); R2lo / R2hi = gold74 log OR -/+ 1 sd.
# Binarisation B1 (B2): ACL tear >= .5 (complete = 1); menisci tear = 1 (>= .2 incl. intrasubstance signal);
#   OA = P_FT (any compartment cartilage subregion with full-thickness loss, MOAKS m_cm_*) (T123 composite >= .5);
#   Effusion grade >= 2 i.e. >= .8 (grade >= 1); Synovitis Hoffa grade >= 1 i.e. >= .333 (grade >= 2); Baker's 1 (same).
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm, multivariate_normal
from scipy.optimize import brentq
H = os.path.expanduser('~/oai_build')
L9 = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
MODELS = {'D5mean_s1043': H + '/teacher/teacher_D5_mean_v00.csv', 's1044_mix50': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv',
          's1045_288': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv'}
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
BOOT = int(os.environ.get('BOOT', '400')); RB = int(os.environ.get('RBOOT', '200'))


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def tetra(a, b):
    c11 = ((a == 1) & (b == 1)).sum() + .5; c10 = ((a == 1) & (b == 0)).sum() + .5
    c01 = ((a == 0) & (b == 1)).sum() + .5; c00 = ((a == 0) & (b == 0)).sum() + .5; N = c11 + c10 + c01 + c00
    h = norm.ppf(1 - (c11 + c10) / N); k = norm.ppf(1 - (c11 + c01) / N); p11 = c11 / N
    f = lambda r: multivariate_normal.cdf([-h, -k], mean=[0, 0], cov=[[1, r], [r, 1]]) - p11
    if f(-.995) > 0: return -.995
    if f(.995) < 0: return .995
    return float(brentq(f, -.995, .995, xtol=1e-4))


def phi(a, b):
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')


# ---------------- labels ----------------
T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
B1 = pd.DataFrame(index=T.index); B2 = pd.DataFrame(index=T.index)
thr = lambda c, x: np.where(T[c].notna(), (T[c] >= x - 1e-6).astype(float), np.nan)
B1['ACL'] = thr('ACL', .5); B2['ACL'] = thr('ACL', 1.)
for c in ('Medial Meniscus', 'Lateral Meniscus'):
    B1[c] = thr(c, 1.); B2[c] = thr(c, .2)
for c in OA:
    cc = S.reindex(T.index)[['m_cm_' + r for r in COMP[c]]].astype(float).values
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10); fin = np.isfinite(cc)
    y = ((ft >= 1) & fin).any(1).astype(float); y[~fin.any(1)] = np.nan
    B1[c] = y; B2[c] = thr(c, .5)
B1['Effusion'] = thr('Effusion', .8); B2['Effusion'] = thr('Effusion', .3)
B1['Synovitis'] = thr('Synovitis', 1 / 3); B2['Synovitis'] = thr('Synovitis', 2 / 3)
B1["Baker's"] = thr("Baker's", 1.); B2["Baker's"] = thr("Baker's", 1.)
P = {}
for k, f in MODELS.items():
    d = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    P[k] = d[L9]
common = T.index
for k in P: common = common.intersection(P[k].dropna().index)
print('OAI knees with all 3 model predictions and a T123 row: %d' % len(common))
print('B1 coverage / prevalence: %s' % {c: (int(B1.loc[common, c].notna().sum()), round(float(B1.loc[common, c].mean()), 3)) for c in L9})
print('B2 coverage / prevalence: %s' % {c: (int(B2.loc[common, c].notna().sum()), round(float(B2.loc[common, c].mean()), 3)) for c in L9})
sys.stdout.flush()


def cellkey(a, b):
    return (a * 2 + b).astype(int)   # 0 = 00, 1 = 01, 2 = 10, 3 = 11 (a first)


def or_cells(p1, p2, lor):
    """2x2 joint with marginals p1 (a), p2 (b) and log odds ratio lor; returns [p00, p01, p10, p11]."""
    lo = max(0., p1 + p2 - 1) + 1e-9; hi = min(p1, p2) - 1e-9
    g = lambda p11: np.log(p11) + np.log(1 - p1 - p2 + p11) - np.log(p1 - p11) - np.log(p2 - p11) - lor
    p11 = brentq(g, lo, hi)
    return np.array([1 - p1 - p2 + p11, p2 - p11, p1 - p11, p11])


def weighted_eval(Ft, Fo, yt, key, target, a_list=(.1, .2), B=200, seed=0):
    po = np.bincount(key, minlength=4) / len(key)
    wc = target / np.maximum(po, 1e-12); w = wc[key]; p = w / w.sum(); ess = 1 / (p ** 2).sum()
    rng = np.random.default_rng(seed); n = len(key); D = {x: [] for x in a_list}
    outs = {x: pct(Ft) - x * pct(Fo) for x in a_list}
    for _ in range(B):
        ii = rng.choice(n, n, p=p); y = yt[ii]
        if y.sum() < 5 or (len(y) - y.sum()) < 5: continue
        b0 = auc(Ft[ii], y)
        for x in a_list: D[x].append(auc(outs[x][ii], y) - b0)
    return ess, {x: (float(np.mean(v)), float(np.mean(np.array(v) > 0))) for x, v in D.items()}


RES = {}
for a, b in itertools.combinations(L9, 2):
    m = B1.loc[common, [a, b]].dropna(); ids = m.index; ya = m[a].values.astype(int); yb = m[b].values.astype(int)
    e = {'n': len(ids), 'prev': [round(float(ya.mean()), 4), round(float(yb.mean()), 4)], 'phi': phi(ya, yb), 'tet': tetra(ya, yb)}
    m2 = B2.loc[common, [a, b]].dropna(); e['B2'] = {'n': len(m2), 'phi': phi(m2[a].values, m2[b].values),
                                                    'tet': tetra(m2[a].values.astype(int), m2[b].values.astype(int))}
    gg = T.loc[common, [a, b]].dropna(); e['graded_rho'] = float(spearmanr(gg[a], gg[b]).correlation); e['graded_n'] = len(gg)
    e['model_rho'] = {k: float(spearmanr(P[k].loc[ids, a], P[k].loc[ids, b]).correlation) for k in P}
    e['model_rho_all'] = {k: float(spearmanr(P[k].loc[common, a], P[k].loc[common, b]).correlation) for k in P}
    key = cellkey(ya, yb)
    c = G74[a + ' | ' + b]   # [c00, c01, c10, c11] gold74, a first
    cs = np.array(c, float) + .5; t1 = cs / cs.sum()
    lor = np.log(cs[3] * cs[0] / (cs[1] * cs[2])); lsd = np.sqrt((1 / cs).sum())
    p1, p2 = ya.mean(), yb.mean()
    targets = {'R1': t1, 'R2': or_cells(p1, p2, lor), 'R2lo': or_cells(p1, p2, lor - lsd), 'R2hi': or_cells(p1, p2, lor + lsd)}
    e['g74_lor'] = float(lor); e['g74_lor_sd'] = float(lsd)
    e['oai_lor'] = float(np.log(((key == 3).sum() + .5) * ((key == 0).sum() + .5) / (((key == 1).sum() + .5) * ((key == 2).sum() + .5))))
    e['dir'] = {}
    for t, o in ((a, b), (b, a)):
        yt = ya if t == a else yb
        dd = {}
        for k in P:
            Ft = P[k].loc[ids, t].values.astype(float); Fo = P[k].loc[ids, o].values.astype(float)
            base = auc(Ft, yt); outs = {x: pct(Ft) - x * pct(Fo) for x in (-.1, .1, .2)}
            dl = {x: auc(outs[x], yt) - base for x in outs}
            rng = np.random.default_rng(1); bd = {x: [] for x in outs}
            for _ in range(BOOT):
                ii = rng.integers(0, len(yt), len(yt)); y = yt[ii]
                if y.sum() < 3 or y.sum() > len(y) - 3: continue
                b0 = auc(Ft[ii], y)
                for x in outs: bd[x].append(auc(outs[x][ii], y) - b0)
            zt = ns(Ft); zo = ns(Fo); ct = np.corrcoef(zt, yt)[0, 1]; co = np.corrcoef(zo, yt)[0, 1]
            rr = {}
            for nm, tg in targets.items():
                ess, wd = weighted_eval(Ft, Fo, yt, key, tg, B=RB, seed=2)
                rr[nm] = {'ess': round(float(ess)), 'd.1': wd[.1][0], 'P.1': wd[.1][1], 'd.2': wd[.2][0], 'P.2': wd[.2][1]}
            dd[k] = {'base': float(base), 'd': {str(x): float(v) for x, v in dl.items()},
                     'P': {str(x): float(np.mean(np.array(v) > 0)) for x, v in bd.items()},
                     'sd': {str(x): float(np.std(v)) for x, v in bd.items()},
                     'rho_ns': float(np.corrcoef(zt, zo)[0, 1]), 'q': float(co / ct) if ct != 0 else float('nan'), 'rw': rr}
        e['dir'][t + ' <- ' + o] = dd
    RES[a + ' | ' + b] = e
    mr = np.mean(list(e['model_rho'].values()))
    line = '%-34s n %4d prev %.3f/%.3f | label phi %+.3f tet %+.3f (B2 phi %+.3f tet %+.3f) graded rho %+.3f | model rho %+.3f [%s]' % (
        a + ' | ' + b, e['n'], *e['prev'], e['phi'], e['tet'], e['B2']['phi'], e['B2']['tet'], e['graded_rho'], mr,
        '/'.join('%+.2f' % v for v in e['model_rho'].values()))
    print(line)
    for dk, dd in e['dir'].items():
        print('    %-36s ' % dk + ' || '.join('%s d-.1 %+.4f d.1 %+.4f (P %.2f) d.2 %+.4f (P %.2f) gap %+.2f R2 d.1 %+.4f d.2 %+.4f R1 d.1 %+.4f (ess %d)' % (
            k[:5], v['d']['-0.1'], v['d']['0.1'], v['P']['0.1'], v['d']['0.2'], v['P']['0.2'], v['rho_ns'] - v['q'], v['rw']['R2']['d.1'],
            v['rw']['R2']['d.2'], v['rw']['R1']['d.1'], v['rw']['R1']['ess']) for k, v in dd.items()))
    sys.stdout.flush()
print('JSON_BEGIN')
print(json.dumps(RES))
print('JSON_END')
