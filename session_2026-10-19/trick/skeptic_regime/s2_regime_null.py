# s2_regime_null.py (skeptic, CPU, research only):
#  (1) report-regime null: R7 OOF on random 58-study subsets of non-gold train with report labels; how often does the OA-decor gain
#      reach R7's gold58 (expert) value? This asks whether gold58's +gain is outside what a 58-study report-regime sample produces.
#  (2) same-study paired regime contrast (expert minus report labels on the same 58, same predictions), bootstrap.
#  (3) per ordered helper (t <- -h) deltas on gold58, expert vs report labels: does the per-direction pattern follow the coupling gap?
#  (4) single-label vs macro: how much of the gain is "subtract general abnormality" (global non-OA) vs OA-specific, per target.
import itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
G = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
R = 'C:/Users/eeshs/Downloads/rsna-knee'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
auc = lambda s, y: roc_auc_score(y, s)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
R7 = r7.loc[ids, ['p|' + c for c in L]].values
WR = {'Medial OA': .5, 'Lateral OA': .2, 'PF OA': .5, 'Synovitis': .8, "Baker's": .5, 'Lateral Meniscus': .4, 'Fracture': .65}
F = np.zeros((n, 12))
for j, c in enumerate(L):
    rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); w = WR.get(c, .5)
    b = w * rr + (1 - w) * pct(R6[:, j]); F[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
OA = [4, 5, 6]


def dOA(X, Yy, a, mask=None):
    d = 0.
    for t in OA:
        o = [h for h in OA if h != t]
        m = np.ones(len(X), bool) if mask is None else mask[:, t]
        y = Yy[m, t]
        if y.min() == y.max(): return np.nan
        s = pct(X[m, t]) - a * .5 * (pct(X[m, o[0]]) + pct(X[m, o[1]]))
        d += auc(s, y) - auc(X[m, t], y)
    return d / 12


labsrc = {'sol56': (R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', None),
          'v4blend': (R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', .25)}
print('(1) report-regime 58-subset null for R7 OOF (non-gold train)')
obs = {a: dOA(R7, Y, a) for a in (.1, .2)}
print('  R7 OOF gold58 expert: a.1 %+.5f a.2 %+.5f' % (obs[.1], obs[.2]))
allg = set(g.StudyInstanceUID)
rng = np.random.default_rng(2026)
for k, (p, na) in labsrc.items():
    lab = pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L].astype(float)
    idx = [u for u in r7.index if u not in allg and u in lab.index]
    X = r7.loc[idx, ['p|' + c for c in L]].values; Z = lab.loc[idx, L].values
    ok = ~np.isnan(Z[:, OA]).any(1)
    if na is not None: ok &= ~np.isclose(Z[:, OA], na).any(1)
    X, Zb = X[ok], (Z[ok] >= .5).astype(int)
    full = {a: dOA(X, Zb, a) for a in (.1, .2)}
    res = {a: [] for a in (.1, .2)}
    for _ in range(3000):
        ix = rng.choice(len(X), n, replace=False)
        for a in (.1, .2):
            v = dOA(X[ix], Zb[ix], a)
            if np.isfinite(v): res[a].append(v)
    for a in (.1, .2):
        v = np.array(res[a])
        print('  %-7s n_pool %d full %+.5f | 58-subsets mean %+.5f sd %.5f P(>= gold58 obs %+.5f) %.4f  P(>0) %.3f' % (
            k, len(X), full[a], v.mean(), v.std(), obs[a], (v >= obs[a]).mean(), (v > 0).mean()))
    # same, but subsets matched on report-label OA prevalence (v4/sol56 prevalence on gold58 itself)
print('\n(2) same 58 studies, same predictions: expert minus report delta, bootstrap')
lab = pd.read_csv(labsrc['sol56'][0], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(ids)[L].values.astype(float)
Zg = (lab >= .5).astype(int)
for nm, X in (('c4', F), ('R7oof', R7)):
    for a in (.1, .2):
        e, r = dOA(X, Y, a), dOA(X, Zg, a)
        bs = []
        for _ in range(3000):
            ix = rng.integers(0, n, n)
            de, dr = dOA(X[ix], Y[ix], a), dOA(X[ix], Zg[ix], a)
            if np.isfinite(de) and np.isfinite(dr): bs.append((de, dr))
        bs = np.array(bs); diff = bs[:, 0] - bs[:, 1]
        print('  %-5s a.%d expert %+.5f report(sol56) %+.5f diff %+.5f | boot diff mean %+.5f sd %.5f P>0 %.3f | boot expert P>0 %.3f' % (
            nm, int(a * 10), e, r, e - r, diff.mean(), diff.std(), (diff > 0).mean(), (bs[:, 0] > 0).mean()))
print('\n(3) per ordered helper t <- -a*pct(h) on gold58 (label AUC delta x 1, a .2), expert vs sol56 labels on the same studies')
for nm, X in (('c4', F), ('R7oof', R7)):
    for t in OA:
        row = []
        for h in OA:
            if h == t: continue
            s = pct(X[:, t]) - .2 * pct(X[:, h])
            row.append('%s<-%s exp %+.4f rep %+.4f' % (L[t][:3], L[h][:3], auc(s, Y[:, t]) - auc(X[:, t], Y[:, t]),
                                                     auc(s, Zg[:, t]) - auc(X[:, t], Zg[:, t])))
        print('  %-5s ' % nm + ' | '.join(row))
print('\n(4) per OA target, a .2: helper = other-2-OA vs helper = mean of 9 non-OA labels vs helper = Effusion+ACL+Fracture (arbitrary)')
NON = [j for j in range(12) if j not in OA]
for nm, X in (('c4', F), ('R7oof', R7)):
    out = []
    for t in OA:
        o = [h for h in OA if h != t]
        base = auc(X[:, t], Y[:, t])
        s1 = pct(X[:, t]) - .2 * np.mean([pct(X[:, h]) for h in o], 0)
        s2 = pct(X[:, t]) - .2 * np.mean([pct(X[:, h]) for h in NON], 0)
        out.append('%s base %.4f oa %+.4f nonOA %+.4f' % (L[t][:3], base, auc(s1, Y[:, t]) - base, auc(s2, Y[:, t]) - base))
    print('  %-5s ' % nm + ' | '.join(out))
# (5) how many nonsense single helpers (non-OA label h) help Lateral OA / PF OA at a .2 on c4? (is the target just 'subtract anything'?)
print('\n(5) c4, a .2: Lateral OA and PF OA minus EACH single label h (label AUC delta)')
for t in (5, 6):
    print('  %-10s ' % L[t] + ', '.join('%s %+.4f' % (L[h][:6], auc(pct(F[:, t]) - .2 * pct(F[:, h]), Y[:, t]) - auc(F[:, t], Y[:, t])) for h in range(12) if h != t))
