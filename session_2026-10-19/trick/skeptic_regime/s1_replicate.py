# s1_replicate.py (skeptic, CPU, research only): independent re-derivation of the OA-decorrelation numbers on gold58,
# with sklearn AUC and pandas average-tie pct (kernel convention), plus selection-aware nulls:
#   (a) c4 rebuild and per-member deltas at a .1 / .2
#   (b) all 220 label triples (within-triple decor) and all 66 pairs: where does the OA trio rank?
#   (c) study-level decomposition + greedy leave-k-out
#   (d) expert vs report phi among OA labels on the same 58, and c4 prediction rank-corr among OA columns
import itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
G = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
R = 'C:/Users/eeshs/Downloads/rsna-knee'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
print('gold58 n', n, 'OA positives', Y[:, 4:7].sum(0).tolist())
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv'); S43 = rd('s43_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
r224ids = pd.read_csv(G + '/gold58_r224_ids.csv', dtype=str)
print('r224 ids file cols', list(r224ids.columns), 'order == gold58 order:', r224ids.iloc[:, 0].tolist() == ids)
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
R7 = r7.loc[ids, ['p|' + c for c in L]].values
WR = {'Medial OA': .5, 'Lateral OA': .2, 'PF OA': .5, 'Synovitis': .8, "Baker's": .5, 'Lateral Meniscus': .4, 'Fracture': .65}
F = np.zeros((n, 12))
for j, c in enumerate(L):
    rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); w = WR.get(c, .5)
    b = w * rr + (1 - w) * pct(R6[:, j]); F[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
auc = lambda s, y: roc_auc_score(y, s)
base = np.array([auc(F[:, j], Y[:, j]) for j in range(12)])
print('c4 gold58 macro %.5f per-label %s' % (base.mean(), np.round(base, 4).tolist()))
OA = [4, 5, 6]


def decor(X, grp, a, sgn=-1):
    O = X.copy(); U = {t: pct(X[:, t]) for t in grp}
    for t in grp:
        o = [h for h in grp if h != t]
        O[:, t] = U[t] + sgn * a * np.mean([U[h] for h in o], 0)
    return O


def macro_delta(X, Yy, grp, a, sgn=-1):
    O = decor(X, grp, a, sgn)
    return sum(auc(O[:, t], Yy[:, t]) - auc(X[:, t], Yy[:, t]) for t in grp) / 12


print('\n(a) per-member OA-decor macro delta (expert gold58)')
for k, X in {'c4': F, 'R6': R6, 'r384': R384, 'r224': R224, 'S6fall': S6, 'R7oof': R7, 's43': S43}.items():
    print('  %-7s a.1 %+.5f  a.2 %+.5f  a.3 %+.5f' % (k, macro_delta(X, Y, OA, .1), macro_delta(X, Y, OA, .2), macro_delta(X, Y, OA, .3)))
# member correlation: how independent are the 7 'confirmations'? Spearman of OA columns between members
print('  member Spearman on OA columns vs c4 (mean over 3):', {k: round(np.mean([pd.Series(X[:, t]).corr(pd.Series(F[:, t]), method='spearman') for t in OA]), 3)
                                                             for k, X in {'R6': R6, 'r384': R384, 'S6fall': S6, 'R7oof': R7, 's43': S43}.items()})

print('\n(b) selection-aware null: within-group decor over ALL label triples / pairs (c4, gold58)')
for a in (.1, .2):
    tri = {t: macro_delta(F, Y, list(t), a) for t in itertools.combinations(range(12), 3)}
    v = np.array(list(tri.values())); oa = tri[(4, 5, 6)]
    rank = int((v > oa).sum()) + 1
    top = sorted(tri.items(), key=lambda z: -z[1])[:6]
    print('  a %.1f triples (220): OA %+.5f rank %d; null mean %+.5f sd %.5f; >= +.001: %d; top6 %s' % (
        a, oa, rank, v.mean(), v.std(), int((v >= .001).sum()), [([L[i][:6] for i in k], round(d, 5)) for k, d in top]))
    pr = {p: macro_delta(F, Y, list(p), a) for p in itertools.combinations(range(12), 2)}
    vp = np.array(list(pr.values()))
    toppr = sorted(pr.items(), key=lambda z: -z[1])[:6]
    print('  a %.1f pairs (66): OA pairs %s; null mean %+.5f sd %.5f; top6 %s' % (
        a, {L[p[0]][:3] + '-' + L[p[1]][:3]: round(pr[p], 5) for p in [(4, 5), (4, 6), (5, 6)]}, vp.mean(), vp.std(),
        [([L[i][:6] for i in k], round(d, 5)) for k, d in toppr]))
    # same null with the positive sign (borrowing) for completeness
    trib = np.array([macro_delta(F, Y, list(t), a, +1) for t in itertools.combinations(range(12), 3)])
    print('  a %.1f triples borrow(+): mean %+.5f max %+.5f' % (a, trib.mean(), trib.max()))

print('\n(c) study-level decomposition at a .2 (c4): pairwise ordering changes per OA label')
O = decor(F, OA, .2)
contrib = np.zeros(n)
for t in OA:
    y = Y[:, t]; P_, N_ = np.where(y == 1)[0], np.where(y == 0)[0]; npairs = len(P_) * len(N_)
    before = (F[P_, t][:, None] > F[N_, t][None, :]).astype(float) + .5 * (F[P_, t][:, None] == F[N_, t][None, :])
    after = (O[P_, t][:, None] > O[N_, t][None, :]).astype(float) + .5 * (O[P_, t][:, None] == O[N_, t][None, :])
    d = (after - before) / npairs / 12
    print('  %-10s pos %d: pairs fixed %d broken %d net %+.1f pairs (label AUC %+.4f)' % (L[t], len(P_), int((d > 0).sum()), int((d < 0).sum()),
          (after - before).sum(), (after - before).sum() / npairs))
    np.add.at(contrib, P_, d.sum(1) / 2); np.add.at(contrib, N_, d.sum(0) / 2)
order = np.argsort(-contrib)
print('  total %+.5f; top-5 study contributions %s; share of top 3: %.2f' % (contrib.sum(), np.round(contrib[order[:5]], 5).tolist(),
      contrib[order[:3]].sum() / contrib.sum()))
print('  top-5 studies: expert OA labels', [Y[i, 4:7].tolist() for i in order[:5]])
keep = np.arange(n); out = []
for k in range(1, 7):
    best = None
    for i in keep:
        ix = keep[keep != i]
        if any(Y[ix, t].min() == Y[ix, t].max() for t in OA): continue
        val = macro_delta(F[ix], Y[ix], OA, .2)
        if best is None or val < best[0]: best = (val, i)
    keep = keep[keep != best[1]]; out.append(round(best[0], 5))
print('  greedy worst leave-k-out (k=1..6) at a .2:', out)
keep = np.arange(n); out = []
for k in range(1, 7):
    best = None
    for i in keep:
        ix = keep[keep != i]
        val = macro_delta(F[ix], Y[ix], OA, .1)
        if best is None or val < best[0]: best = (val, i)
    keep = keep[keep != best[1]]; out.append(round(best[0], 5))
print('  greedy worst leave-k-out (k=1..6) at a .1:', out)

print('\n(d) coupling: phi among OA labels on the same 58 (expert vs report), and c4 prediction Spearman')
labs = {'v4blend': R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
        'sol56': R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv'}
phi = lambda a, b: np.corrcoef(a, b)[0, 1]
prs = [(4, 5), (4, 6), (5, 6)]
print('  expert  ', {L[a][:3] + '-' + L[b][:3]: round(phi(Y[:, a], Y[:, b]), 3) for a, b in prs})
for k, p in labs.items():
    Z = pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(ids)[L].values.astype(float)
    Zb = (Z >= .5).astype(int)
    print('  %-8s' % k, {L[a][:3] + '-' + L[b][:3]: round(phi(Zb[:, a], Zb[:, b]), 3) for a, b in prs},
          'agree w expert per OA label', [round((Zb[:, t] == Y[:, t]).mean(), 3) for t in OA])
print('  c4 pred Spearman', {L[a][:3] + '-' + L[b][:3]: round(pd.Series(F[:, a]).corr(pd.Series(F[:, b]), method='spearman'), 3) for a, b in prs})
rng = np.random.default_rng(5); bs = []
for _ in range(4000):
    ix = rng.integers(0, n, n); bs.append([phi(Y[ix, a], Y[ix, b]) for a, b in prs])
bs = np.array(bs)
print('  expert phi bootstrap 90%% CI', {L[a][:3] + '-' + L[b][:3]: np.round(np.nanpercentile(bs[:, k], [5, 95]), 3).tolist() for k, (a, b) in enumerate(prs)})
