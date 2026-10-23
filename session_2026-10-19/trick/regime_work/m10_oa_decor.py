# m10_oa_decor.py (10-19 trick lens, research only, CPU): "OA decorrelation" T_a: for each OA label t in {Medial, Lateral, PF},
#   s_t' = pct(s_t) - a * mean(pct(s_o1), pct(s_o2))      (o1, o2 = the other two OA labels; applied to the FINAL output)
# Mechanism: reports couple the OA compartments ("tricompartmental", patellar-facet chondrosis called "lateral compartment", host
# 733491); the expert rubric scores each compartment separately at a strict high-grade >= 1 cm threshold (733343).
# Checks: per-target and per-member deltas on gold58 (expert), bootstrap + split-half for c4, report-regime OOF control (R7 OOF and
# our S6 seeds on non-gold train, 3 readers): the mechanism predicts gold58 > 0 and report OOF < 0.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R, s6seed
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
OA = [L.index(c) for c in ('Medial OA', 'Lateral OA', 'PF OA')]
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv'); S43 = rd('s43_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
R7 = r7.loc[ids, ['p|' + c for c in L]].values
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
P = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); b = w * rr + (1 - w) * pct(R6[:, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])


def T(X, a):
    out = {}
    for t in OA:
        o = [h for h in OA if h != t]
        out[t] = pct(X[:, t]) - a * .5 * (pct(X[:, o[0]]) + pct(X[:, o[1]]))
    return out


def dvec(X, y, a):
    tt = T(X, a); return np.array([auc(tt[t], y[:, t]) - auc(X[:, t], y[:, t]) for t in OA])


A = (.05, .1, .2, .3)
print('gold58 (expert): per-target delta [Medial, Lateral, PF] and macro (= sum / 12)')
for k, X in {'c4': P, 'R6': R6, 'r384': R384, 'r224': R224, 'S6fall': S6, 'R7oof': R7, 's43': S43}.items():
    print('  %-7s ' % k + ' | '.join('a%.2f %s macro %+.5f' % (a, np.round(dvec(X, Y, a), 4).tolist(), dvec(X, Y, a).sum() / 12) for a in (.1, .2)))
rng = np.random.default_rng(11); bb = {a: [] for a in A}
for _ in range(4000):
    ix = rng.integers(0, n, n)
    if any(Y[ix, t].min() == Y[ix, t].max() for t in OA): continue
    for a in A: bb[a].append(dvec(P[ix], Y[ix], a).sum() / 12)
for a in A:
    v = np.array(bb[a]); print('  c4 boot a%.2f: mean %+.5f sd %.5f P>0 %.3f' % (a, v.mean(), v.std(), (v > 0).mean()))
hs = []
for _ in range(2000):
    perm = rng.permutation(n); h = [perm[:29], perm[29:]]
    if any(Y[x, t].min() == Y[x, t].max() for x in h for t in OA): continue
    hs.append([dvec(P[x], Y[x], .1).sum() / 12 for x in h])
hs = np.array(hs)
print('  c4 a.10 split-half: both>0 %.3f  both<0 %.3f  opposite %.3f  mean %+.5f' % (((hs > 0).all(1)).mean(), ((hs < 0).all(1)).mean(),
      ((hs[:, 0] * hs[:, 1]) < 0).mean(), hs.mean()))
# leave-one-out influence: how many single studies does the c4 a.10 gain hang on?
base = dvec(P, Y, .1).sum() / 12; infl = []
for i in range(n):
    ix = np.delete(np.arange(n), i); infl.append(dvec(P[ix], Y[ix], .1).sum() / 12)
infl = np.array(infl); print('  c4 a.10 full %+.5f; leave-one-out min %+.5f max %+.5f; studies whose removal drops it below 0: %d' % (base, infl.min(), infl.max(), (infl <= 0).sum()))
print('report-regime OOF control (non-gold train): macro delta (sum over the 3 OA targets / 12)')
LB = R + '/_labels/'
srcs = {'v4blend': (LB + 'rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', .25), 'sol56': (LB + 'rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', None),
        'jevhyb': (R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv', None)}
oof = {'R7oof': r7[['p|' + c for c in L]].set_axis(L, axis=1)}
for sd in (1044, 1045): oof['s%d' % sd] = s6seed(sd)[0]
allg = set(g.StudyInstanceUID)
for k, (p, na) in srcs.items():
    lab = pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L].astype(float)
    for mk, O in oof.items():
        idx = [u for u in O.index if u not in allg and u in lab.index]
        X = O.loc[idx, L].values; yl = lab.loc[idx, L].values
        tt = {a: T(X, a) for a in (.1, .2)}; r = {a: 0. for a in (.1, .2)}
        for t in OA:
            m = ~np.isnan(yl[:, t])
            if na is not None: m &= ~np.isclose(yl[:, t], na)
            y = (yl[m, t] >= .5).astype(int)
            for a in (.1, .2): r[a] += auc(tt[a][t][m], y) - auc(X[m, t], y)
        print('  %-7s %-6s a.10 %+.5f  a.20 %+.5f' % (k, mk, r[.1] / 12, r[.2] / 12))
