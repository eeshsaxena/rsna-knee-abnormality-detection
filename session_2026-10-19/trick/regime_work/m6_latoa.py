# m6_latoa.py (10-19 trick lens, research only, CPU): the one pre-registered NEGATIVE-weight pair, Lateral OA <- -PF OA.
# Mechanism (host 733491, quoted by avg-HU and answered by the host): reports call lateral PATELLAR FACET chondrosis "lateral compartment"
# chondromalacia; the expert rubric scores the lateral TIBIOFEMORAL compartment only. Report-trained models therefore leak PF cartilage
# signal into Lateral OA. Prediction: (a) label level, report Lateral OA is more coupled to PF OA than expert Lateral OA is;
# (b) s' = pct(LatOA) - a pct(PFOA) helps on expert labels for most members; (c) it does NOT help (or hurts) on report labels.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R, s6seed
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
jl, jp, jm = L.index('Lateral OA'), L.index('PF OA'), L.index('Medial OA')
def phi(a, b):
    a = a.astype(float); b = b.astype(float); return np.corrcoef(a, b)[0, 1]
LB = R + '/_labels/'
srcs = {'v4blend': (LB + 'rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', .25), 'sol56': (LB + 'rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', None),
        'jevhyb': (R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv', None)}
print('(a) label-level coupling Lateral OA ~ PF OA (phi) and P(LatOA+ | PF+) vs P(LatOA+ | PF-)')
print('   expert gold58: phi %.3f  P(L|PF+) %.2f  P(L|PF-) %.2f  (n PF+ %d)' % (phi(Y[:, jl], Y[:, jp]), Y[Y[:, jp] == 1, jl].mean(), Y[Y[:, jp] == 0, jl].mean(), Y[:, jp].sum()))
allg = set(g.StudyInstanceUID)
for k, (p, na) in srcs.items():
    lab = pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L].astype(float)
    for part, idx in [('gold58', ids), ('non-gold', [u for u in lab.index if u not in allg])]:
        x = lab.reindex(idx)
        m = x[['Lateral OA', 'PF OA']].notna().all(1)
        if na is not None: m &= ~np.isclose(x['Lateral OA'], na) & ~np.isclose(x['PF OA'], na)
        a = (x.loc[m, 'Lateral OA'] >= .5).values; b = (x.loc[m, 'PF OA'] >= .5).values
        print('   %-8s %-8s n %4d phi %.3f  P(L|PF+) %.2f  P(L|PF-) %.2f' % (k, part, m.sum(), phi(a, b), a[b].mean(), a[~b].mean()))
# (b) members on gold58
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv'); NA = rd('nartaa_gold58.csv'); S43 = rd('s43_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
R7 = r7.loc[ids, ['p|' + c for c in L]].values
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
P = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); b = w * rr + (1 - w) * pct(R6[:, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
M = {'c4': P, 'R6': R6, 'r384': R384, 'r224': R224, 'S6fall': S6, 'R7oof': R7, 's43': S43}
A = [.05, .1, .2, .3]
f = lambda X, a: pct(X[:, jl]) - a * pct(X[:, jp])
print('(b) expert gold58 Lateral OA AUC delta of pct(LatOA) - a pct(PFOA)   [macro = /12]')
for k, X in M.items():
    b0 = auc(X[:, jl], Y[:, jl])
    print('   %-7s base %.4f  ' % (k, b0) + '  '.join('a%.2f %+.4f' % (a, auc(f(X, a), Y[:, jl]) - b0) for a in A) +
          '   | rho(LatOA,PFOA) within expert-LatOA-neg %.2f' % pd.Series(X[Y[:, jl] == 0, jl]).corr(pd.Series(X[Y[:, jl] == 0, jp]), method='spearman'))
rng = np.random.default_rng(1019); dd = []; agree = []
for _ in range(4000):
    ix = rng.integers(0, n, n)
    if Y[ix, jl].min() == Y[ix, jl].max(): continue
    dd.append(auc(f(P[ix], .1), Y[ix, jl]) - auc(P[ix, jl], Y[ix, jl]))
dd = np.array(dd) / 12
print('   c4 a.10 boot macro: mean %+.5f sd %.5f P>0 %.3f' % (dd.mean(), dd.std(), (dd > 0).mean()))
hs = []
for _ in range(1000):
    perm = rng.permutation(n); h = [perm[:29], perm[29:]]
    if any(Y[x, jl].min() == Y[x, jl].max() for x in h): continue
    hs.append([auc(f(P[x], .1), Y[x, jl]) - auc(P[x, jl], Y[x, jl]) for x in h])
hs = np.array(hs); print('   c4 a.10 split-half: sign agreement %.3f  both>0 %.3f  mean %+.4f (label units)' % ((np.sign(hs[:, 0]) == np.sign(hs[:, 1])).mean(), ((hs > 0).all(1)).mean(), hs.mean()))
# null: all 132 negative-weight pairs at a = .1 on c4 gold58
null = []
for t in range(12):
    for h in range(12):
        if t == h: continue
        null.append((auc(pct(P[:, t]) - .1 * pct(P[:, h]), Y[:, t]) - auc(P[:, t], Y[:, t]), L[t], L[h]))
null.sort(key=lambda z: -z[0]); vals = np.array([z[0] for z in null])
rank = [i for i, z in enumerate(null) if z[1] == 'Lateral OA' and z[2] == 'PF OA'][0] + 1
print('   null of 132 negative pairs (a .1, c4 gold58): mean %+.4f sd %.4f frac>0 %.2f; Lateral OA<- -PF OA rank %d; top5 %s' % (
    vals.mean(), vals.std(), (vals > 0).mean(), rank, [(round(z[0], 4), z[1], z[2]) for z in null[:5]]))
# (c) report-regime OOF control (non-gold train), members with OOF: R7 and our S6 seeds s1044 / s1045
print('(c) report-label OOF control: Lateral OA AUC delta of pct(LatOA) - a pct(PFOA)')
oof = {'R7oof': r7[['p|' + c for c in L]].set_axis(L, axis=1)}
for sd in (1044, 1045):
    try: oof['s%d' % sd] = s6seed(sd)[0]
    except Exception as e: print('   s%d unavailable: %s' % (sd, e))
for k, (p, na) in srcs.items():
    lab = pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L].astype(float)
    for mk, O in oof.items():
        idx = [u for u in O.index if u not in allg and u in lab.index]
        x = lab.loc[idx, 'Lateral OA']; m = x.notna() & (~np.isclose(x, na) if na is not None else True)
        y = (x[m] >= .5).astype(int).values; X = O.loc[np.array(idx)[m.values]]
        b0 = auc(X['Lateral OA'].values, y)
        print('   %-7s %-6s n %d base %.4f  ' % (k, mk, len(y), b0) + '  '.join('a%.2f %+.4f' % (a, auc(pct(X['Lateral OA'].values) - a * pct(X['PF OA'].values), y) - b0) for a in A))
