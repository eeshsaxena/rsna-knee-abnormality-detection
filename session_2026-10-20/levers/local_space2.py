# local_space2.py (10-20 levers, Claude): gold58 INFO-ONLY paired bootstraps for the two OAI survivors (convex p2, SUB_HIGH) and
# a matched gold74 report-vs-expert conditional table. gold58 is NOT a selection ruler; this only asks whether gold58 contradicts.
import numpy as np, pandas as pd
exec(open('local_space.py').read().split('# ---------------- 1. gold58')[0])
from scipy.stats import rankdata
g = pd.read_csv(D + 'gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); Y58 = g58[OA].values.astype(int)
R6 = pd.read_csv(D + 'R6_gold58_serve.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R384 = np.load(D + 'gold58_r384_kernelserve_cpu.npy'); R224 = np.load(D + 'gold58_r224_kernelserve.npy')
S6 = pd.read_csv(D + 'n42gpu_gold58.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
RAP = .75 * pctc(R384) + .25 * pctc(R224)
out = np.zeros((58, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); b = w * RAP[:, j] + (1 - w) * pct(R6[:, j]); out[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
F = out[:, IOA]
NR = (.4 / .6) * pctc(R6[:, IOA]) + (.2 / .6) * pctc(S6[:, IOA])   # non-Raptor (high-gap) part of c4's OA output
def subhigh(F, a, sub):
    U = pctc(F); S = pctc(sub); o = np.empty_like(U)
    for t in range(3):
        oo = [k for k in range(3) if k != t]; o[:, t] = U[:, t] - a * (S[:, oo[0]] + S[:, oo[1]]) / 2
    return o
def m3(S, Y): return np.mean([auc(S[:, k], Y[:, k]) for k in range(3)])
V = {'lin a.2': eoah(F, .2), 'p2 a.2': eoah(F, .2, 2.), 'p1.5 a.2': eoah(F, .2, 1.5), 'lin a.25': eoah(F, .25), 'p2 a.15': eoah(F, .15, 2.),
     'pw lin a.2': eoah(F, .2, 1., True), 'pw p2 a.2': eoah(F, .2, 2., True), 'SUB_HIGH a.2': subhigh(F, .2, NR), 'SUB_HIGH a.25': subhigh(F, .25, NR)}
rng = np.random.default_rng(5)
B = []
while len(B) < 3000:
    ix = rng.integers(0, 58, 58)
    if all(0 < Y58[ix, k].sum() < 58 for k in range(3)): B.append(ix)
ref = {'sym': 'lin a.2', 'pw': 'pw lin a.2'}
print('gold58 c4 (INFO ONLY): OA-mean x 3/12 = macro; paired vs linear at the same mode, 3000 study bootstraps (pct fixed on full 58)')
for k, S in V.items():
    r = V[ref['pw' if k.startswith('pw') else 'sym']]
    d = np.array([m3(S[ix], Y58[ix]) - m3(r[ix], Y58[ix]) for ix in B]) * 3 / 12
    print('  %-14s vs %-10s: %+.5f macro (sd %.5f, P>0 %.2f)' % (k, ref['pw' if k.startswith('pw') else 'sym'], d.mean(), d.std(), (d > 0).mean()))
# matched gold74: P(Y_t = 1 | k other positives), report labels vs expert labels on the SAME knees
RL = pd.read_csv(LBL, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID'); gi = g.set_index('StudyInstanceUID')
idx = gi.index.intersection(RL.index); E = gi.loc[idx, OA].astype(float); R = RL.loc[idx, OA].astype(float)
ok = E.notna().all(axis=1) & R.notna().all(axis=1); E = E[ok].values.astype(int); R = (R[ok].values >= .5).astype(int)
for nm, Yx in (('expert', E), ('report', R)):
    k = np.stack([Yx[:, [o for o in range(3) if o != t]].sum(1) for t in range(3)], 1)
    print('  gold74 %-6s P(Y_t=1 | k others positive, same labeller): ' % nm + ' | '.join('k %d %.2f (%d/%d)' % (kk, Yx[k == kk].mean(), Yx[k == kk].sum(), (k == kk).sum()) for kk in range(3))
          + ' | 111 cell %d of %d' % (int((Yx.sum(1) == 3).sum()), len(Yx)))
