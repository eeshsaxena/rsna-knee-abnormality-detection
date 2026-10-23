# g58_nested.py: c4's NESTED structure kept (Raptor r384/r224 -> stage 3 with R6 -> final with S6fall), only the per-stage combiner
# transform changes (T applied to percentiles, or z-scored logits). Gold58 (58 studies, Kaggle annotation), paired bootstrap vs c4,
# plus split-half sign agreement. Also the same operators on the OOF-ruler member family on gold58 (R7 gold rows + s43 full model).
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from load import *
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv'); S43 = rd('s43_gold58.csv')
R7 = r7().loc[ids, ['p|' + c for c in L]].values
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
zst = lambda v: (v - v.mean()) / (v.std() + 1e-12)
TF = {'rank': lambda v: pct(v), 'pow2': lambda v: pct(v) ** 2, 'pow3': lambda v: pct(v) ** 3,
      'invgeo': lambda v: -np.log(1 - pct(v) + .5 / len(v)), 'nscore': lambda v: norm.ppf((rankdata(v) - .5) / len(v)),
      'zlogit': lambda v: zst(lg(v))}


def c4(T, ix=None):
    ix = np.arange(n) if ix is None else ix; P = np.zeros((len(ix), 12))
    for j, c in enumerate(L):
        w = C4.get(c, .5)
        T2 = zst if T is TF['zlogit'] else T          # intermediate stages are not probabilities: z-score only
        rr = .75 * T(R384[ix, j]) + .25 * T(R224[ix, j])
        b = w * T2(rr) + (1 - w) * T(R6[ix, j]) if T is not TF['rank'] else w * rr + (1 - w) * T(R6[ix, j])
        P[:, j] = .8 * T2(b) + .2 * T(S6[ix, j])
    return P


mac = lambda P, yy: np.mean([auc(P[:, j], yy[:, j]) for j in range(12)])
rng = np.random.default_rng(7); idxs = []
while len(idxs) < 2000:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
P0 = c4(TF['rank']); a0 = mac(P0, Y)
B0 = np.array([[auc(P0[ix, j], Y[ix, j]) for j in range(12)] for ix in idxs])
print('c4 (rank, as shipped) gold58 macro %.5f' % a0)
for nm in ['pow2', 'pow3', 'invgeo', 'nscore', 'zlogit']:
    P = c4(TF[nm]); a = mac(P, Y)
    Bb = np.array([[auc(P[ix, j], Y[ix, j]) for j in range(12)] for ix in idxs]); d = (Bb - B0).mean(1)
    per = [auc(P[:, j], Y[:, j]) - auc(P0[:, j], Y[:, j]) for j in range(12)]
    # split-half: sign agreement of the macro delta between disjoint halves (ranks recomputed within each half)
    agree = []; k = 0
    while k < 400:
        perm = rng.permutation(n); A, Bh = perm[:29], perm[29:]
        if any(Y[A, j].sum() in (0, 29) or Y[Bh, j].sum() in (0, 29) for j in range(12)): continue
        k += 1
        dA = mac(c4(TF[nm], A), Y[A]) - mac(c4(TF['rank'], A), Y[A]); dB = mac(c4(TF[nm], Bh), Y[Bh]) - mac(c4(TF['rank'], Bh), Y[Bh])
        agree.append((dA, dB))
    agree = np.array(agree)
    exF = np.mean([p for jj, p in enumerate(per) if L[jj] != 'Fracture']) * 11 / 12
    print('   (macro delta with Fracture held at c4: %+.5f)' % exF)
    print('nested %-7s d %+.5f  boot sd %.5f P>0 %.3f | half-A>0 %.2f, both halves same sign %.2f | labels +%d/-%d %s' % (
        nm, a - a0, d.std(), (d > 0).mean(), (agree[:, 0] > 0).mean(), (np.sign(agree[:, 0]) == np.sign(agree[:, 1])).mean(),
        sum(p > 0 for p in per), sum(p < 0 for p in per), sorted([(L[j], round(per[j], 4)) for j in range(12)], key=lambda t: -abs(t[1]))[:3]))
print('--- OOF-ruler family on gold58 (R7 gold rows + s43 full model), .5/.5 and .8/.2 ---')
for w in (.5, .8):
    base = np.column_stack([w * pct(R7[:, j]) + (1 - w) * pct(S43[:, j]) for j in range(12)]); ab = mac(base, Y)
    Bb0 = np.array([[auc(base[ix, j], Y[ix, j]) for j in range(12)] for ix in idxs])
    for nm in ['pow2', 'invgeo', 'nscore', 'zlogit']:
        T = TF[nm]; P = np.column_stack([w * T(R7[:, j]) + (1 - w) * T(S43[:, j]) for j in range(12)])
        Bb = np.array([[auc(P[ix, j], Y[ix, j]) for j in range(12)] for ix in idxs]); d = (Bb - Bb0).mean(1)
        print('  R7/s43 %.1f/%.1f %-7s d %+.5f sd %.5f P>0 %.3f (base %.4f)' % (w, 1 - w, nm, mac(P, Y) - ab, d.std(), (d > 0).mean(), ab))
