# e1_stages.py (10-19 skeptic lens, research only, CPU): where does E1's gold58 +.00048 come from, and is the gate spec
# self-consistent? Re-uses stack2/load.py and the same gold58 serve predictions as stack2/g58_nested.py.
# Variants (T = pct ** 2):
#   g58     : exactly stack2/g58_nested.py pow2 (Raptor mix RE-RANKED then squared at stage 3, squared at final)
#   exp1rr  : same code path with exponent 1 (Raptor mix re-ranked, else c4) -> is it bit-identical to c4?
#   st3only : pow2 at stage 3 only (re-ranked mix), final stays .8 pct + .2 pct
#   finonly : c4 stage 3, pow2 only at the final .8 / .2 S6fall stage
#   norr    : kernel-natural build: stage 3 = w * rr**2 + (1-w) * pct(R6)**2 with rr NOT re-ranked, final pow2
import sys, numpy as np
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import *
import pandas as pd
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')


def build(v, ix=None):
    ix = np.arange(n) if ix is None else ix; P = np.zeros((len(ix), 12))
    for j, c in enumerate(L):
        w = C4.get(c, .5)
        rr = .75 * pct(R384[ix, j]) + .25 * pct(R224[ix, j])
        if v == 'c4':
            b = w * rr + (1 - w) * pct(R6[ix, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[ix, j])
        elif v == 'g58':
            rr2 = .75 * pct(R384[ix, j]) ** 2 + .25 * pct(R224[ix, j]) ** 2
            b = w * pct(rr2) ** 2 + (1 - w) * pct(R6[ix, j]) ** 2; P[:, j] = .8 * pct(b) ** 2 + .2 * pct(S6[ix, j]) ** 2
        elif v == 'exp1rr':
            b = w * pct(rr) + (1 - w) * pct(R6[ix, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[ix, j])
        elif v == 'st3only':
            rr2 = .75 * pct(R384[ix, j]) ** 2 + .25 * pct(R224[ix, j]) ** 2
            b = w * pct(rr2) ** 2 + (1 - w) * pct(R6[ix, j]) ** 2; P[:, j] = .8 * pct(b) + .2 * pct(S6[ix, j])
        elif v == 'finonly':
            b = w * rr + (1 - w) * pct(R6[ix, j]); P[:, j] = .8 * pct(b) ** 2 + .2 * pct(S6[ix, j]) ** 2
        elif v == 'norr':
            rr2 = .75 * pct(R384[ix, j]) ** 2 + .25 * pct(R224[ix, j]) ** 2
            b = w * rr2 + (1 - w) * pct(R6[ix, j]) ** 2; P[:, j] = .8 * pct(b) ** 2 + .2 * pct(S6[ix, j]) ** 2
    return P


mac = lambda P, yy: np.mean([auc(P[:, j], yy[:, j]) for j in range(12)])
rng = np.random.default_rng(7); idxs = []
while len(idxs) < 2000:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
P0 = build('c4'); a0 = mac(P0, Y)
B0 = np.array([[auc(P0[ix, j], Y[ix, j]) for j in range(12)] for ix in idxs])
print('c4 gold58 macro %.5f' % a0)
for v in ['g58', 'exp1rr', 'st3only', 'finonly', 'norr']:
    P = build(v); a = mac(P, Y)
    per = np.array([auc(P[:, j], Y[:, j]) - auc(P0[:, j], Y[:, j]) for j in range(12)])
    Bb = np.array([[auc(P[ix, j], Y[ix, j]) for j in range(12)] for ix in idxs]); d = (Bb - B0).mean(1)
    exF = per[:11].sum() / 12   # Fracture is the last label; macro delta with Fracture held at c4
    same_order = all((pd.Series(P[:, j]).rank().values == pd.Series(P0[:, j]).rank().values).all() for j in range(12))
    print('%-8s d %+.5f sd %.5f P>0 %.3f | exFracture %+.5f | Fracture %+.4f | ranks identical to c4: %s | per-label %s' % (
        v, a - a0, d.std(), (d > 0).mean(), exF, per[11], same_order, np.round(per, 4).tolist()))
