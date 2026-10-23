#!/usr/bin/env python3
"""k9n_weights.py (2026-10-08, Claude; laptop, grids only). MODEL/GUESS: which 3-leg rank weights (P parent, F our
10-member K9 fleet SOLO, N the public .949 CoAtNet by nartaa) should K9N-a / K9N-b ship?

Same macro single-d binormal as SP/jump2/lbmath.py and SP/p949/p949_model.py (validated on 11 OOF rank-blend pairs,
error +.0000 to +.0009, one-signed optimistic): AUC = Phi(d/sqrt2); rank blend with weights w:
AUC_b = Phi(sum w_i d_i / sqrt(2 w'Rw)). Displayed reads x are intervals [x-.0005, x+.0005).
Inputs (MEASURED reads): P .943, F .944 (K7i SOLO; K9 SOLO not read yet), P+F at W .50 = .946, N .949.
r_PF inverted from the .946 read; r_PN calibrated (GUESS) on the 10-08 LB cluster (92 teams at .950 = best P+N blend);
r_FN NOT measured: scanned .86-.94 (N and our fleet are different archs, but mix50 carries a Raptor teacher).
Extra (GUESS): N's .949 is the best of the author's public reads (.939/.940/.942/.945/.946/.949), so its true strength on
a fresh draw may sit below .949 (winner's curse); scanned N_true in {.9475, .949}.
Output: per candidate weight triple, the modelled AUC at the central point, the min over the scenario grid, and the max
regret vs the per-scenario best triple. Optimism correction: subtract ~.0005 from every modelled value before reading it
as a displayed public LB (verify_blendmath sec 2).
  python k9n_weights.py   -> k9n_weights.out
"""
import itertools
import numpy as np
from scipy.stats import norm


def d(a): return np.sqrt(2) * norm.ppf(a)
def iv(x): return (x - .0005, x + .0005 - 1e-9)


def blend2(a, b, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r
    return norm.cdf(((1 - W) * d(a) + W * d(b)) / np.sqrt(2 * v))


def best2(a, b, r):
    ws = np.linspace(0, 1, 401); x = blend2(a, b, r, ws); i = x.argmax(); return ws[i], x[i]


def blend3(s, R, w):
    w = np.asarray(w, float); dv = np.array([d(x) for x in s])
    return norm.cdf((w @ dv) / np.sqrt(2 * (w @ R @ w)))


def psd(R): return np.linalg.eigvalsh(R).min() >= -1e-9


RG = np.linspace(.70, .9999, 6000)
out = []


def pr(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)


# ---- r_PF from the .946 read (P .943, F .944, W .50): MODEL from MEASURED reads
ok = np.zeros_like(RG, bool)
for P in np.linspace(*iv(.943), 9):
    for F in np.linspace(*iv(.944), 9):
        b = blend2(P, F, RG, .5); ok |= (b >= .9455) & (b < .9465)
rPF = (RG[ok].min(), RG[ok].max()); rPFc = RG[np.argmin(np.abs(blend2(.943, .944, RG, .5) - .946))]
# ---- r_PN from the .950 cluster (GUESS calibration, as p949_model.py)
okN = np.zeros_like(RG, bool)
for P in np.linspace(*iv(.943), 7):
    for N in np.linspace(*iv(.949), 7):
        bb = np.array([best2(P, N, r)[1] for r in RG[::10]])
        okN[::10] |= (bb >= .9495) & (bb < .9505)
rPN = (RG[okN].min(), RG[okN].max())
rPNc = RG[::10][np.argmin(np.abs(np.array([best2(.943, .949, r)[1] for r in RG[::10]]) - .950))]
pr(f'r_PF {rPF[0]:.3f}-{rPF[1]:.3f} point {rPFc:.3f} (MODEL); r_PN {rPN[0]:.3f}-{rPN[1]:.3f} point {rPNc:.3f} (GUESS)')

CANDS = {
    'user-a  .30/.30/.40': (.30, .30, .40),
    'user-b  .40/.30/.30': (.40, .30, .30),
    '.25/.25/.50': (.25, .25, .50),
    '.20/.25/.55': (.20, .25, .55),
    '.20/.20/.60': (.20, .20, .60),
    '.15/.25/.60': (.15, .25, .60),
    '.25/.15/.60': (.25, .15, .60),
    '.15/.20/.65': (.15, .20, .65),
    '.10/.20/.70': (.10, .20, .70),
    '.35/.35/.30': (.35, .35, .30),
    '.30/.25/.45': (.30, .25, .45),
    'K9N-a .10/.30/.60': (.10, .30, .60),
    '.20/.30/.50': (.20, .30, .50),
}
scen = []
for NT in (.9475, .949):
    for rFN in (.86, .88, .90, .92, .94):
        for rPN_ in (rPN[0], rPNc, rPN[1]):
            for rPF_ in (rPF[0], rPFc, rPF[1]):
                R = np.array([[1, rPF_, rPN_], [rPF_, 1, rFN], [rPN_, rFN, 1]])
                if psd(R):
                    scen.append((NT, rFN, rPN_, rPF_, R))
pr(f'{len(scen)} PSD scenarios (N_true x r_FN x r_PN x r_PF)')
# per-scenario best over a .01 simplex grid
grid = [(wp, wf, round(1 - wp - wf, 2)) for wp in np.arange(0, 1.0001, .01) for wf in np.arange(0, 1.0001 - wp, .01)]
best = []
for NT, rFN, rPN_, rPF_, R in scen:
    s = (.943, .944, NT)
    vals = [blend3(s, R, w) for w in grid]
    i = int(np.argmax(vals)); best.append((grid[i], vals[i]))
res = {}
for name, w in CANDS.items():
    v = np.array([blend3((.943, .944, NT), R, w) for NT, rFN, rPN_, rPF_, R in scen])
    reg = np.array([b[1] for b in best]) - v
    res[name] = (v, reg)
# central scenarios: N_true .949 / .9475, r_PN point, r_PF point, r_FN .90 and .92
def cen(NT, rFN):
    R = np.array([[1, rPFc, rPNc], [rPFc, 1, rFN], [rPNc, rFN, 1]]); return (.943, .944, NT), R


pr('\nmodelled AUC (subtract ~.0005 for a displayed public read).  C90/C92 = central r_PN/r_PF with r_FN .90/.92, N .949;'
   ' W92 = C92 with N_true .9475')
pr(f'{"weights P/F/N":22s} {"C90":>7s} {"C92":>7s} {"W92":>7s} {"min":>7s} {"max":>7s} {"maxregret":>9s} {"meanregret":>10s}')
for name, w in CANDS.items():
    v, reg = res[name]
    c90 = blend3(*cen(.949, .90), w); c92 = blend3(*cen(.949, .92), w); w92 = blend3(*cen(.9475, .92), w)
    pr(f'{name:22s} {c90:7.4f} {c92:7.4f} {w92:7.4f} {v.min():7.4f} {v.max():7.4f} {reg.max():9.4f} {reg.mean():10.4f}')
# minimax-regret over the full .01 grid
mm = None
for w in grid:
    v = np.array([blend3((.943, .944, NT), R, w) for NT, rFN, rPN_, rPF_, R in scen])
    r = (np.array([b[1] for b in best]) - v).max()
    if mm is None or r < mm[1]:
        mm = (w, r, v.min(), v.mean())
pr(f'\nminimax-regret triple on the .01 grid: P/F/N {mm[0]} max regret {mm[1]:.4f}, min AUC {mm[2]:.4f}, mean {mm[3]:.4f}')
bw = np.array([b[0] for b in best])
pr(f'per-scenario best weights: P {bw[:, 0].min():.2f}-{bw[:, 0].max():.2f}, F {bw[:, 1].min():.2f}-{bw[:, 1].max():.2f}, '
   f'N {bw[:, 2].min():.2f}-{bw[:, 2].max():.2f} (median N {np.median(bw[:, 2]):.2f})')
pr('Model is one-signed optimistic by ~+.0002-.0009 (verify_blendmath sec 2); rank ties at the .001 display make any '
   'difference < .0005 invisible on the public LB.')
open(__file__.replace('.py', '.out'), 'w').write('\n'.join(out) + '\n')
