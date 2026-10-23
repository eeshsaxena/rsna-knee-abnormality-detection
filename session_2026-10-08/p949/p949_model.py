"""p949_model.py (2026-10-08): MODEL/GUESS estimate of public LB for blends with the public .949 CoAtNet (nartaa).
Same macro single-d binormal as SP/jump2/lbmath.py (validated on 11 OOF rank-blend pairs, error +.0000 to +.0009,
one-signed optimistic). AUC = Phi(d/sqrt2); rank blend with weights w: AUC_b = Phi(sum w_i d_i / sqrt(2 w'Rw)).
Displayed reads x are intervals [x-.0005, x+.0005).
P = parent .943, F = our 10-member fleet SOLO .944, P+F at W .50 = .946 (MEASURED reads), N = nartaa .949 (MEASURED read).
r_PF is inverted from the .946 read. r_PN is calibrated (GUESS) from the 10-08 LB cluster: 92 teams at .950, 45 of them
at .943 on 10-04, 81 of 92 last-submitted after the 10-07 19:34 UTC post, only 4 of 14 at .951 submitted after it.
Reading: the best parent+N blend shows .950, rarely .951 -> best(P,N) in [.9495,.9505) (GUESS, assumes they use the
.943 parent and a near-best W). r_FN is not measured: scanned.
"""
import itertools
import numpy as np
from scipy.stats import norm

def d(a): return np.sqrt(2) * norm.ppf(a)
def A(dd): return norm.cdf(dd / np.sqrt(2))
def iv(x): return (x - .0005, x + .0005 - 1e-9)

def blend2(a, b, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r
    return norm.cdf(((1 - W) * d(a) + W * d(b)) / np.sqrt(2 * v))

def best2(a, b, r):
    ws = np.linspace(0, 1, 401); x = blend2(a, b, r, ws); i = x.argmax(); return ws[i], x[i]

def blend3(s, R, w):
    w = np.asarray(w, float); dv = np.array([d(x) for x in s])
    return norm.cdf((w @ dv) / np.sqrt(2 * (w @ R @ w)))

def best3(s, R, step=.01):
    best = (None, -1)
    for wp in np.arange(0, 1 + 1e-9, step):
        for wf in np.arange(0, 1 - wp + 1e-9, step):
            w = (wp, wf, 1 - wp - wf); b = blend3(s, R, w)
            if b > best[1]: best = (w, b)
    return best

def psd(R): return np.linalg.eigvalsh(R).min() >= -1e-9

RG = np.linspace(.70, .9999, 6000)
out = []
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

# ---- 1. r_PF from the .946 read (P .943, F .944, W .50)
ok = np.zeros_like(RG, bool)
for P in np.linspace(*iv(.943), 9):
    for F in np.linspace(*iv(.944), 9):
        b = blend2(P, F, RG, .5); ok |= (b >= .9455) & (b < .9465)
rPF = (RG[ok].min(), RG[ok].max()); rPFc = RG[np.argmin(np.abs(blend2(.943, .944, RG, .5) - .946))]
pr(f'r_PF interval {rPF[0]:.3f}-{rPF[1]:.3f}, point {rPFc:.3f} (MODEL from MEASURED reads)')

# ---- 2. r_PN from the .950 cluster (GUESS calibration)
okN = np.zeros_like(RG, bool)
for P in np.linspace(*iv(.943), 7):
    for N in np.linspace(*iv(.949), 7):
        bb = np.array([best2(P, N, r)[1] for r in RG[::10]])
        okN[::10] |= (bb >= .9495) & (bb < .9505)
rPN = (RG[okN].min(), RG[okN].max()); rPNc = RG[::10][np.argmin(np.abs(np.array([best2(.943, .949, r)[1] for r in RG[::10]]) - .950))]
pr(f'r_PN interval (best P+N blend displays .950) {rPN[0]:.3f}-{rPN[1]:.3f}, point {rPNc:.3f} (GUESS calibration)')
for r in (rPN[0], rPNc, rPN[1]):
    w, b = best2(.943, .949, r); pr(f'  r_PN {r:.3f}: best W_N {w:.2f} -> {b:.4f}; W_N .50 -> {blend2(.943, .949, r, .5):.4f}')

# ---- 3. scenarios
pr('\n(a) P + F + N, best 3-way rank weights; (b) F + N best 2-way; (c) for reference P+N best, our current P+F .50')
pr('r_FN       r_PN   r_PF  | (a) best w(P,F,N) -> AUC | (a) fixed .25/.25/.50 | (b) best W_N -> AUC | (b) W_N .50')
rows = []
for rFN in (.86, .88, .90, .92, .94):
    for rPN_ in (rPN[0], rPNc, rPN[1]):
        for rPF_ in (rPF[0], rPFc, rPF[1]):
            R = np.array([[1, rPF_, rPN_], [rPF_, 1, rFN], [rPN_, rFN, 1]])
            if not psd(R): continue
            w, a = best3((.943, .944, .949), R)
            a_fix = blend3((.943, .944, .949), R, (.25, .25, .50))
            wb, bb = best2(.944, .949, rFN); b50 = blend2(.944, .949, rFN, .5)
            rows.append((rFN, rPN_, rPF_, w, a, a_fix, wb, bb, b50))
            if abs(rPF_ - rPFc) < 1e-9:
                pr(f'{rFN:.2f}  {rPN_:.3f} {rPF_:.3f} | ({w[0]:.2f},{w[1]:.2f},{w[2]:.2f}) -> {a:.4f} | {a_fix:.4f} | {wb:.2f} -> {bb:.4f} | {b50:.4f}')

A_ = np.array([r[4] for r in rows]); B_ = np.array([r[7] for r in rows]); Af = np.array([r[5] for r in rows])
pr(f'\nrange over all PSD combos: (a) best {A_.min():.4f}-{A_.max():.4f}, (a) fixed .25/.25/.50 {Af.min():.4f}-{Af.max():.4f}, (b) best {B_.min():.4f}-{B_.max():.4f}')
# central: r_FN = r_PF point (fleet as different from N as from parent), r_PN point
R = np.array([[1, rPFc, rPNc], [rPFc, 1, rPFc], [rPNc, rPFc, 1]])
w, a = best3((.943, .944, .949), R); wb, bb = best2(.944, .949, rPFc)
pr(f'CENTRAL (r_FN = r_PF {rPFc:.3f}, r_PN {rPNc:.3f}): (a) w {tuple(round(x,2) for x in w)} -> {a:.4f}; (b) W_N {wb:.2f} -> {bb:.4f}')
pr('Model is one-signed optimistic by ~+.0002-.0009 at high W for low-r pairs (verify_blendmath sec 2).')
open(__file__.replace('.py', '.out'), 'w').write('\n'.join(out) + '\n')
