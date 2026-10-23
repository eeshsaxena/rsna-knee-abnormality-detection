"""lbmath.py (jump2, 2026-10-05): binormal inversions of our MEASURED public LB reads. Light CPU (grids only).
AUC = Phi(d/sqrt2); global rank blend at W: d_b=(1-W)d1+W d2, v=(1-W)^2+W^2+2W(1-W)r, AUC_b=Phi(d_b/sqrt(2v)).
Every displayed read x is the interval [x-.0005, x+.0005).
"""
import numpy as np
from scipy.stats import norm

def d(a): return np.sqrt(2) * norm.ppf(a)
def A(dd): return norm.cdf(dd / np.sqrt(2))
def blend(P, s, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r
    return norm.cdf(((1 - W) * d(P) + W * d(s)) / np.sqrt(2 * v))
def iv(x): return (x - .0005, x + .0005)
def grid(lo, hi, n=11): return np.linspace(lo, hi - 1e-9, n)
RG = np.linspace(0.5, 0.9999, 5000)

def r_interval(Pr, sr, br, W, n=11):
    """all r in RG consistent with parent in Pr, solo in sr, blend read in br at weight W."""
    ok = np.zeros_like(RG, bool)
    for P in grid(*Pr, n):
        for s in grid(*sr, n):
            b = blend(P, s, RG, W)
            ok |= (b >= br[0]) & (b < br[1])
    return (RG[ok].min(), RG[ok].max()) if ok.any() else None

def r_point(P, s, b, W):
    i = np.argmin(np.abs(blend(P, s, RG, W) - b)); return RG[i]

def best(P, s, r):
    ws = np.linspace(0, 1, 201); b = blend(P, s, r, ws); i = b.argmax(); return ws[i], b[i]

out = []
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

PAR = iv(.943)
pr('== (1) K2: F1 mix50 solo .932, parent + F1 mix50 at W .30 = .943')
k2 = r_interval(PAR, iv(.932), iv(.943), .30)
pr(f'r_LB(F1 mix50) interval {k2[0]:.3f}-{k2[1]:.3f}; point (P .943, s .932, b .943) {r_point(.943, .932, .943, .30):.3f}')
for r in [k2[0], (k2[0] + k2[1]) / 2, k2[1]]:
    w, b = best(.943, .932, r); pr(f'  at r {r:.3f}: W* {w:.2f}, best blend {b:.4f} (lift {b - .943:+.4f}); W .30 -> {blend(.943, .932, r, .30):.4f}; W .50 -> {blend(.943, .932, r, .5):.4f}')
# solo-alone check of how strong a member at that r must be
pr('  member solo needed for +.001/+.002/+.003 true lift at the K2 r interval (W*):')
for r in [k2[0], (k2[0] + k2[1]) / 2, k2[1]]:
    row = []
    for tgt in (.001, .002, .003):
        ss = np.linspace(.90, .975, 1501); bb = np.array([best(.943, s, r)[1] for s in ss]); j = np.argmax(bb >= .943 + tgt)
        row.append(f'{ss[j]:.4f}' if bb[j] >= .943 + tgt else 'n/a')
    pr(f'    r {r:.3f}: ' + ' / '.join(row))

pr('\n== (2) K3: TTR v4 solo .923; parent + mean(F1 mix50, TTR v4) at W .25 = .943')
def ens2(s1, s2, r1, r2, rmm):
    v = (1 + rmm) / 2; de = (d(s1) + d(s2)) / 2 / np.sqrt(v); re = (r1 + r2) / 2 / np.sqrt(v); return A(de), re
def psd(r1, r2, rmm):
    return 1 - r1 ** 2 - r2 ** 2 - rmm ** 2 + 2 * r1 * r2 * rmm >= 0
for rmm in (0.88, 0.90, 0.92, 0.94):
    lo, hi = 9, -9
    for P in grid(*PAR, 7):
        for s1 in grid(*iv(.932), 5):
            for s2 in grid(*iv(.923), 5):
                for r1 in np.linspace(k2[0], k2[1], 9):
                    r2 = np.linspace(0.5, 0.999, 500)
                    se, re = ens2(s1, s2, r1, r2, rmm)
                    good = psd(r1, r2, rmm) & (re < 1)
                    b = blend(P, se, np.minimum(re, .9999), .25)
                    hit = good & (b >= PAR[0]) & (b < PAR[1])
                    if hit.any():
                        lo, hi = min(lo, r2[hit].min()), max(hi, r2[hit].max())
    se, re = ens2(.932, .923, (k2[0] + k2[1]) / 2, (k2[0] + k2[1]) / 2, rmm)
    pr(f'r_mm {rmm:.2f}: 2-member solo (central) {se:.4f}; r_LB(TTR v4) consistent with K3 STRICT: {lo:.3f}-{hi:.3f}')
pr('K3 read alone (fleet = 2-member mean, solo e): r_e interval by assumed e')
for e in (.928, .930, .932, .934):
    ri = r_interval(PAR, iv(e), iv(.943), .25)
    pr(f'  e {e:.3f}: r_e {ri[0]:.3f}-{ri[1]:.3f}' if ri else f'  e {e:.3f}: none')
pr('TTR v4 solo alone: binormal break-even r at W .25 for a .923 member (blend >= parent): '
   + f'{RG[np.argmax(blend(.943, .923, RG, .25) < .943)]:.3f}')

pr('\n== (3) aastik: s74-c (no fleet) .943; s75 W .5 = .945 and W .6 = .945')
for s in np.arange(.920, .9461, .0025):
    a = r_interval(PAR, iv(s), iv(.945), .5); b = r_interval(PAR, iv(s), iv(.945), .6)
    both = None
    if a and b:
        # joint consistency: same P and same s for both reads
        ok = np.zeros_like(RG, bool)
        for P in grid(*PAR, 11):
            for ss in grid(*iv(s), 11):
                b5 = blend(P, ss, RG, .5); b6 = blend(P, ss, RG, .6)
                ok |= (b5 >= .9445) & (b5 < .9455) & (b6 >= .9445) & (b6 < .9455)
        both = (RG[ok].min(), RG[ok].max()) if ok.any() else None
    pr(f'  fleet solo {s:.4f}: W.5 r {a[0]:.3f}-{a[1]:.3f} | W.6 r {b[0]:.3f}-{b[1]:.3f} | BOTH {"%.3f-%.3f" % both if both else "none"}' if a and b else f'  fleet solo {s:.4f}: W.5 {a} W.6 {b}')
pr('  joint (s, r) feasible region summary:')
feas = []
for s in np.arange(.915, .950, .0005):
    for P in grid(*PAR, 5):
        b5 = blend(P, s, RG, .5); b6 = blend(P, s, RG, .6)
        ok = (b5 >= .9445) & (b5 < .9455) & (b6 >= .9445) & (b6 < .9455)
        if ok.any(): feas.append((s, RG[ok].min(), RG[ok].max()))
if feas:
    S = np.array(feas)
    pr(f'    solo range {S[:, 0].min():.4f}-{S[:, 0].max():.4f}; r range {S[:, 1].min():.3f}-{S[:, 2].max():.3f}')
    for s in np.unique(np.round(S[:, 0], 4))[::4]:
        m = np.isclose(S[:, 0], s); pr(f'    s {s:.4f}: r {S[m, 1].min():.3f}-{S[m, 2].max():.3f}')
pr('  for comparison, our F1 mix50 at K2 r: predicted W .5 / .6 reads:')
for r in [k2[0], k2[1]]:
    pr(f'    r {r:.3f}: W.5 {blend(.943, .932, r, .5):.4f}  W.6 {blend(.943, .932, r, .6):.4f}')

open(__file__.replace('lbmath.py', 'lbmath.out'), 'w').write('\n'.join(out) + '\n')
