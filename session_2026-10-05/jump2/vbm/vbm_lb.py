"""vbm_lb.py (verify_blendmath 2026-10-05): laptop, light CPU. Re-runs the key LB inversions of blendmath.md under
(a) the note's display model (round: [x-.0005, x+.0005)) and (b) a truncation display model ([x, x+.001)),
plus spot checks of quoted numbers."""
import numpy as np
from scipy.stats import norm
def d(a): return np.sqrt(2) * norm.ppf(a)
def A(dd): return norm.cdf(dd / np.sqrt(2))
def blend(P, s, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r
    return norm.cdf(((1 - W) * d(P) + W * d(s)) / np.sqrt(2 * v))
RG = np.linspace(0.5, 0.9999, 5000)
def grid(lo, hi, n=11): return np.linspace(lo, hi - 1e-9, n)
def rint(Pr, sr, br, W):
    ok = np.zeros_like(RG, bool)
    for P in grid(*Pr):
        for s in grid(*sr):
            b = blend(P, s, RG, W); ok |= (b >= br[0]) & (b < br[1])
    return (RG[ok].min(), RG[ok].max()) if ok.any() else None
def joint(Pr, sr, b1, W1, b2, W2):
    ok = np.zeros_like(RG, bool)
    for P in grid(*Pr):
        for s in grid(*sr):
            x = blend(P, s, RG, W1); y = blend(P, s, RG, W2)
            ok |= (x >= b1[0]) & (x < b1[1]) & (y >= b2[0]) & (y < b2[1])
    return (RG[ok].min(), RG[ok].max()) if ok.any() else None
for name, iv in [('ROUND', lambda x: (x - .0005, x + .0005)), ('TRUNC', lambda x: (x, x + .001))]:
    print(f'===== display model {name}')
    k2 = rint(iv(.943), iv(.932), iv(.943), .30); print(f'K2 r_LB(F1 mix50): {k2[0]:.3f}-{k2[1]:.3f}')
    print('aastik joint (W .5 and W .6 both .945), parent s74-c .943:')
    for s in (.927, .930, .9325, .935, .9375, .940):
        j = joint(iv(.943), iv(s), iv(.945), .5, iv(.945), .6)
        print(f'  fleet solo {s:.4f}: ' + (f'r {j[0]:.3f}-{j[1]:.3f}' if j else 'none'))
print('\n===== spot checks (point values)')
for r in (.80, .85):
    print(f'solo .927 r {r}: W.5 {blend(.943, .927, r, .5):.4f} W.6 {blend(.943, .927, r, .6):.4f}')
print('TTR v4 .923 break-even r at W .25:', RG[np.argmax(blend(.943, .923, RG, .25) < .943)].round(3))
print('d ratios: F1m', round(d(.932) / d(.8811), 3), 'TTRv4', round(d(.923) / d(.8764), 3),
      '| TTR mix50 via F1m ratio:', round(A(d(.8836) * d(.932) / d(.8811)), 4), 'via TTRv4 ratio:', round(A(d(.8836) * d(.923) / d(.8764)), 4))
# k-member gains
def ens(s, r, rmm, k):
    v = (1 + (k - 1) * rmm) / k; return A(d(s) / np.sqrt(v)), r / np.sqrt(v)
for k in (2, 5, 10, 15): print(f'k{k} from (.932, r .922) rmm .955 ->', tuple(round(x, 4) for x in ens(.932, .922, .955, k)))
# K4: A1 fleet at various solo / r; blend at W .2 / .3 / .5
print('\nK4 sensitivity: parent .943 + fleet (solo s, r) at W .15/.20/.30/.50')
for s, r in [(.932, .939), (.934, .939), (.936, .939), (.938, .939), (.936, .92), (.936, .95), (.930, .93)]:
    print(f'  s {s:.3f} r {r:.3f}: ' + ' '.join(f'W{w:.2f} {blend(.943, s, r, w):.4f}' for w in (.15, .2, .3, .5)))
# g74 calibration fit
sp = np.array([.939, .913, .912, .904, .882, .879]); re = np.array([.948, .934, .927, .917, .920, .900])
b, a = np.polyfit(sp - .905, re, 1); res = re - (a + b * (sp - .905))
print(f'\ng74 map: r = {a:.3f} + {b:.3f}*(sp-.905); corr {np.corrcoef(sp, re)[0,1]:.3f}; resid sd (ddof=2) {np.sqrt((res**2).sum()/4):.4f}')
for x in (.95, .96, .97, .98): print(f'  sp {x}: r {a + b*(x-.905):.3f}')
# leave-one-out stability of slope
sl = [np.polyfit(np.delete(sp, i), np.delete(re, i), 1)[0] for i in range(6)]
print('  LOO slopes', np.round(sl, 3))
# what 'r_mm' does same-seed-different-fold give vs raw/ema: at sp .98 the raw/ema pair had sp(OOF) .995 -> r .995
print('  raw/EMA anchor (OOF sp .984-.995, r_eff .985-.995) vs map at sp .985/.995 ->', round(a + b*(.985-.905), 3), round(a + b*(.995-.905), 3))
