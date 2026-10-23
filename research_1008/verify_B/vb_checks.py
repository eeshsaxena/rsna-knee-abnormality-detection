"""verify_B checks (2026-10-08, adversarial verify of B_blend_lb_model.md). Laptop, light numpy only."""
import numpy as np
from scipy.stats import norm
def d(a): return np.sqrt(2) * norm.ppf(a)
def bk(s, R, w):
    w = np.asarray(w, float); dv = np.array([d(x) for x in s]); return float(norm.cdf((w @ dv) / np.sqrt(2 * (w @ R @ w))))
def b2(a, b, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r
    return norm.cdf(((1 - W) * d(a) + W * d(b)) / np.sqrt(2 * v))

print('1. One-factor r_FN as a function of r_GN (r_PF .945, r_PN .945, r_PG .89): r_FN = r_PN*r_GN/r_PG * (r_PF/r_PN)')
for rGN in (.80, .838, .86, .88, .90):
    lN2 = .945 * rGN / .89
    print(f'   r_GN {rGN:.3f} -> l_N^2 {lN2:.3f} -> r_FN {lN2:.3f}  (l_P {.945/np.sqrt(lN2):.3f})')

print('\n2. r(N, comp-data-only 2.5D reader mix) from goodpjw N + readers@.30 = .948, by assumed reader strength')
RG = np.linspace(.6, .9999, 8000)
for rd in (.933, .935, .936, .938):
    ok = np.zeros_like(RG, bool)
    for a in np.linspace(.9485, .94949, 9):
        for b in np.linspace(rd - .0005, rd + .00049, 9):
            m = b2(a, b, RG, .30)
            ok |= (m >= .9475) & (m < .9485)
    print(f'   readers {rd}: r interval {RG[ok].min():.3f}-{RG[ok].max():.3f}')

print('\n3. K9N-a (.10/.30/.60) and F+N(.40/.60) raw model by r_FN (r_PN .945, r_PF .945, N .949, F .944, P .943)')
for rFN in (.88, .89, .90, .91, .92, .93, .94):
    R = np.array([[1, .945, .945], [.945, 1, rFN], [.945, rFN, 1]])
    a = bk((.943, .944, .949), R, (.10, .30, .60)); f = bk((.943, .944, .949), R, (0, .40, .60))
    b = bk((.943, .944, .949), R, (.30, .30, .40))
    print(f'   r_FN {rFN:.2f}: K9N-a {a:.4f} (vs cluster .9500: {a-.95:+.4f})  F+N {f:.4f} ({f-a:+.4f} vs a)  K9N-b {b:.4f} ({b-a:+.4f} vs a)')

print('\n3b. Same at r_PN low end .921 (P less redundant): P weight value')
for rFN in (.89, .92):
    R = np.array([[1, .945, .921], [.945, 1, rFN], [.921, rFN, 1]])
    a = bk((.943, .944, .949), R, (.10, .30, .60)); f = bk((.943, .944, .949), R, (0, .40, .60)); b = bk((.943, .944, .949), R, (.30, .30, .40))
    print(f'   r_PN .921 r_FN {rFN}: K9N-a {a:.4f}  F+N {f:.4f} ({f-a:+.4f})  K9N-b {b:.4f} ({b-a:+.4f})')

print('\n4. P(K9N-a true private > band X) integrating r_FN prior + model error + optimism (MODEL/GUESS)')
rng = np.random.default_rng(1)
M = 200000
for (mu, sd, opt) in [(.90, .025, 0.0), (.90, .025, .0005), (.915, .025, 0.0), (.915, .025, .0005), (.92, .02, .0005)]:
    rFN = np.clip(rng.normal(mu, sd, M), .84, .97)
    off = .0010 - .031 * (rFN - .90) - rng.uniform(0, 2 * opt, M) + rng.normal(0, .0006, M) + rng.normal(0, .0004, M)
    print(f'   r_FN ~N({mu},{sd}), optimism U(0,{2*opt:.4f}): E offset {off.mean():+.4f}, P(>0) {np.mean(off > 0):.2f}, '
          f'P(display >= .951 | band .9500) {np.mean(.9500 + off + rng.normal(0, .0005, M) >= .9505):.2f}')
