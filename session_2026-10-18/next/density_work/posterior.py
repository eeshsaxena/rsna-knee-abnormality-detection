# posterior.py (MODEL): position of R6 / HW / a25 inside the .958 band from the frozen "natural" reference teams.
# Reference set (MEASURED, natlist.py): teams in the .958 band with best sid < 57190000 (10-14 17:13 UTC), frozen through
# 10-16 06:22..15:17, not in a tie run: 1 below R6, 5 between R6 and HW, 5 above HW. a25 passed exactly 10 frozen teams
# above HW (33 -> 23 above us in band, nothing else changed between 19:35 and 21:13 UTC 10-17).
# Natural density inside the band: f(x) ~ exp(-k x), x in [0,1) from the band floor, k ~ U(kmin, kmax) (tail thins upward).
import numpy as np, sys
rng = np.random.default_rng(0)
def post(n0=1, n1=5, n2=5, kmin=0.2, kmax=1.0, M=400000, vfrac=23/33, vjit=0.1):
    k = rng.uniform(kmin, kmax, M)
    u6 = rng.uniform(0, 1, M); uH = rng.uniform(0, 1, M)
    sw = u6 > uH; u6[sw], uH[sw] = uH[sw], u6[sw]
    F = lambda x: (1 - np.exp(-k * x)) / (1 - np.exp(-k))
    a, b = F(u6), F(uH)
    w = a**n0 * (b - a)**n1 * (1 - b)**n2
    w /= w.sum()
    # a25: 1 - u25 = vfrac * (1 - uH) * jitter (local density in the top slice not exactly uniform)
    u25 = 1 - vfrac * (1 - uH) * rng.uniform(1 - vjit, 1 + vjit, M)
    idx = rng.choice(M, 60000, p=w)
    return u6[idx], uH[idx], u25[idx]
def q(x): return f"mean {x.mean():.3f} p10 {np.percentile(x,10):.3f} p50 {np.median(x):.3f} p90 {np.percentile(x,90):.3f}"
for name, kw in [("base k 0.2-1.0", {}), ("flat density (k 0-0.05)", dict(kmin=0.0001, kmax=0.05)), ("steep k 0.8-1.4", dict(kmin=0.8, kmax=1.4)),
                 ("incl. tie-run member below R6 (n0=2)", dict(n0=2)), ("count 57164725 as copy-level too (n0=0)", dict(n0=0))]:
    u6, uH, u25 = post(**kw)
    print(f"== {name}\n  u_R6  {q(u6)}\n  u_HW  {q(uH)}\n  u_a25 {q(u25)}\n  gap to .959 (1e-4 units) {q((1-u25)*10)}\n  HW - R6 (1e-4) {q((uH-u6)*10)}")
