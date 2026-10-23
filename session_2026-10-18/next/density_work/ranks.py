# ranks.py (MODEL): public rank vs extra public gain over a25, today and projected; P(display .959) for a probe.
import numpy as np, sys
sys.argv = [sys.argv[0]]
from posterior import post
from math import erf, sqrt
u6, uH, u25 = post()
gap = 1 - u25                                     # band units (1 = .001)
def counts(d):                                    # d = days after 10-18 06:45 UTC; rates MEASURED (rates.py), linear (MODEL)
    g960 = 102 + 14.5 * d; g959 = 159 + 25.0 * d; v = 29 + 15.5 * d
    return dict(g961=68 + 9 * d, g960=g960, b959=g959 - g960, g959=g959, v=v)
def rank_after(g, d, gp):
    c = counts(d); s = 1 - gp + g                 # position measured from the .958 floor, in band units
    if s < 1: return 1 + c["g959"] + c["v"] * (1 - s) / gp
    if s < 2:
        k = 0.52; y = s - 1; frac = (np.exp(-k * y) - np.exp(-k)) / (1 - np.exp(-k)); return 1 + c["g960"] + c["b959"] * frac
    if s < 3:
        k = 0.69; y = s - 2; frac = (np.exp(-k * y) - np.exp(-k)) / (1 - np.exp(-k)); return 1 + c["g961"] + (c["g960"] - c["g961"]) * frac
    return np.nan
print("gap to .959 (x1e-4): p10 %.2f p50 %.2f p90 %.2f" % tuple(np.percentile(gap * 10, [10, 50, 90])))
print("\nextra public gain over a25 (x1e-4) -> median public rank [p10-p90 over the position posterior]; P(display >= .959)")
for when, d in [("today 10-18 06:45 UTC", 0.0), ("10-22 06:00 UTC (last read)", 4.47), ("10-22 23:59 UTC (deadline)", 4.72)]:
    print(" ", when, counts(d))
    for g in [0, 1, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 20]:
        r = np.array([rank_after(g / 10, d, x) for x in gap[:4000]])
        print(f"    +{g:>4}: r{np.nanmedian(r):.0f} [r{np.nanpercentile(r,90):.0f}-r{np.nanpercentile(r,10):.0f}]  P(>=.959) {np.mean(gap < g/10):.2f}  P(>=.960) {np.mean(gap + 1 < g/10):.2f}")
print("\nprobe read key: true public delta mu, paired public noise sd -> P(best-id move), P(display .959)")
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
rng = np.random.default_rng(3)
for sd in [3, 5]:
    for mu in [-2, 0, 1, 2, 3, 5, 7]:
        dpub = mu + sd * rng.standard_normal(gap.size)
        print(f"  sd {sd}e-4 mu {mu:+}e-4: P(move) {Phi(mu/sd):.2f}  P(.959) {np.mean(dpub > gap*10):.2f}")
