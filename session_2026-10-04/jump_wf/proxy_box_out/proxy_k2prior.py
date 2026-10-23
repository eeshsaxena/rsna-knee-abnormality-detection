"""proxy_k2prior.py : prior predictive of the two pending K2 reads (MODEL + GUESS priors, stated below)."""
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(1); n = 200000
dv = lambda a: np.sqrt(2) * norm.ppf(a)
def B(P, a, r, w):
    v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * r
    return norm.cdf(((1 - w) * dv(P) + w * dv(a)) / np.sqrt(2 * v))
# GUESS priors: solo = fleet gold58 .9144 + Dread-like gold58->LB offset (+.007..+.011) with gold58 bootstrap SE .014 shrunk -> N(.922, .008)
s = np.clip(rng.normal(0.922, 0.008, n), 0.89, 0.95)
# r_LB: half mass HIGH (crop/Tian ratio transfer, U[.93,.98]), half LOW-MID (gold58 no-transfer, U[.88,.93])
r = np.where(rng.random(n) < 0.5, rng.uniform(0.93, 0.98, n), rng.uniform(0.88, 0.93, n))
P = rng.uniform(0.9425, 0.9435, n)
b = B(P, s, r, 0.30)
sd = np.round(s, 3); bd = np.round(b, 3)
print("SOLO displayed: " + " ".join(f"{v:.3f}:{(sd == v).mean():.2f}" for v in np.unique(sd) if (sd == v).mean() > 0.01))
print("BLEND W.30 displayed: " + " ".join(f"{v:.3f}:{(bd == v).mean():.2f}" for v in np.unique(bd) if (bd == v).mean() > 0.01))
print(f"P(BLEND >= .943) {(bd >= 0.943).mean():.3f} | P(BLEND >= .942) {(bd >= 0.942).mean():.3f} | P(BLEND <= .939) {(bd <= 0.939).mean():.3f} | median {np.median(b):.4f}")
