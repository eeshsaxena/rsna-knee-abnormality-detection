"""proxy_ctx4.py : the 4 context curves (fleet into OOS bases) vs the macro binormal with r = rA (latent) and r = rA + offset (.062, LOBO-validated)."""
import json, numpy as np
from scipy.stats import norm
G = json.load(open("proxy_gold58_out.json")); F = json.load(open("proxy_fit_out.json"))
dv = lambda a: np.sqrt(2) * norm.ppf(a)
def B(P, a, r, w):
    v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * r
    return norm.cdf(((1 - w) * dv(P) + w * dv(a)) / np.sqrt(2 * v))
off = F["offset_mean"]; aM = G["arms"]["fleet_f0_mix50"]["macro"]
allE0, allE1 = [], []
for b, c in G["curves"].items():
    aB = c["base"]; ra = c["rA"]
    p0 = [B(aB, aM, ra, w) - aB for w in c["W"]]; p1 = [B(aB, aM, min(.999, ra + off), w) - aB for w in c["W"]]
    e0 = np.array(p0) - c["measured"]; e1 = np.array(p1) - c["measured"]; allE0 += list(e0); allE1 += list(e1)
    z = np.abs(e1) / np.array(c["boot_se"])
    print(f"{b:28s} rA {ra:.3f} -> r {ra+off:.3f} | meas " + " ".join(f"{x:+.4f}" for x in c["measured"]) +
          " | pred(rA+off) " + " ".join(f"{x:+.4f}" for x in p1) + f" | MAE rA {np.abs(e0).mean():.4f} rA+off {np.abs(e1).mean():.4f} | max |err|/bootSE {z.max():.2f}")
print(f"all 20 points: MAE rA {np.mean(np.abs(allE0)):.4f} (bias {np.mean(allE0):+.4f}) | rA+offset {np.mean(np.abs(allE1)):.4f} (bias {np.mean(allE1):+.4f})")
