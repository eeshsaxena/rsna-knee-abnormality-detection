# slots_mc.py (10-19, think/picks): does spending the remaining slots on probes help or hurt PRIVATE, under the A15 rule
# (A = public argmax at the last read, B stays R6u)?  MODEL: each probe X = c4 + one change with true effect t ~ N(mu, tau^2),
# public paired noise N(0, sp^2) MINUS c4's own public luck L (calibrated so P(single probe beats c4) = pwin, MEASURED 0/9 recent,
# 4/19 since 10-16), private paired noise N(0, (0.65 sp)^2). Private value through S(x) of value_shift.py (base field).
import json, numpy as np
from scipy.stats import norm
V = json.load(open("value_shift_base.json")); grid = np.array(V["grid_x1e4"]) * 1e-4; S = np.array(V["S"]); Bz = np.array(V["B"])
Sx = lambda x: np.interp(x, grid, S, left=0.0, right=1.0); Bx = lambda x: np.interp(x, grid, Bz, left=0.0, right=1.0)
Z = np.load("post_base_report.npz"); ENT = list(Z["ent"]); PV = Z["priv"].astype(float); e = {k: i for i, k in enumerate(ENT)}
rng = np.random.default_rng(5)
n = 100_000; idx = rng.choice(PV.shape[0], n, replace=False); xa = PV[idx, e["c4"]]; xb = PV[idx, e["R6u"]]
base_s = Sx(np.maximum(xa, xb)).mean(); base_b = Bx(np.maximum(xa, xb)).mean()
out = {"base_pair_silver": float(base_s), "base_pair_bronze": float(base_b), "rows": []}
for name, mu, tau, sp, pwin in [("weight tweaks near a tuned optimum", -1e-4, 2e-4, 5e-4, 0.20), ("neutral new directions", 0.0, 2e-4, 5e-4, 0.25),
                                ("real diversity leg (new model)", 2e-4, 3e-4, 6e-4, 0.35), ("single-label rare-class tweak", 0.0, 2e-4, 9e-4, 0.25)]:
    sd_d = np.sqrt(tau ** 2 + sp ** 2); L = sd_d * norm.ppf(1 - pwin) + mu     # public delta = t + eps - L; P(delta > 0) = pwin
    for K in [1, 5, 10]:
        t = mu + tau * rng.standard_normal((n, K)); d = t + sp * rng.standard_normal((n, K)) - L
        best = d.argmax(1); win = d[np.arange(n), best] > 0
        tw = t[np.arange(n), best]; pn = 0.65 * sp * rng.standard_normal(n)
        xA = np.where(win, xa + tw + pn, xa)
        s = Sx(np.maximum(xA, xb)).mean(); b = Bx(np.maximum(xA, xb)).mean()
        out["rows"].append(dict(case=name, K=K, mu=mu, tau=tau, sp=sp, pwin=pwin, P_A_switch=float(win.mean()),
                                E_true_gain_if_switch=float(tw[win].mean()) if win.any() else 0.0,
                                d_pair_silver=float(s - base_s), d_pair_bronze=float(b - base_b)))
json.dump(out, open("slots_mc.json", "w"), indent=1)
print("base pair silver %.3f bronze %.3f" % (base_s, base_b))
for r in out["rows"]:
    print(f"  {r['case']:38s} K {r['K']:2d}: P(A switches) {r['P_A_switch']:.2f}  E[true gain | switch] {r['E_true_gain_if_switch']*1e4:+.2f}e-4  "
          f"d silver {r['d_pair_silver']*100:+.1f} pts  d bronze+ {r['d_pair_bronze']*100:+.1f} pts")
