# value_shift.py (10-19, think/picks): what is a private shift of A worth?  S(x) = P(silver | our private margin over copy = x) on the
# base field of field_mc.py (same parameters, MODEL/GUESS), then E[S(max(x_A + delta, x_B))] for the pair c4 + R6u and for c4 alone.
# Also the public-to-private translation of a +3e-4 public probe win (posterior shrinkage, MODEL).
import sys, json, numpy as np
from scipy.stats import norm
FSARG = sys.argv[1] if len(sys.argv) > 1 else "base"
sys.argv = [sys.argv[0], "base_report", FSARG, "2000"]
src = open("field_mc.py").read(); cut = src.index("idx = rng.choice")
exec(compile(src[:cut], "field_mc_head", "exec"))            # builds MU, SD, OTHER, P, silver, bronze (our own script)
grid = np.linspace(-2e-4, 1.2e-3, 71); S = []; Bz = []
for x in grid:
    n = 4000
    y = MU + SD * rng.standard_normal((n, nF)) + OTHER * (P["lin_sd"] * rng.standard_normal((n, 1)))
    r = 1 + (y > x).sum(1) + (P["n_copy"] if x < 0 else 0)
    S.append((r <= silver).mean()); Bz.append((r <= bronze).mean())
S = np.array(S); Bz = np.array(Bz)
Sx = lambda x: np.interp(x, grid, S); Bx = lambda x: np.interp(x, grid, Bz)
Z = np.load("post_base_report.npz"); ENT = list(Z["ent"]); PV = Z["priv"].astype(float); e = {k: i for i, k in enumerate(ENT)}
xa, xb = PV[:, e["c4"]], PV[:, e["R6u"]]
out = {"grid_x1e4": (grid * 1e4).round(2).tolist(), "S": S.round(4).tolist(), "B": Bz.round(4).tolist(), "shift": {}}
for d in [0, 0.5e-4, 1e-4, 2e-4, 3e-4, 5e-4]:
    out["shift"][f"{d*1e4:.1f}e-4"] = dict(A_alone_silver=float(Sx(xa + d).mean()), pair_silver=float(Sx(np.maximum(xa + d, xb)).mean()),
                                         A_alone_bronze=float(Bx(xa + d).mean()), pair_bronze=float(Bx(np.maximum(xa + d, xb)).mean()))
# public +3e-4 win of a one-line probe on c4 -> expected TRUE gain (posterior mean) for knob prior sd tau and public paired noise sd sp
tab = {}
for tau in [1e-4, 2e-4, 3e-4]:
    for sp in [3e-4, 5e-4, 8e-4]:
        lam = tau ** 2 / (tau ** 2 + sp ** 2); tab[f"tau{tau*1e4:.0f}_sp{sp*1e4:.0f}"] = round(lam * 3e-4 * 1e4, 2)
out["true_gain_of_pub3e-4_x1e4"] = tab
xs_ = grid[np.argmax(S >= 0.5)]; xb_ = grid[np.argmax(Bz >= 0.5)]
out["x_star_silver_x1e4"] = float(xs_ * 1e4); out["x_star_bronze_x1e4"] = float(xb_ * 1e4)
print("FIELD", FSARG, "silver threshold x* ~ %.1fe-4, bronze threshold ~ %.1fe-4 (private margin over the copy block)" % (xs_ * 1e4, xb_ * 1e4))
json.dump(out, open(f"value_shift_{FSARG}.json", "w"), indent=1)
print("S(x) silver curve (x in 1e-4):", [(round(g * 1e4, 1), round(s, 2)) for g, s in zip(grid[::4], S[::4])])
print("bronze curve:", [(round(g * 1e4, 1), round(s, 2)) for g, s in zip(grid[::4], Bz[::4])])
for k, v in out["shift"].items(): print(" shift", k, {a: round(b, 3) for a, b in v.items()})
print(" E[true gain | public +3e-4] x1e-4 (no selection correction):", tab)
