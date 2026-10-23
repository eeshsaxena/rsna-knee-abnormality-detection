# The parent's unknown position p0 inside its 0.943 display bucket shifts EVERY read. The v9 swap (true delta ~0)
# acts as a p0 control. Condition C25's private delta on (C25 read, v9 read). Also P(read | prior set).
import numpy as np, glob
from risk_sim import pmed, disp, SP
rng = np.random.default_rng(5)
for PRI in ["opt", "cen", "pes"]:
    R = [r for f in sorted(glob.glob(SP + "/endgame3/risk_draws_%s*.npy" % PRI)) for r in np.load(f, allow_pickle=True)]
    n = len(R)
    c25 = np.array([disp(r["p0"], r["C25"][0]) for r in R]); prv = np.array([r["C25"][1] for r in R])
    v9 = np.round(0.943 + np.array([r["p0"] for r in R]) + rng.normal(0, 0.0004, n), 3)   # v9: delta ~N(0,.0004) public
    pr = {d: np.mean(c25 == d) for d in [0.941, 0.942, 0.943, 0.944]}; pr[0.945] = np.mean(c25 >= 0.945)
    print(PRI, "P(C25 read):", " ".join("%.3f:%.2f" % kv for kv in pr.items()))
    for d in [0.943, 0.944, 0.945]:
        out = []
        for v in [0.942, 0.943, 0.944]:
            m = ((c25 == d) if d < 0.945 else (c25 >= 0.945)) & (v9 == v)
            if m.sum() >= 25: out.append("v9 %.3f: P(>+.001) %.2f bronze %.2f (n=%d)" % (v, (prv[m] > 0.001).mean(), pmed(prv[m], 0, 2).mean(), m.sum()))
        print("   C25 %.3f%s | " % (d, "+" if d >= 0.945 else " ") + " | ".join(out))
