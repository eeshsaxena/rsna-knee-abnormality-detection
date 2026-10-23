# Extra checks: (1) optimism of the top public pick among many candidates; (2) value of the 2nd pick;
# (3) how often a cluster of 0.944 reads is explained by the parent's unknown bucket offset p0.
import numpy as np, glob, sys
from risk_sim import pmed, disp, SP
for PRI in ["opt", "cen", "pes"]:
    R = [r for f in sorted(glob.glob(SP + "/endgame3/risk_draws_%s*.npy" % PRI)) for r in np.load(f, allow_pickle=True)]
    ALL = ["C15", "C20", "C25", "C35", "C45", "F30", "F50", "CF"]
    PREF = {k: i for i, k in enumerate(["C25", "CF", "F30", "C20", "C15", "F50", "C35", "C45"])}
    def top(r, ks): return sorted(ks, key=lambda k: (-disp(r["p0"], r[k][0]), PREF[k]))
    opt1 = np.mean([(disp(r["p0"], r[top(r, ALL)[0]][0]) - 0.943 - r["p0"]) - r[top(r, ALL)[0]][1] for r in R]) * 1e4
    a = [r[top(r, ALL)[0]][1] for r in R]
    b = [max(r[top(r, ALL)[0]][1], r[top(r, ALL)[1]][1]) for r in R]
    best_single = max(np.mean(pmed(np.array([r[k][1] for r in R]), 0, 2)) for k in ALL)
    print("%s | top-1 of 8 by display: optimism %+.1f e-4, bronze %.2f | top-2: bronze %.2f | best fixed single %.2f"
          % (PRI, opt1, pmed(np.array(a), 0, 2).mean(), pmed(np.array(b), 0, 2).mean(), best_single))
# (3) p0 confound: a candidate with true public delta d reads >= .944 with prob P(p0 + d >= .0005)
for d in [0.0, 0.0002, 0.0005, 0.0010, 0.0015]:
    p = np.mean(np.round(0.943 + np.random.default_rng(1).uniform(-0.0005, 0.0005, 200000) + d, 3) >= 0.944)
    print("true public delta %+.4f (no noise) -> P(display >= .944) %.2f" % (d, p))
