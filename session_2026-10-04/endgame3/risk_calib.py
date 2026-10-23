# Anchor check: does the binormal model reproduce the MEASURED public deltas at some rho? (public split only)
import numpy as np, sys
from risk_sim import world
rng = np.random.default_rng(11)
def run(m, r, n=120):
    out = {k: [] for k in ["C15", "C20", "C25", "C35", "C45"]}
    for _ in range(n):
        c = world(rng, m, r, 0.92, 0.8, 0.6)
        for k in out: out[k].append(c[k][0])
    return {k: np.mean(v) * 1e4 for k, v in out.items()}
for m, r in [(0.894, 0.80), (0.894, 0.85), (0.927, 0.80), (0.927, 0.85), (0.927, 0.88), (0.927, 0.91), (0.924, 0.82), (0.924, 0.86), (0.924, 0.90)]:
    d = run(m, r); print("m %.3f rho %.2f | public delta e-4: " % (m, r) + " ".join("%s %+.1f" % kv for kv in d.items()), flush=True)
