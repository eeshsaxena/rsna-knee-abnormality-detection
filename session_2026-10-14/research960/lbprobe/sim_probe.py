# sim_probe.py: MODEL of public vs private deltas for one-change probes on the HFR7-RR structure.
import numpy as np
from sim_noise import *
NP = 6000   # "population" size per draw to estimate the true delta
def run(changes, reps=40):
    res = {c: [] for c in changes}
    for rep in range(reps):
        ys, Ss = [], []
        for p in PREV:
            y = (rng.random(N) < p).astype(int); e = rng.standard_normal((N, k)) @ L.T
            ys.append(y); Ss.append(alpha[None, :] * y[:, None] + e)
        perm = rng.permutation(N); pub, prv = perm[:NPUB], perm[NPUB:]
        for c in changes:
            (wa, ra), (wb, rb) = c
            dpub = macro(ys, Ss, wb, pub, r384w=rb) - macro(ys, Ss, wa, pub, r384w=ra)
            dprv = macro(ys, Ss, wb, prv, r384w=rb) - macro(ys, Ss, wa, prv, r384w=ra)
            res[c].append((dpub, dprv))
    for c, v in res.items():
        v = np.array(v); dp, dv = v[:, 0], v[:, 1]
        print(f"{c}: mean pub {dp.mean():+.2e} sd {dp.std():.2e} | priv mean {dv.mean():+.2e} sd {dv.std():.2e} | corr {np.corrcoef(dp, dv)[0,1]:+.2f} | P(sign pub==sign priv) {np.mean(np.sign(dp)==np.sign(dv)):.2f} | P(priv>0 | pub>0) {np.mean(dv[dp>0]>0) if (dp>0).any() else float('nan'):.2f}")
run([((.2,.65),(.3,.65)), ((.2,.65),(.4,.65)), ((.2,.65),(.1,.65)), ((.0,.65),(.2,.65)), ((.2,.65),(.2,.5)), ((.2,.65),(.2,.8))])
