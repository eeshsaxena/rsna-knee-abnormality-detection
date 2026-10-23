# sim_cal.py RHO_F: recalibrate F's error correlation so that the population gain of wf 0 -> .2 matches the LB evidence (small).
import sys, numpy as np, sim_noise as S
from scipy.stats import rankdata
rho = float(sys.argv[1])
S.C[3, :3] = S.C[:3, 3] = rho
S.L = np.linalg.cholesky(S.C)
rng = S.rng
def popgain(pairs, Np=20000, reps=2):
    out = {p: [] for p in pairs}
    for r in range(reps):
        for li, prev in enumerate(S.PREV):
            y = (rng.random(Np) < prev).astype(int); e = rng.standard_normal((Np, S.k)) @ S.L.T
            Sc = S.alpha[None, :] * y[:, None] + e
            N0 = S.N; S.N = Np
            for p in pairs:
                (wa, ra), (wb, rb) = p
                out[p].append(S.auc(S.blend(Sc, wb, r384w=rb), y) - S.auc(S.blend(Sc, wa, r384w=ra), y))
            S.N = N0
    for p, v in out.items(): print(f"rho_F {rho}: population delta {p}: {np.mean(v):+.2e}")
popgain([((0,.65),(.2,.65)), ((.2,.65),(.3,.65)), ((.2,.65),(.4,.65)), ((.2,.65),(.1,.65)), ((.2,.65),(.2,.5)), ((.2,.65),(.2,.8)), ((.2,.65),(.2,1.0))])
