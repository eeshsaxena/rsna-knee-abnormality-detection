# sim_probe2.py RHO_F: public vs private deltas for one-change probes (calibrated F), incl. single-label probes.
import sys, numpy as np, sim_noise as S
rho = float(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
S.C[3, :3] = S.C[:3, 3] = rho; S.L = np.linalg.cholesky(S.C); rng = S.rng
def per_label_auc(ys, Ss, wf_vec, idx, r384w=.65):
    return np.array([S.auc(S.blend(Sc, wf_vec[i], r384w=r384w)[idx], y[idx]) for i, (y, Sc) in enumerate(zip(ys, Ss))])
probes = {"wf .2->.3 all": (np.full(12, .2), np.full(12, .3)),
          "wf .2->.1 all": (np.full(12, .2), np.full(12, .1)),
          "wf .2->.4 all": (np.full(12, .2), np.full(12, .4)),
          "wf .2->.4 Fracture only": (np.full(12, .2), np.r_[np.full(11, .2), .4]),
          "wf .2->.4 Effusion only": (np.full(12, .2), np.r_[np.full(7, .2), .4, np.full(4, .2)]),
          "wf .2->.4 4 labels": (np.full(12, .2), np.r_[.4, .4, .2, .2, .4, .2, .2, .2, .2, .2, .4, .2])}
res = {p: [] for p in probes}; NP = 5000
for rep in range(reps):
    ys, Ss = [], []
    for p in S.PREV:
        y = (rng.random(S.N) < p).astype(int); e = rng.standard_normal((S.N, S.k)) @ S.L.T
        ys.append(y); Ss.append(S.alpha[None, :] * y[:, None] + e)
    perm = rng.permutation(S.N); pub, prv = perm[:S.NPUB], perm[S.NPUB:]
    for name, (wa, wb) in probes.items():
        dpub = (per_label_auc(ys, Ss, wb, pub) - per_label_auc(ys, Ss, wa, pub)).mean()
        dprv = (per_label_auc(ys, Ss, wb, prv) - per_label_auc(ys, Ss, wa, prv)).mean()
        res[name].append((dpub, dprv))
for name, v in res.items():
    v = np.array(v); dp, dv = v[:, 0], v[:, 1]
    pp = dv[dp > 0]
    print(f"rho {rho} | {name:26s}: pub {dp.mean():+.1e} sd {dp.std():.1e} | priv {dv.mean():+.1e} sd {dv.std():.1e} | P(pub>0) {np.mean(dp>0):.2f} | E[priv | pub>0] {pp.mean() if len(pp) else float('nan'):+.1e} | P(priv>0|pub>0) {np.mean(pp>0) if len(pp) else float('nan'):.2f}")
