# sim_probe3.py: same as sim_probe2 with train report-label prevalences (MEASURED on train, GUESS for test) + run noise + rerank.
import sys, numpy as np, sim_noise as S
S.PREV = np.array([.278,.174,.533,.254,.367,.247,.452,.605,.132,.253,.294,.071])
rho = float(sys.argv[1]); reps = int(sys.argv[2])
S.C[3, :3] = S.C[:3, 3] = rho; S.L = np.linalg.cholesky(S.C); rng = S.rng
def pl(ys, Ss, wf_vec, idx):
    return np.array([S.auc(S.blend(Sc, wf_vec[i])[idx], y[idx]) for i, (y, Sc) in enumerate(zip(ys, Ss))])
probes = {"wf .2->.3": (np.full(12,.2), np.full(12,.3)), "wf .2->.1": (np.full(12,.2), np.full(12,.1)),
          "wf .2->.15": (np.full(12,.2), np.full(12,.15)), "wf .2->0": (np.full(12,.2), np.full(12,0.)),
          "wf .2->.4 Fracture only": (np.full(12,.2), np.r_[np.full(11,.2), .4]),
          "wf .2->.4 Effusion only": (np.full(12,.2), np.r_[np.full(7,.2), .4, np.full(4,.2)])}
res = {p: [] for p in probes}; runs = []
for rep in range(reps):
    ys, Ss = [], []
    for p in S.PREV:
        y = (rng.random(S.N) < p).astype(int); e = rng.standard_normal((S.N, S.k)) @ S.L.T
        ys.append(y); Ss.append(S.alpha[None, :] * y[:, None] + e)
    perm = rng.permutation(S.N); pub, prv = perm[:S.NPUB], perm[S.NPUB:]
    for name, (wa, wb) in probes.items():
        res[name].append(((pl(ys, Ss, wb, pub) - pl(ys, Ss, wa, pub)).mean(), (pl(ys, Ss, wb, prv) - pl(ys, Ss, wa, prv)).mean()))
    if rep < 3:
        base = S.macro(ys, Ss, .2, pub); v = np.array([S.macro(ys, Ss, .2, pub, noise=True) for _ in range(30)])
        runs.append((v.std(), np.mean(v == base)))
print("run noise public macro sd %.1e, P(exact) %.2f" % tuple(np.mean(runs, 0)))
for name, v in res.items():
    v = np.array(v); dp, dv = v[:, 0], v[:, 1]; pp = dv[dp > 0]
    print(f"rho {rho} | {name:24s}: pub {dp.mean():+.1e} sd {dp.std():.1e} | priv {dv.mean():+.1e} sd {dv.std():.1e} | P(pub>0) {np.mean(dp>0):.2f} | P(priv>0|pub>0) {np.mean(pp>0) if len(pp) else float('nan'):.2f}")
