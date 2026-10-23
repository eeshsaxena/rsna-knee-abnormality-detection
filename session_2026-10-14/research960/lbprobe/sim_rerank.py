# sim_rerank.py: population and public-paired effect of re-ranking the Raptor mix (ours) vs not (varshitha), and of the r224 share.
import numpy as np, sim_noise as S
S.C[3, :3] = S.C[:3, 3] = .73; S.L = np.linalg.cholesky(S.C); rng = S.rng
def H(Sc, rerank, a224=.35):
    R7, r384, r224 = S.pct(Sc[:, 0]), S.pct(Sc[:, 1]), S.pct(Sc[:, 2])
    mix = (1 - a224) * r384 + a224 * r224
    return .5 * R7 + .5 * (S.pct(mix) if rerank else mix)
res = {"rerank vs none": [], "R7+224(none) vs R7 only": [], "a224 .35 vs .5": []}
for rep in range(40):
    out = {k: ([], []) for k in res}
    for p in S.PREV:
        n = S.N; y = (rng.random(n) < p).astype(int); e = rng.standard_normal((n, S.k)) @ S.L.T
        Sc = S.alpha[None, :] * y[:, None] + e
        perm = rng.permutation(n); pub, prv = perm[:S.NPUB], perm[S.NPUB:]
        a = H(Sc, True); b = H(Sc, False); r7only = .5 * S.pct(Sc[:, 0]) + .5 * S.pct(Sc[:, 1]); c5 = H(Sc, False, .5)
        for key, (u, v) in {"rerank vs none": (a, b), "R7+224(none) vs R7 only": (b, r7only), "a224 .35 vs .5": (c5, b)}.items():
            out[key][0].append(S.auc(u[pub], y[pub]) - S.auc(v[pub], y[pub])); out[key][1].append(S.auc(u[prv], y[prv]) - S.auc(v[prv], y[prv]))
    for k in res: res[k].append((np.mean(out[k][0]), np.mean(out[k][1])))
for k, v in res.items():
    v = np.array(v); print(f"{k:28s}: pub mean {v[:,0].mean():+.1e} sd {v[:,0].std():.1e} | priv mean {v[:,1].mean():+.1e} sd {v[:,1].std():.1e}")
