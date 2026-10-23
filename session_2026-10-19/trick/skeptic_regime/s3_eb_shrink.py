# s3_eb_shrink.py (skeptic, CPU, research only): how much of the OA-trio in-sample gain survives shrinkage toward the family of
# all 220 within-triple decorrelations (c4, gold58)?
#  (1) bootstrap noise sd of every triple's macro delta (paired, same resamples) vs the cross-triple spread -> empirical-Bayes shrink
#  (2) nested: pick the best of 220 triples on an inner half, score on the outer half; and fixed OA trio on the same outer halves
import itertools, numpy as np, pandas as pd
from scipy.stats import rankdata
G = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
WR = {'Medial OA': .5, 'Lateral OA': .2, 'PF OA': .5, 'Synovitis': .8, "Baker's": .5, 'Lateral Meniscus': .4, 'Fracture': .65}
F = np.zeros((n, 12))
for j, c in enumerate(L):
    rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); w = WR.get(c, .5)
    b = w * rr + (1 - w) * pct(R6[:, j]); F[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
TRI = list(itertools.combinations(range(12), 3)); OAi = TRI.index((4, 5, 6))


def all_deltas(ix, a):
    X, Yy = F[ix], Y[ix]
    U = np.stack([pct(X[:, j]) for j in range(12)], 1)
    b = np.array([auc(X[:, j], Yy[:, j]) for j in range(12)])
    cache = {}
    out = np.empty(len(TRI))
    for k, tri in enumerate(TRI):
        d = 0.
        for t in tri:
            o = tuple(h for h in tri if h != t)
            key = (t, o)
            if key not in cache:
                cache[key] = auc(U[:, t] - a * .5 * (U[:, o[0]] + U[:, o[1]]), Yy[:, t]) - b[t]
            d += cache[key]
        out[k] = d / 12
    return out


ok = lambda ix: all(0 < Y[ix, j].sum() < len(ix) for j in range(12))
rng = np.random.default_rng(7)
for a in (.1, .2):
    full = all_deltas(np.arange(n), a)
    B = []
    while len(B) < 300:
        ix = rng.integers(0, n, n)
        if ok(ix): B.append(all_deltas(ix, a))
    B = np.array(B)
    noise_var = B.var(0)                      # per-triple bootstrap variance
    V_tot = full.var(); V_noise = noise_var.mean(); V_sig = max(V_tot - V_noise, 0.)
    mu = full.mean(); Bk = V_sig / (V_sig + noise_var[OAi]) if V_sig > 0 else 0.
    shr = mu + Bk * (full[OAi] - mu)
    # bootstrap rank of the OA trio among triples (how often is it the best?)
    best_is_oa = (B.argmax(1) == OAi).mean(); oa_rank = (B > B[:, [OAi]]).sum(1) + 1
    print('a %.1f: OA in-sample %+.5f (rank %d/220); cross-triple mean %+.5f sd %.5f; mean bootstrap noise sd %.5f (OA %.5f)' % (
        a, full[OAi], int((full > full[OAi]).sum()) + 1, mu, np.sqrt(V_tot), np.sqrt(V_noise), np.sqrt(noise_var[OAi])))
    print('       signal sd %.5f; EB shrink factor %.2f -> shrunk OA %+.5f | boot: OA best %.3f, median rank %d' % (
        np.sqrt(V_sig), Bk, shr, best_is_oa, int(np.median(oa_rank))))
    # nested: best triple on inner half vs fixed OA trio, outer half
    pick, fixed, chose_oa = [], [], 0; cnt = 0
    while cnt < 300:
        perm = rng.permutation(n); h1, h2 = perm[:29], perm[29:]
        if not (ok(h1) and ok(h2)): continue
        cnt += 1
        d1, d2 = all_deltas(h1, a), all_deltas(h2, a)
        for din, dout in ((d1, d2), (d2, d1)):
            k = int(din.argmax()); chose_oa += k == OAi
            pick.append(dout[k] if din[k] > 0 else 0.); fixed.append(dout[OAi])
    pick, fixed = np.array(pick), np.array(fixed)
    print('       nested pick-best-of-220 outer mean %+.5f (P>0 %.3f, P<0 %.3f; OA chosen %.3f) | fixed OA trio outer mean %+.5f (P>0 %.3f, P<0 %.3f)' % (
        pick.mean(), (pick > 0).mean(), (pick < 0).mean(), chose_oa / len(pick), fixed.mean(), (fixed > 0).mean(), (fixed < 0).mean()))
