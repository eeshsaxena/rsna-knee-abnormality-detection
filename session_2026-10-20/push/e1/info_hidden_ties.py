# info_hidden_ties.py (10-20, Claude): MODEL (simulation, no real data): how many exact rank ties does c4's float32 kernel path create
# at hidden-test size, and how much AUC can tie-breaking (which E1's squares do as a side effect) move? Gaussian-copula members with
# the gold58 member Spearman range (.82-.94, info_f32_ties / local_c4 logs), n = 390 public / 910 private / 1300 all, prevalence 10 %.
import numpy as np
from scipy.stats import rankdata
pct = lambda v: rankdata(v) / len(v)
rng = np.random.default_rng(11)
def sim(n, w, e1, prev=.10, rho=.88, reps=200):
    tied_rows, tied_pairs_frac = [], []
    for _ in range(reps):
        z = rng.standard_normal(n); C = [rho ** .5 * z + (1 - rho) ** .5 * rng.standard_normal(n) for _ in range(4)]   # r384, r224, R6, S6
        V = w * (.75 * pct(C[0]) ** e1 + .25 * pct(C[1]) ** e1) + (1 - w) * pct(C[2]) ** e1
        V = V.astype(np.float32).astype(np.float64)
        F = (.8 * pct(V) ** e1 + .2 * pct(C[3]) ** e1).astype(np.float32)
        y = (z + rng.standard_normal(n) * 1.0) > np.quantile(z + rng.standard_normal(n), 1 - prev)
        u, cnt = np.unique(F, return_counts=True); tied_rows.append(int((cnt[cnt > 1]).sum()))
        P, N = F[y], F[~y]
        if len(P) and len(N):
            tp = sum(int((N == p).sum()) for p in P); tied_pairs_frac.append(tp / (len(P) * len(N)))
    return np.mean(tied_rows), np.mean(tied_pairs_frac)
for n in (58, 390, 910):
    for w in (.5, .65, .8, .2):
        r1, f1 = sim(n, w, 1); r2, f2 = sim(n, w, 2)
        print('n %4d w %.2f | c4 path: tied rows %.1f, pos-neg pairs tied %.2e (max AUC move from breaking them %.1e) | E1 path: tied rows %.1f, pairs %.2e' % (
            n, w, r1, f1, f1 / 2, r2, f2))
