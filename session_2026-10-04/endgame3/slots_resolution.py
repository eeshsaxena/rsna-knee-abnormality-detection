# Read resolution (CPU, small): how precisely does ONE public read (390 studies, 3-decimal display) measure
#  (1) a member's solo macro AUC, (2) a parent+fleet blend delta, (3) a PAIRED difference between two blends from
#  the same commit (fleet with vs without a second family), vs the same quantities on the private 910.
# Binormal geometry identical to round2/lens_rules/noise_calib.py.
import numpy as np
from scipy.stats import norm, rankdata
rng = np.random.default_rng(23)
prev = np.array([.204, .147, .394, .151, .359, .257, .450, .593, .126, .247, .168, .068])
A_P = np.array([.985, .95, .93, .92, .955, .95, .93, .945, .905, .96, .94, .93]); A_P = A_P - (A_P.mean() - 0.943)
dP = np.sqrt(2) * norm.ppf(A_P); N, NPUB = 1300, 390
def kscale(t):
    lo, hi = 0.1, 1.5
    for _ in range(40):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if norm.cdf(k * dP / np.sqrt(2)).mean() < t else (lo, k)
    return (lo + hi) / 2
def macro(S, Y):
    r = rankdata(S, axis=0); npos = Y.sum(0); nneg = len(Y) - npos
    return (((r * Y).sum(0) - npos * (npos + 1) / 2) / (npos * nneg)).mean()
k1, k2 = kscale(0.930), kscale(0.925)
r1, r2, r12 = 0.80, 0.80, 0.86     # F1 vs parent, F2 vs parent, F1 vs F2 (within-class)
W = 0.35; rows = []
for t in range(400):
    Y = (rng.random((N, 12)) < prev).astype(float)
    eP = rng.standard_normal((N, 12)); z1 = rng.standard_normal((N, 12)); z2 = rng.standard_normal((N, 12))
    e1 = r1 * eP + np.sqrt(1 - r1 ** 2) * z1
    a = (r12 - r1 * r2) / np.sqrt(1 - r1 ** 2); b = np.sqrt(max(1 - r2 ** 2 - a ** 2, 1e-6)); e2 = r2 * eP + a * z1 + b * z2
    SPm = dP * Y + eP; S1 = k1 * dP * Y + e1; S2 = k2 * dP * Y + e2
    rP, r1_, r2_ = (rankdata(S, axis=0) / N for S in (SPm, S1, S2))
    fl = (r1_ + r2_) / 2                                # 2-family fleet (rank mean as a stand-in for the raw mean)
    A = (1 - W) * rP + W * r1_; B = (1 - W) * rP + W * fl
    idx = rng.permutation(N); pu, pr = idx[:NPUB], idx[NPUB:]
    row = []
    for sub in (pu, pr):
        p0 = macro(SPm[sub], Y[sub])
        row += [macro(S1[sub], Y[sub]), macro(A[sub], Y[sub]) - p0, macro(B[sub], Y[sub]) - macro(A[sub], Y[sub])]
    rows.append(row)
R = np.array(rows)
names = ["F1 solo AUC", "P+F1 delta @W.35", "paired: (P+F1F2) - (P+F1) @W.35"]
print("quantity | mean public | sd public | sd private | sd(public - private) | corr(pub, priv)")
for j, n in enumerate(names):
    pu, pv = R[:, j], R[:, 3 + j]
    print("%-36s %+.5f  %.5f  %.5f  %.5f  %.2f" % (n, pu.mean(), pu.std(), pv.std(), (pu - pv).std(), np.corrcoef(pu, pv)[0, 1]))
