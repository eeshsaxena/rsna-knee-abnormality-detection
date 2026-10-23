# blendspace_g58.py: c4 recipe on gold58 in rank space (as shipped) vs standardized-logit space vs probit-of-rank space.
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
R6 = pd.read_csv(D + "/R6_gold58_serve.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
print('ranges', [(round(float(M.min()), 3), round(float(M.max()), 3)) for M in (R6, R384, R224, S6)])
def auc(s, y):
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
pct = lambda v: rankdata(v) / len(v)
def zl(v):
    v = np.clip(v, 1e-6, 1 - 1e-6); z = np.log(v / (1 - v)); return (z - z.mean()) / (z.std() + 1e-9)
prq = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
def c4(T):
    P = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = .75 * T(R384[:, j]) + .25 * T(R224[:, j]); w = C4.get(c, .5)
        b = w * rr + (1 - w) * T(R6[:, j]); P[:, j] = .8 * T(b) + .2 * T(S6[:, j])
    return P
rng = np.random.default_rng(5); idxs = []
while len(idxs) < 2000:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
P0 = c4(pct); m0 = np.mean([auc(P0[:, j], Y[:, j]) for j in range(12)])
for name, T in [('logit-z', zl), ('probit-rank', prq)]:
    P1 = c4(T); m1 = np.mean([auc(P1[:, j], Y[:, j]) for j in range(12)])
    d = [np.mean([auc(P1[ix, j], Y[ix, j]) - auc(P0[ix, j], Y[ix, j]) for j in range(12)]) for ix in idxs[:600]]
    print('%-12s macro %.5f vs rank %.5f  delta %+.5f  P>0 %.3f sd %.5f' % (name, m1, m0, m1 - m0, np.mean(np.array(d) > 0), np.std(d)))
