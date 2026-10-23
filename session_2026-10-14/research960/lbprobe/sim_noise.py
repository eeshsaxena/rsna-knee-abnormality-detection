# sim_noise.py: MODEL of (a) hidden-rerun noise from Raptor nondeterminism and (b) public sampling noise of small blend-weight
# changes, for the HFR7-RR structure: final = (1-wf) * [ .5 R7 + .5 (.65 r384 + .35 r224) ] + wf * F, all per-label percentile ranks.
import numpy as np
from scipy.stats import rankdata, norm
rng = np.random.default_rng(0)
N, NPUB = 1300, 390
PREV = np.array([.12,.06,.25,.15,.30,.10,.30,.45,.20,.12,.20,.05])   # GUESS-level prevalences, 12 labels
def auc(s, y):
    r = rankdata(s); npos = y.sum(); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
def pct(x): return rankdata(x) / len(x)
# component quality (single-model AUC) and error correlation (GUESS calibrated so R7+Raptor gains ~.005 over parts, F adds little)
A = {"R7": .950, "r384": .949, "r224": .945, "F": .944}
names = list(A); k = len(names)
alpha = np.array([np.sqrt(2) * norm.ppf(A[m]) for m in names])
C = np.full((k, k), .62); np.fill_diagonal(C, 1.0)
C[1, 2] = C[2, 1] = .88          # r384 vs r224 (same family)
C[3, 0] = C[0, 3] = .70; C[3, 1] = C[1, 3] = C[3, 2] = C[2, 3] = .66   # F (ConvNeXt-T fleet) closer to R7 (ConvNeXt)
L = np.linalg.cholesky(C)
LOGIT_SCALE = 1.6                 # latent units -> logit units (GUESS: logits sd ~2.5)
SD384, SD224 = 6.4e-5, 1.3e-4     # MEASURED run noise in logit units (8 / 7 commits, 3 rows x 12 labels, median)
def draw_pop():
    ys, Ss = [], []
    for p in PREV:
        y = (rng.random(N) < p).astype(int)
        e = rng.standard_normal((N, k)) @ L.T
        S = alpha[None, :] * y[:, None] + e
        ys.append(y); Ss.append(S)
    return ys, Ss
def blend(S, wf, noise=False, r384w=.65):
    R7, r384, r224, F = S[:, 0], S[:, 1].copy(), S[:, 2].copy(), S[:, 3]
    if noise:
        r384 = r384 + rng.normal(0, SD384 / LOGIT_SCALE, N); r224 = r224 + rng.normal(0, SD224 / LOGIT_SCALE, N)
    rap = r384w * pct(r384) + (1 - r384w) * pct(r224)
    H = .5 * pct(R7) + .5 * pct(rap)
    return (1 - wf) * pct(H) + wf * pct(F)
def macro(ys, Ss, wf, idx, noise=False, r384w=.65):
    return np.mean([auc(blend(S, wf, noise, r384w)[idx], y[idx]) for y, S in zip(ys, Ss)])
if __name__ == "__main__":
    # (a) run noise: same population, same public subset, rerun Raptor noise
    out = []
    for rep in range(3):
        ys, Ss = draw_pop(); pub = rng.permutation(N)[:NPUB]
        base = macro(ys, Ss, .2, pub)
        vals = np.array([macro(ys, Ss, .2, pub, noise=True) for _ in range(60)])
        out.append((np.std(vals), np.mean(vals == base), np.max(np.abs(vals - base))))
    print("RUN NOISE (public macro AUC): sd %.2e  P(exact same) %.2f  max|d| %.2e" % tuple(np.mean(out, axis=0)))
