# Binormal (equal-variance, macro-as-one-label) blend model, re-anchored on public probes (10-08).
# MODEL only. Historic optimism of this model: +.0000 to +.0009 (always optimistic); subtract ~.0005.
import numpy as np
from scipy.stats import norm
d = lambda a: np.sqrt(2) * norm.ppf(a)
A = lambda x: norm.cdf(x / np.sqrt(2))
def blend(aucs, w, R):
    w = np.asarray(w, float); w = w / w.sum(); dv = np.array([d(a) for a in aucs])
    return A(w @ dv / np.sqrt(w @ R @ w))
def r_from_pair(a1, a2, w1, target):
    # solve r such that blend(a1,a2,w1) == target
    lo, hi = 0.5, 0.9999
    for _ in range(80):
        m = (lo + hi) / 2
        R = np.array([[1, m], [m, 1]])
        if blend([a1, a2], [w1, 1 - w1], R) > target: lo = m
        else: hi = m
    return (lo + hi) / 2
print("== implied correlations from public probes (display rounding +-.0005 on each read)")
for N in (0.9485, 0.949, 0.9495):
    for B in (0.9485, 0.949, 0.9495):
        if B < N - 0.0006: continue
        rPN = r_from_pair(N, 0.943, 0.6, B)
        print(f"N {N:.4f}  N.6+P.4={B:.4f} -> r_PN {rPN:.3f}")
for N in (0.9485, 0.949, 0.9495):
    for B in (0.9475, 0.948, 0.9485):
        rRN = r_from_pair(N, 0.935, 0.7, B)
        print(f"N {N:.4f}  N.7+R(.935).3={B:.4f} -> r_RN {rRN:.3f}")
print("== K9N candidates: P .943, F .944, N .949; r_PF .945; r_PN .94/.96; r_FN scan")
cands = {"K9N-a P.10/F.30/N.60": (.10, .30, .60), "K9N-b P.30/F.30/N.40": (.30, .30, .40),
         "FN P0/F.30/N.70": (0, .30, .70), "FN P0/F.35/N.65": (0, .35, .65), "FN P0/F.40/N.60": (0, .40, .60),
         "P.05/F.30/N.65": (.05, .30, .65), "N alone": (0, 0, 1)}
for rPN in (0.94, 0.96):
    for rFN in (0.88, 0.90, 0.92, 0.94):
        R = np.array([[1, .945, rPN], [.945, 1, rFN], [rPN, rFN, 1]])
        if np.linalg.eigvalsh(R).min() <= 0: continue
        row = "  ".join(f"{k.split()[0]}:{blend([.943,.944,.949], w, R):.4f}" for k, w in cands.items())
        print(f"rPN {rPN} rFN {rFN}: {row}")
print("labels:", list(cands))
print("== what a stronger own family buys (F solo s, r_FN), P=0, best W_F")
for s in (0.944, 0.947, 0.950):
    for rFN in (0.86, 0.90, 0.93):
        best = max((blend([s, .949], [wf, 1 - wf], np.array([[1, rFN], [rFN, 1]])), wf) for wf in np.arange(0, 1.01, .05))
        print(f"F solo {s:.3f} r_FN {rFN:.2f}: best {best[0]:.4f} at W_F {best[1]:.2f}")
