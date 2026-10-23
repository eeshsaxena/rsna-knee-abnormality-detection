# hanley_noise.py: analytic (Hanley-McNeil) sampling sd of per-label and macro AUC at public 390 / private 910,
# two prevalence regimes (gold58 expert-like, train-report rates), and paired sd for estimate correlation r. MODEL.
import numpy as np
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
G58 = [.414, .155, .448, .397, .259, .19, .362, .603, .466, .207, .328, .31]      # MEASURED gold58 (n 58)
REP = [.204, .147, .394, .151, .359, .257, .450, .593, .126, .247, .168, .068]    # CLAIMED report rates (picks_cov.py)
# per-label AUC levels at a ~.96 macro public (GUESS profile; only the level of 1-A matters)
A = np.array([.990, .960, .955, .940, .970, .965, .955, .970, .935, .965, .955, .940])
def hm_var(a, npos, nneg):
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    return (a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg)
for reg, prev in [("gold58", G58), ("report", REP)]:
    for N in (390, 910):
        npos = np.maximum(np.round(N * np.array(prev)), 2); nneg = N - npos
        v = np.array([hm_var(a, p, q) for a, p, q in zip(A, npos, nneg)])
        macro = np.sqrt(v.sum()) / 12
        top = sorted(zip(np.sqrt(v), L, npos.astype(int)), reverse=True)[:4]
        share = (v / v.sum())
        print(f"{reg:7s} n{N}: macro abs sd {macro*1e4:5.1f}e-4 | top labels " + ", ".join(f"{l} {s:.4f} (npos {p})" for s, l, p in top)
              + f" | share of var top4 {np.sort(share)[-4:].sum():.2f}")
        for r in (.6, .7, .8, .9, .95):
            print(f"     paired sd at estimate corr r={r}: {macro*np.sqrt(2*(1-r))*1e4:5.1f}e-4")
