# wrongbase.py: same member, two bases. Gain of .8 base + .2 member over the base, on gold58: base = R7 OOF alone (the KC / S / K14
# gate form) vs base = the full c4 blend. Shows how much the single-model base inflates a member's apparent value.
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
D = sys.argv[1]
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(D + "/" + f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R6 = rd("R6_gold58_serve.csv"); S6 = rd("n42gpu_gold58.csv"); S43 = rd("s43_gold58.csv")
R7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, ["p|" + l for l in L]].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
c4 = np.zeros((n, 12))
for j, c in enumerate(L):
    rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); w = C4.get(c, .5)
    c4[:, j] = .8 * pct(w * rr + (1 - w) * pct(R6[:, j])) + .2 * pct(S6[:, j])
mac = lambda P: np.mean([auc(P[:, j], Y[:, j]) for j in range(12)])
add = lambda B, M, w: np.column_stack([(1 - w) * pct(B[:, j]) + w * pct(M[:, j]) for j in range(12)])
for mname, M in (("s43", S43), ("S6fall", S6), ("r384", R384), ("R7oof", R7), ("R6", R6)):
    for bname, B in (("R7 alone", R7), ("R6 alone", R6), ("c4 blend", c4)):
        if mname in bname: continue
        print("%-7s .2 on %-9s: %+.5f" % (mname, bname, mac(add(B, M, .2)) - mac(B)))
