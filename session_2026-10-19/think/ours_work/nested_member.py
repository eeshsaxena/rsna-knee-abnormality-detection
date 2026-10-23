# nested_member.py: "add one of our members only on the labels where it looks strong" tested out of sample on gold58.
# Half A (29 studies) picks the labels where adding the member at weight w beats c4; half B scores those picks vs c4.
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
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def col(j, idx, member=None, w=0.0, where="end"):
    c = L[j]
    rr = .75 * pct(R384[idx, j]) + .25 * pct(R224[idx, j]); wc = C4.get(c, 0.5)
    o = pct(R6[idx, j])
    if member is not None and where == "ours": o = (1 - w) * o + w * pct(member[idx, j])
    f = .8 * pct(wc * rr + (1 - wc) * o) + .2 * pct(S6[idx, j])
    if member is not None and where == "end": f = (1 - w) * pct(f) + w * pct(member[idx, j])
    return f


rng = np.random.default_rng(42)
for name, M, w, where in (("s43", S43, 0.1, "end"), ("s43", S43, 0.2, "end"), ("R7", R7, 0.2, "ours"), ("R7", R7, 0.1, "end"), ("R7", R7, 0.2, "end")):
    gains = []; npick = []; k = 0
    while k < 600:
        perm = rng.permutation(n); A, B = perm[:29], perm[29:]
        if any(Y[A, j].sum() in (0, 29) or Y[B, j].sum() in (0, 29) for j in range(12)): continue
        k += 1
        pick = [j for j in range(12) if auc(col(j, A, M, w, where), Y[A, j]) > auc(col(j, A), Y[A, j])]
        npick.append(len(pick))
        gB = np.mean([(auc(col(j, B, M, w, where), Y[B, j]) - auc(col(j, B), Y[B, j])) if j in pick else 0.0 for j in range(12)])
        gains.append(gB)
    gains = np.array(gains)
    print("%-4s w %.2f %-4s: labels picked on half A %.1f / 12 | held-out macro gain %+.5f (P>0 %.3f, sd %.5f)" % (
        name, w, where, np.mean(npick), gains.mean(), (gains > 0).mean(), gains.std()))
