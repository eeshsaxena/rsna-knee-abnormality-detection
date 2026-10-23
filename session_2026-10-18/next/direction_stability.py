# direction_stability.py (10-18, Claude): for each candidate change vs c4, over repeated split-halves of gold58:
#  (1) how often half A alone would pick the same DIRECTION (up / down vs the c4 value), and
#  (2) the held-out (half B) gain of the FIXED candidate value when half A agreed vs disagreed,
#  (3) the plain held-out gain distribution of the fixed candidate (variance check).
# Same blend as the kernel. Light CPU only.
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
D = sys.argv[1]; R384F = sys.argv[2]; NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
CAND = [("Fracture", 0.8), ("Fracture", 0.9), ("Fracture", 1.0), ("PF OA", 0.1), ("PF OA", 0.3), ("Lateral OA", 0.1), ("Lateral OA", 0.0),
        ("Effusion", 0.65), ("Contusion", 0.2), ("Medial Meniscus", 0.6), ("Baker's", 0.65)]
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
R7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, ["p|" + l for l in L]].values
R384 = np.load(R384F); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
pct = lambda v: rankdata(v) / len(v)
GRID = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]
def auc(s, y):
    p = y.sum(); q = len(y) - p
    if p == 0 or q == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def lab_auc(idx, j, w, A=0.25, ws6=0.2):
    rr = (1 - A) * pct(R384[idx, j]) + A * pct(R224[idx, j]); b = w * rr + (1 - w) * pct(R7[idx, j])
    return auc((1 - ws6) * pct(b) + ws6 * pct(S6[idx, j]), Y[idx, j])
rng = np.random.default_rng(1); out = {}
stats = {c: dict(agree=0, held=[], held_agree=[], held_dis=[]) for c in CAND}
reps = 0
while reps < NREP:
    perm = rng.permutation(n); A_, B_ = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    if any(Y[A_, L.index(l)].sum() in (0, len(A_)) or Y[B_, L.index(l)].sum() in (0, len(B_)) for l, _ in CAND): continue
    reps += 1
    for (l, w) in CAND:
        j = L.index(l); w0 = C4.get(l, 0.5)
        tr = {x: lab_auc(A_, j, x) for x in GRID}; pick = max(GRID, key=lambda x: (tr[x], -abs(x - w0)))
        same = (pick - w0) * (w - w0) > 0
        gain = (lab_auc(B_, j, w) - lab_auc(B_, j, w0)) / 12.0     # macro contribution
        s = stats[(l, w)]; s['agree'] += int(same); s['held'].append(gain); (s['held_agree'] if same else s['held_dis']).append(gain)
for (l, w), s in stats.items():
    h = np.array(s['held'])
    out['%s %.2f' % (l, w)] = dict(dir_agree=round(s['agree'] / reps, 3), held_mean=float(h.mean()), held_p_gt0=float((h > 0).mean()),
                                   held_mean_if_agree=float(np.mean(s['held_agree'])) if s['held_agree'] else None)
    print("%-17s %.2f (c4 %.2f) | half-A picks same direction %.2f | held-out macro %+.5f P>0 %.2f" % (
        l, w, C4.get(l, 0.5), s['agree'] / reps, h.mean(), (h > 0).mean()))
json.dump(dict(reps=reps, cand=out), open(D + "/../direction_stability.json", "w"), indent=1)
print("reps", reps)
