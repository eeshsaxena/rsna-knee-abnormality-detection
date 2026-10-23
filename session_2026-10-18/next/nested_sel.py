# nested_sel.py (10-18, Claude): does "pick per-label Raptor weights from a gold58 sweep" generalise?
# Repeated split-half on gold58: choose w_j per label on half A (grid), score the chosen table vs the c4 table on half B.
# Same blend as the kernel (screen_a25.final): rr = (1-A) pct(r384) + A pct(r224); b = w rr + (1-w) pct(ours); final = .8 pct(b) + .2 pct(S6).
# Variants: free pick; shrink halfway to c4; Raptor-DOWN only; require an in-sample gain >= .005 on that label.
# Also: the held-out value of each fixed A19 entry when its labels were NOT re-picked (honest only for entries chosen before; reported for reference).
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
D = sys.argv[1]; R384F = sys.argv[2]; NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
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

def lab_score(idx, j, w, A=0.25, ws6=0.2):
    rr = (1 - A) * pct(R384[idx, j]) + A * pct(R224[idx, j])
    b = w * rr + (1 - w) * pct(R7[idx, j])
    return (1 - ws6) * pct(b) + ws6 * pct(S6[idx, j])

def lab_auc(idx, j, w): return auc(lab_score(idx, j, w), Y[idx, j])

rng = np.random.default_rng(0)
res = {k: [] for k in ["free", "shrink50", "down_only", "free_min005"]}
perlab = {k: np.zeros(12) for k in res}; cnt = np.zeros(12)
picks = {l: [] for l in L}
reps = 0
while reps < NREP:
    perm = rng.permutation(n); A_, B_ = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    if any(Y[A_, j].sum() in (0, len(A_)) or Y[B_, j].sum() in (0, len(B_)) for j in range(12)): continue
    reps += 1
    gains = {k: np.zeros(12) for k in res}
    for j, l in enumerate(L):
        w0 = C4.get(l, 0.5)
        tr = {w: lab_auc(A_, j, w) for w in GRID}; base_tr = lab_auc(A_, j, w0)
        wbest = max(GRID, key=lambda w: (tr[w], -abs(w - w0)))
        picks[l].append(wbest)
        down = [w for w in GRID if w <= w0]; wdown = max(down, key=lambda w: (tr[w], -abs(w - w0)))
        cand = {"free": wbest, "shrink50": round((wbest + w0) / 2, 3), "down_only": wdown,
                "free_min005": wbest if tr[wbest] - base_tr >= 0.005 else w0}
        te0 = lab_auc(B_, j, w0)
        for k, w in cand.items():
            gains[k][j] = lab_auc(B_, j, w) - te0
    for k in res:
        res[k].append(gains[k].mean()); perlab[k] += gains[k]
    cnt += 1
out = {}
for k, v in res.items():
    v = np.array(v); out[k] = dict(mean=float(v.mean()), sd=float(v.std()), p_gt0=float((v > 0).mean()), p_lt0=float((v < 0).mean()),
                                    per_label={l: round(float(perlab[k][j] / reps), 5) for j, l in enumerate(L)})
    print("%-12s held-out macro gain vs c4: mean %+.5f sd %.5f P>0 %.3f P<0 %.3f" % (k, v.mean(), v.std(), (v > 0).mean(), (v < 0).mean()))
print("per-label held-out gain (free pick):", {l: round(float(perlab['free'][j] / reps), 4) for j, l in enumerate(L)})
print("modal in-half picks:", {l: max(set(picks[l]), key=picks[l].count) for l in L})
print("reps", reps)
json.dump(dict(reps=reps, variants=out, modal_picks={l: max(set(picks[l]), key=picks[l].count) for l in L}),
          open(D + "/../nested_sel.json", "w"), indent=1)
