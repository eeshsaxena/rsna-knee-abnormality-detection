# quick_r6_slate.py (10-19, Claude): 10-minute re-check of the A19 slate with R6 (real ours leg, gold58 held-out) instead of R7 OOF.
# Same blend as the c4 kernel: rr = .75 pct(r384) + .25 pct(r224); b = w rr + (1-w) pct(ours); final = .8 pct(b) + .2 pct(S6fall).
# Paired study bootstrap (predictions fixed, ranks over all 58), plus split-half direction agreement for the Fracture and PF OA axes.
import numpy as np, pandas as pd, json
from scipy.stats import rankdata
D = 'g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
R6 = pd.read_csv(D + "/R6_gold58_serve.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, ["p|" + l for l in L]].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p
    if p == 0 or q == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def final(table, ours, A=0.25, ws6=0.2):
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = (1 - A) * pct(R384[:, j]) + A * pct(R224[:, j]); w = table.get(c, 0.5)
        b = w * rr + (1 - w) * pct(ours[:, j]); out[:, j] = (1 - ws6) * pct(b) + ws6 * pct(S6[:, j])
    return out
SL = {'c2max': {**C4, 'Fracture': 1.0, 'PF OA': 0.1}, 'fr100': {**C4, 'Fracture': 1.0}, 'fr90': {**C4, 'Fracture': 0.9},
      'c2': {**C4, 'Fracture': 0.9, 'PF OA': 0.1}, 'pf10': {**C4, 'PF OA': 0.1}, 'fr80': {**C4, 'Fracture': 0.8},
      'lat10': {**C4, 'Lateral OA': 0.1}, 'ef65': {**C4, 'Effusion': 0.65}}
rng = np.random.default_rng(7); B = 2000
idxs = []
while len(idxs) < B:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
def boot(P): return np.array([[auc(P[ix, j], Y[ix, j]) for j in range(12)] for ix in idxs])
res = {}
for leg, ours in (('R6', R6), ('R7', R7)):
    P0 = final(C4, ours); b0 = boot(P0); base = np.nanmean([auc(P0[:, j], Y[:, j]) for j in range(12)])
    print('%s leg: c4 macro %.5f | ours alone %.4f' % (leg, base, np.nanmean([auc(ours[:, j], Y[:, j]) for j in range(12)])))
    for k, t in SL.items():
        P1 = final(t, ours); pt = np.nanmean([auc(P1[:, j], Y[:, j]) for j in range(12)]) - base
        d = np.nanmean(boot(P1) - b0, axis=1); res[(leg, k)] = (pt, float((d > 0).mean()))
        print('  %-6s %+.5f  P>0 %.3f  sd %.5f' % (k, pt, (d > 0).mean(), d.std()))
json.dump({'%s|%s' % k: v for k, v in res.items()}, open('quick_r6_slate.json', 'w'), indent=1)
