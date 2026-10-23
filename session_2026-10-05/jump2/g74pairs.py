# g74pairs.py (jump2 blendmath, 2026-10-05): pairwise member agreement on gold74 (excluded from every fleet fold,
# so any fold-k model is out-of-sample there). Calibrates gold74 Spearman against fold-0 OOF Spearman / r_eff so that
# a later fold-0 vs fold-1 TTR mix50 pair on gold74 can be read as a fold-to-fold r_mm. CPU, tiny.
import os, sys, glob, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
H = os.path.expanduser
T = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker\x27s","Contusion","Fracture"]
fs = sorted(glob.glob(H("~/jump2/gold74_preds/*.csv")))
D = {os.path.basename(f)[:-4]: pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[T].astype(float) for f in fs}
ids = sorted(set.intersection(*[set(d.index) for d in D.values()]))
print("members", list(D), "n", len(ids))
def sp(a, b): return float(np.mean([spearmanr(a[:, j], b[:, j]).correlation for j in range(12)]))
X = {k: d.loc[ids, T].values for k, d in D.items()}
rng = np.random.default_rng(0)
for a, b in itertools.combinations(X, 2):
    s = sp(X[a], X[b]); bs = []
    for _ in range(200):
        k = rng.integers(0, len(ids), len(ids)); bs.append(sp(X[a][k], X[b][k]))
    print(f"{a:28s} {b:28s} sp_g74 {s:.3f} (boot sd {np.std(bs):.3f})")
