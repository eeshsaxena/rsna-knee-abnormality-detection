# g58_twin.py (r5 task C): RSNA-only. Per-label gold58 / gold74 AUC of the OAI-trained fleet set DT (K14, T123v00, 5 folds)
# vs the OAI-free fleet set D2 (s1044 frac75, 5 folds; the exact s1043 twin's gold preds are not on disk). Paired bootstrap.
import sys, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
R = "C:/Users/eeshs/Downloads/rsna-knee/"
BP = R + "session_2026-10-19/think/ours_work/boxpreds/"
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
g0 = pd.read_csv(R + "session_2026-10-18/next/g58/gold74_eval_set.csv", dtype={"StudyInstanceUID": str})
def mean_of(pat, ids):
    fs = sorted(glob.glob(BP + pat)); assert len(fs) == 5, (pat, len(fs))
    return sum(pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values for f in fs) / 5
def auc(s, y):
    m = ~np.isnan(y); s, y = s[m], y[m].astype(int)
    p = y.sum(); q = len(y) - p
    if p == 0 or q == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
for tag, g in (("gold58", g0[g0.source == "original_gold58"]), ("gold74", g0)):
    ids = g.StudyInstanceUID.tolist(); Yf = g[L].values.astype(float); n = len(ids)
    DT = mean_of("k14_test/gold/preds/f*_s1043_ttr_frac75oaiT123v00__raw__A.csv", ids)
    D2 = mean_of("r7t_labels/gold_preds/f*_s1044_ttr_frac75__raw__A.csv", ids)
    Y = Yf; a = np.array([[auc(M[:, j], Y[:, j]) for j in range(12)] for M in (D2, DT)])
    rng = np.random.default_rng(7); B = []
    for b in range(3000):
        i = rng.integers(0, n, n); row = []
        for j in range(12):
            y = Y[i, j]; row.append(auc(DT[i, j], y) - auc(D2[i, j], y))
        B.append(row)
    B = np.array(B)
    print(f"== {tag} n {n}: macro D2 {np.nanmean(a[0]):.4f} DT {np.nanmean(a[1]):.4f} delta {np.nanmean(a[1]-a[0]):+.4f} sd {np.nanstd(np.nanmean(B,1)):.4f}")
    for j, l in enumerate(L):
        print(f"   {l:17s} pos {int(np.nansum(Y[:, j])):3d}/{int((~np.isnan(Y[:, j])).sum()):2d} D2 {a[0, j]:.4f} DT {a[1, j]:.4f} delta {a[1, j]-a[0, j]:+.4f} sd {np.nanstd(B[:, j]):.4f} P>0 {np.nanmean(B[:, j] > 0):.2f}")
    oa = B[:, 4:7].mean(1); mn = B[:, 2:4].mean(1)
    print(f"   OA-mean delta {np.mean(a[1, 4:7]-a[0, 4:7]):+.4f} sd {np.nanstd(oa):.4f} P>0 {np.nanmean(oa > 0):.2f} | menisci-mean {np.mean(a[1, 2:4]-a[0, 2:4]):+.4f} sd {np.nanstd(mn):.4f} P>0 {np.nanmean(mn > 0):.2f}")
