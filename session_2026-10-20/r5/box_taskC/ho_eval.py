# ho_eval.py (r5 task C): AGGREGATES ONLY. In-domain held-out OAI AUC of the OAI-trained twin set DT (T123v00, 5 folds)
# vs the OAI-free twin set D (same recipe, seed, folds; RSNA only) on follow-up knee-visits whose knee DT never saw.
# Cluster (subject) paired bootstrap. Labels: T123 allv soft labels binarised at >= .5 (the DT training definition).
import os, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
H = os.path.expanduser
LAB = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA"]
ids = [l.strip() for l in open(H("~/r5_oai/taskC/ids_heldout.txt")) if l.strip()]
L = pd.read_csv(H("~/oai_build/labels_v/oai_labels_T123_allv.csv")); L = L.set_index(L.columns[0]).loc[ids]
tr = set(l.strip() for l in open(H("~/oai_build/teacher/ids_v00.txt")) if l.strip()); trs = set(s.split("_")[1] for s in tr)
subj = np.array([s.split("_")[1] for s in ids]); strict = ~np.isin(subj, list(trs))
def load(pat):
    fs = sorted(glob.glob(H(f"~/r5_oai/taskC/preds/{pat}")));
    return fs, [pd.read_csv(f).set_index("StudyInstanceUID").loc[ids] for f in fs]
fD, PD = load("f?_s1043_ttr_frac75__raw__A__gpu.csv"); fT, PT = load("f?_s1043_ttr_frac75oaiT123v00__raw__A__gpu.csv")
print("members D", len(PD), "DT", len(PT))
def auc(s, y):
    p = y.sum(); q = len(y) - p
    if p == 0 or q == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def setmean(P): return sum(p[LAB].values for p in P) / len(P)
rng = np.random.default_rng(1020)
for cname, cm in (("knee-heldout", np.ones(len(ids), bool)), ("subject-strict", strict)):
    Y = L[LAB].values[cm]; ok = ~np.isnan(Y); Yb = (Y >= 0.5).astype(float)
    S = subj[cm]; us = np.unique(S); idx = {u: np.where(S == u)[0] for u in us}
    print(f"== {cname}: rows {cm.sum()} subjects {len(us)}")
    for setname, P in (("D (OAI-free)", PD), ("DT (OAI-trained)", PT)):
        M = setmean(P)[cm]
        a = [auc(M[ok[:, j], j], Yb[ok[:, j], j]) for j in range(len(LAB))]
        print(f"   {setname:18s} " + " ".join(f"{l[:9]} {v:.4f}" for l, v in zip(LAB, a)) + f" | OA-mean {np.mean(a[2:]):.4f} Men-mean {np.mean(a[:2]):.4f}")
    MD, MT = setmean(PD)[cm], setmean(PT)[cm]
    dl = np.array([auc(MT[ok[:, j], j], Yb[ok[:, j], j]) - auc(MD[ok[:, j], j], Yb[ok[:, j], j]) for j in range(len(LAB))])
    B = []
    for b in range(1000):
        pick = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        row = []
        for j in range(len(LAB)):
            m = pick[ok[pick, j]]; y = Yb[m, j]
            row.append(auc(MT[m, j], y) - auc(MD[m, j], y))
        B.append(row)
    B = np.array(B)
    for j, l in enumerate(LAB):
        print(f"   delta DT-D {l:16s} {dl[j]:+.4f} sd {np.nanstd(B[:, j]):.4f} P>0 {np.nanmean(B[:, j] > 0):.3f}")
    oa = B[:, 2:].mean(1); mn = B[:, :2].mean(1)
    print(f"   delta OA-mean {dl[2:].mean():+.4f} sd {oa.std():.4f} P>0 {np.mean(oa > 0):.3f} | Men-mean {dl[:2].mean():+.4f} sd {mn.std():.4f} P>0 {np.mean(mn > 0):.3f}")
    # fold-paired single members (D f vs DT f)
    fw = []
    for pd_, pt_ in zip(PD, PT):
        d_ = pd_[LAB].values[cm]; t_ = pt_[LAB].values[cm]
        fw.append(np.mean([auc(t_[ok[:, j], j], Yb[ok[:, j], j]) - auc(d_[ok[:, j], j], Yb[ok[:, j], j]) for j in range(2, 5)]))
    print("   fold-paired single-member OA-mean delta: " + " ".join(f"{v:+.4f}" for v in fw))
