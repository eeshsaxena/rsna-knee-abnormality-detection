# Target-level check (CPU only): silent-cell-only teacher fill vs flat/mixSA mixes.
# (a) gold58 macro AUC of each target table, paired bootstrap vs steven_v4 and vs mix50
# (b) mean per-label Spearman of each table vs the parent's own Raptor arms (raptor_proxy_v5v8) on non-gold train
#     studies = how parent-like a student trained on it is pushed to be.
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
S = "C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
ids = pd.read_csv(S + "/labels_mix50_ecdf.csv")["StudyInstanceUID"]
v4 = pd.read_csv(S + "/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID").reindex(ids)
wd = pd.read_csv(S + "/teacher_out/teacher_train_probs.csv").set_index("StudyInstanceUID").reindex(ids)
rp = pd.read_csv(S + "/teacher_v2_out/raptor_proxy_v5v8.csv").set_index("StudyInstanceUID").reindex(ids)
r2 = pd.read_csv(S + "/kneelabels/rsna-knee-llm-labels/report_labels_v2.csv").set_index("StudyInstanceUID").reindex(ids)
g = pd.read_csv(S + "/kneelabels/jev/gold74_eval_set.csv").set_index("StudyInstanceUID")
g = g[g["source"] == "original_gold58"]
gid = [s for s in g.index if s in set(ids)]
print("gold58 rows matched", len(gid), "| raptor proxy NaN rows", int(rp[LAB].isna().any(axis=1).sum()), "| widedense NaN rows", int(wd[LAB].isna().any(axis=1).sum()))
def logit_match(a, ref):
    a = np.asarray(a, float); ref = np.asarray(ref, float)
    la = logit(np.clip(a, 1e-4, 1 - 1e-4)); lr = logit(np.clip(ref, 1e-4, 1 - 1e-4))
    return expit((la - np.nanmean(la)) / np.nanstd(la) * np.nanstd(lr) + np.nanmean(lr))
unk = {l: (r2[l + "__verdict"].fillna("UNK").values == "UNK") for l in LAB}
def table(teacher, w_unk, w_addr):
    out = pd.DataFrame(index=ids)
    for l in LAB:
        t = teacher[l].values
        ok = np.isfinite(t)
        tm = np.full(len(t), np.nan); tm[ok] = logit_match(t[ok], v4[l].values[ok])
        w = np.where(unk[l], w_unk, w_addr)
        col = np.where(ok, (1 - w) * v4[l].values + w * np.nan_to_num(tm), v4[l].values)
        out[l] = col
    return out
T = {"steven_v4": v4[LAB].copy()}
for tn, te in (("wd", wd), ("raptor", rp)):
    T[f"{tn} flat0.50"] = table(te, 0.5, 0.5)
    T[f"{tn} mixSA(.75/.50)"] = table(te, 0.75, 0.5)
    for wu in (0.5, 0.75, 1.0):
        T[f"{tn} silent-only {wu:.2f}/0"] = table(te, wu, 0.0)
    T[f"{tn} silent-only 1.0/0.25"] = table(te, 1.0, 0.25)
Y = g.loc[gid, LAB].values
def gold_auc(tab, idx=None):
    X = tab.loc[gid, LAB].values
    if idx is not None: X, Yb = X[idx], Y[idx]
    else: Yb = Y
    a = []
    for j in range(len(LAB)):
        y = Yb[:, j] >= 0.5
        if 0 < y.mean() < 1: a.append(roc_auc_score(y, X[:, j]))
    return np.mean(a)
rng = np.random.default_rng(0); B = [rng.integers(0, len(gid), len(gid)) for _ in range(1000)]
base = {k: np.array([gold_auc(T[k], b) for b in B]) for k in ("steven_v4", "wd flat0.50", "raptor flat0.50")}
nong = [s for s in ids if s not in set(g.index)]
pr = rp.loc[nong, LAB]
okr = pr.notna().all(axis=1)
print(f"{'table':28s} gold58   d_vs_v4(SD)       d_vs_flat0.5_same_teacher(SD)   rho_vs_parent_Raptor   frac_cells_changed")
for k, tab in T.items():
    ga = gold_auc(tab)
    bs = np.array([gold_auc(tab, b) for b in B])
    d4 = bs - base["steven_v4"]
    ref = "wd flat0.50" if k.startswith("wd") else "raptor flat0.50"
    dm = bs - base[ref]
    rho = np.mean([spearmanr(tab.loc[nong, l][okr.values], pr[l][okr.values]).correlation for l in LAB])
    ch = float((np.abs(tab[LAB].values - v4[LAB].values) > 1e-9).mean())
    print(f"{k:28s} {ga:.4f}  {d4.mean():+.4f}({d4.std():.4f})  {dm.mean():+.4f}({dm.std():.4f}) P>0 {np.mean(dm>0):.2f}    {rho:.3f}      {ch:.3f}")
