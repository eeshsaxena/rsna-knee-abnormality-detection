# Report-gated teacher fill ("privileged-information gating"): the report decides WHERE the image teacher speaks.
# Cells whose report verdict is UNK (report silent): target = (1-W_SIL)*steven_v4 + W_SIL*teacher(logit-matched).
# Cells the report addresses: target = steven_v4 unchanged (W_ADDR = 0). Differs from mixSA (W_ADDR 0.5) only there.
# Usage: python build_gated.py [teacher_csv] [out_csv] [W_SIL] [W_ADDR]
import sys, numpy as np, pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
S = "C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
teacher = sys.argv[1] if len(sys.argv) > 1 else S + "/teacher_v2_out/raptor_proxy_v5v8.csv"
out = sys.argv[2] if len(sys.argv) > 2 else S + "/bigjump/labels_gated75_logit.csv"
W_SIL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
W_ADDR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
ids = pd.read_csv(S + "/labels_mix50_ecdf.csv")["StudyInstanceUID"]
v4 = pd.read_csv(S + "/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID").reindex(ids)
te = pd.read_csv(teacher).set_index("StudyInstanceUID").reindex(ids)
r2 = pd.read_csv(S + "/kneelabels/rsna-knee-llm-labels/report_labels_v2.csv").set_index("StudyInstanceUID").reindex(ids)
assert not v4[LAB].isna().any().any() and not te[LAB].isna().any().any()
def logit_match(a, ref):
    la = logit(np.clip(a, 1e-4, 1 - 1e-4)); lr = logit(np.clip(ref, 1e-4, 1 - 1e-4))
    return expit((la - la.mean()) / la.std() * lr.std() + lr.mean())
res = pd.DataFrame({"StudyInstanceUID": ids})
for l in LAB:
    w = np.where(r2[l + "__verdict"].fillna("UNK").values == "UNK", W_SIL, W_ADDR)
    res[l] = (1 - w) * v4[l].values + w * logit_match(te[l].values, v4[l].values)
res.to_csv(out, index=False)
# self-check on gold58 (target level) + within-class Spearman vs the teacher
g = pd.read_csv(S + "/kneelabels/jev/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].set_index("StudyInstanceUID")[LAB]
t = res.set_index("StudyInstanceUID").reindex(g.index); T = te.reindex(g.index); V = v4.reindex(g.index)
def rho(x):
    r = [spearmanr(x[c][g[c] == k], T[c][g[c] == k])[0] for c in LAB for k in (0, 1)]; return np.nanmean(r)
print("wrote", out, res.shape, f"W_SIL {W_SIL} W_ADDR {W_ADDR}")
print(f"gold58 target AUC: v4 {np.mean([roc_auc_score(g[c], V[c]) for c in LAB]):.4f} -> gated {np.mean([roc_auc_score(g[c], t[c]) for c in LAB]):.4f}; within-class rho vs teacher: v4 {rho(V):.3f} -> gated {rho(t):.3f}")
