# CPU only: gold58 macro AUC of candidate RSNA target tables for a Raptor seed (no images, no OAI).
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
H="/home/training"
g=pd.read_csv(f"{H}/rsna_train/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].set_index("StudyInstanceUID")
def rd(p): d=pd.read_csv(p).set_index("StudyInstanceUID"); return d[T].astype(float)
v4=rd(f"{H}/rsna_train/llm_labels_v4_blend.csv"); sol=rd(f"{H}/rsna_train/labels_llm_gpt56sol.csv")
jev=rd(f"{H}/jump2/labels_in/jev_hybrid_scores.csv"); f75=rd(f"{H}/novel/frac75/labels_frac75_logit.csv")
pil=None
try: pil=rd(f"{H}/rsna_train/report_labels_v2.csv")
except Exception as e: print("pilk", e)
idx=v4.index
def blend(other):
    o=other.reindex(idx); b=(o+v4)/2; m=(v4==0.25)&o.notna(); b[m]=o[m]; b[o.isna()]=v4[o.isna()]; return b
def bj(b):
    j=jev.reindex(idx); out=(b+j)/2; out[j.isna()]=b[j.isna()]; return out
C={"v4":v4,"sol56":sol,"jev":jev,"frac75":f75,"blend_sol":blend(sol),"blendjev_sol":bj(blend(sol)),"v4jev":bj(v4)}
if pil is not None: C["pilk"]=pil; C["blendjev_pilk"]=bj(blend(pil))
print("gold58 n",len(g),"| v4==0.25 frac %.3f"%(v4==0.25).values.mean())
for k,d in C.items():
    d=d.reindex(g.index); a=[]
    for t in T:
        y=g[t].values; p=d[t].values; ok=np.isfinite(p)
        a.append(roc_auc_score(y[ok],p[ok]) if len(set(y[ok]))==2 else np.nan)
    print(f"{k:14s} macro {np.nanmean(a):.4f} | "+" ".join(f"{x:.3f}" for x in a), "| cov", int(np.isfinite(d.values).all(1).sum()))
