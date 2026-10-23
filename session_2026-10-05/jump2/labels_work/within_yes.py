import pandas as pd, numpy as np, re
from sklearn.metrics import roc_auc_score
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
ID="StudyInstanceUID"; RT="/home/training/rsna_train"
tr=pd.read_csv(f"{RT}/train.csv"); gold=tr[tr[L].notna().any(axis=1)].set_index(ID)
S={"v4":pd.read_csv(f"{RT}/llm_labels_v4_blend.csv"),"mix50":pd.read_csv(f"{RT}/labels_mix50_logit.csv"),
   "teacher":pd.read_csv("labels_in/T_raptor_proxy.csv"),"jev":pd.read_csv("labels_in/jev_final2_scores.csv")}
S={k:v.set_index(ID).loc[gold.index] for k,v in S.items()}
pk=pd.read_csv(f"{RT}/report_labels_v2.csv").set_index(ID).reindex(gold.index)
print("label             nYES pos  " + " ".join(f"{k:>7s}" for k in S) + " | nNotYES pos " + " ".join(f"{k:>7s}" for k in S))
agg={k:[] for k in S}; agg2={k:[] for k in S}
for l in L:
    y=gold[l].values; m=(pk[l+"__verdict"]=="YES").values
    row=f"{l:17s} {m.sum():4d} {int(y[m].sum()):3d}  "
    for k in S:
        a=roc_auc_score(y[m],S[k][l].values[m]) if 0<y[m].sum()<m.sum() else np.nan; agg[k].append(a); row+=f"{a:7.3f} "
    n=~m; row+=f"| {n.sum():4d} {int(y[n].sum()):3d}    "
    for k in S:
        a=roc_auc_score(y[n],S[k][l].values[n]) if 0<y[n].sum()<n.sum() else np.nan; agg2[k].append(a); row+=f"{a:7.3f} "
    print(row)
print("mean within-YES  ", {k:round(np.nanmean(v),3) for k,v in agg.items()})
print("mean within-notYES", {k:round(np.nanmean(v),3) for k,v in agg2.items()})
