# Pre-registered small grid (all variants reported, none dropped): value of report-STATE gating + teacher choice.
import numpy as np, pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
ID="StudyInstanceUID"; RT="/home/training/rsna_train"
tr=pd.read_csv(f"{RT}/train.csv"); gold=tr[tr[L].notna().any(axis=1)].set_index(ID)[L]
ids=pd.read_csv(f"{RT}/llm_labels_v4_blend.csv")[ID]
v4=pd.read_csv(f"{RT}/llm_labels_v4_blend.csv").set_index(ID).reindex(ids)
wd=pd.read_csv(f"{RT}/teacher_train_probs.csv").set_index(ID).reindex(ids)
rp=pd.read_csv("labels_in/T_raptor_proxy.csv").set_index(ID).reindex(ids)
avg=(wd[L].rank(pct=True)+rp[L].rank(pct=True))/2
pk=pd.read_csv(f"{RT}/report_labels_v2.csv").set_index(ID).reindex(ids)
def lm(a, ref):
    la=logit(np.clip(a,1e-4,1-1e-4)); lr=logit(np.clip(ref,1e-4,1-1e-4))
    return expit((la-la.mean())/la.std()*lr.std()+lr.mean())
def build(te, wfun):
    out=pd.DataFrame(index=ids)
    for l in L:
        vd=pk[l+"__verdict"].fillna("UNK").values; w=wfun(vd, v4[l].values)
        out[l]=(1-w)*v4[l].values + w*lm(np.clip(te[l].values,1e-4,1-1e-4), v4[l].values)
    return out
def score(df):
    S=df.loc[gold.index]; a=[roc_auc_score(gold[l],S[l]) for l in L]; return np.mean(a), a
W={
 "flat50":        lambda vd,v: np.full(len(vd),.5),
 "SA(UNK.75)":    lambda vd,v: np.where(vd=="UNK",.75,.5),
 "3state(NO.25,UNK.75,YES.5)": lambda vd,v: np.select([vd=="NO",vd=="UNK"],[.25,.75],.5),
 "4state(+YESunc.65 if v4<.9)": lambda vd,v: np.select([vd=="NO",vd=="UNK",(vd=="YES")&(v<.9)],[.25,.75,.65],.45),
}
T={"wd":wd,"raptorproxy":rp,"avg(wd,rp)":avg}
print("v4 alone %.4f"%score(v4)[0])
rows=[]
for tn,te in T.items():
    print("teacher alone",tn,"%.4f"%score(te)[0])
    for wn,wf in W.items():
        m,a=score(build(te,wf)); rows.append((tn,wn,m)); print(f"  {tn:12s} {wn:30s} {m:.4f}  "+" ".join(f"{x:.3f}" for x in a))
