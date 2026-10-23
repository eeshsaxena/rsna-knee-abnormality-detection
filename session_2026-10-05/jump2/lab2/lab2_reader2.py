# lab2_reader2.py: does ANY existing second text reader add to steven_v4 on gold58? (rank-mean at weight w, CPU, read-only)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
ID="StudyInstanceUID"; RT="/home/training/rsna_train"; LI="/home/training/jump2/labels_in"
tr=pd.read_csv(f"{RT}/train.csv"); gold=tr[tr["ACL"].notna()].set_index(ID)[L].astype(int)
ids=pd.read_csv(f"{RT}/llm_labels_v4_blend.csv")[ID]
def rd(p): return pd.read_csv(p).drop_duplicates(ID).set_index(ID).reindex(ids)
v4=rd(f"{RT}/llm_labels_v4_blend.csv")[L].astype(float)
pk=rd(f"{RT}/report_labels_v2.csv"); pkv=pk[L].astype(float).fillna(pk[L].astype(float).median())
lx=rd(f"{LI}/lixin_gpt56_conf.csv"); lxs=pd.DataFrame({l: np.where(lx[l]>=.5,.5+.5*lx[l+"__conf"],.5-.5*lx[l+"__conf"]) for l in L}, index=ids)
X={"pilk_value(Opus-5 API)":pkv,"lixin_signedconf(GPT-5.6)":lxs,"jev_final2(6-LLM ens)":rd(f"{LI}/jev_final2_scores.csv")[L].astype(float),
   "steven_full":rd(f"{LI}/steven_full.csv")[L].astype(float),"qwen3b_laptop(ours)":rd(f"{LI}/qwen3b_laptop.csv")[L].astype(float)}
R=lambda d: d.rank(pct=True)
rng=np.random.default_rng(0); B=rng.integers(0,58,size=(2000,58))
def mac(S): Sg=S.loc[gold.index]; return np.mean([roc_auc_score(gold[l],Sg[l]) for l in L])
def boot(S):
    Sg=S.loc[gold.index].values; Gv=gold.values; out=[]
    for ix in B:
        a=[roc_auc_score(Gv[ix,j],Sg[ix,j]) for j in range(12) if 0<Gv[ix,j].sum()<58]; out.append(np.mean(a))
    return np.array(out)
b0=boot(R(v4)); print(f"v4 alone {mac(v4):.4f}")
for k,x in X.items():
    leak=np.mean(np.abs(x.loc[gold.index].values-gold.values)<1e-6)
    row=f"{k:28s} alone {mac(x):.4f} leak {leak:.2f} |"
    for w in [.2,.35,.5]:
        S=(1-w)*R(v4)+w*R(x); d=boot(S)-b0
        row+=f" w{w:.2f} {mac(S):.4f} (P>v4 {np.mean(d>0):.2f})"
    print(row)
