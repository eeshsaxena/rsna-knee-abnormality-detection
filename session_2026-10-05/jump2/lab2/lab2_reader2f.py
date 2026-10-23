# lab2_reader2f.py: does ANY existing second text reader add to steven_v4 on gold58? rank-mean at weight w.
# Fast vectorized paired bootstrap (1000 study resamples). CPU, read-only inputs.
import numpy as np, pandas as pd
from scipy.stats import rankdata
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
ID="StudyInstanceUID"; RT="/home/training/rsna_train"; LI="/home/training/jump2/labels_in"
tr=pd.read_csv(f"{RT}/train.csv"); gold=tr[tr["ACL"].notna()].set_index(ID)[L].astype(int)
ids=pd.read_csv(f"{RT}/llm_labels_v4_blend.csv")[ID]
def rd(p): return pd.read_csv(p).drop_duplicates(ID).set_index(ID).reindex(ids)
v4=rd(f"{RT}/llm_labels_v4_blend.csv")[L].astype(float)
pk=rd(f"{RT}/report_labels_v2.csv"); pkv=pk[L].astype(float).fillna(pk[L].astype(float).median())
lx=rd(f"{LI}/lixin_gpt56_conf.csv"); lxs=pd.DataFrame({l: np.where(lx[l]>=.5,.5+.5*lx[l+"__conf"],.5-.5*lx[l+"__conf"]) for l in L}, index=ids)
X={"pilk_value (Opus-5 API)":pkv,"lixin_signedconf (GPT-5.6)":lxs,"jev_final2 (6-LLM text ens)":rd(f"{LI}/jev_final2_scores.csv")[L].astype(float),
   "steven_full":rd(f"{LI}/steven_full.csv")[L].astype(float),"steven_v2":rd(f"{LI}/steven_v2.csv")[L].astype(float)}
R=lambda d: d.rank(pct=True)
Gv=gold.values; rng=np.random.default_rng(0); B=rng.integers(0,58,size=(1000,58))
def aucmat(S):  # returns (n_boot,) macro AUC
    Sv=S.loc[gold.index].values; out=np.zeros((len(B),12)); ok=np.zeros((len(B),12),bool)
    for j in range(12):
        y=Gv[B,j]; s=Sv[B,j]; r=rankdata(s,axis=1); n1=y.sum(1); n0=58-n1
        with np.errstate(invalid='ignore',divide='ignore'):
            out[:,j]=((r*y).sum(1)-n1*(n1+1)/2)/(n1*n0)
        ok[:,j]=(n1>0)&(n0>0)
    return np.nanmean(np.where(ok,out,np.nan),axis=1)
def mac(S):
    Sv=S.loc[gold.index].values; a=[]
    for j in range(12):
        r=rankdata(Sv[:,j]); y=Gv[:,j]; n1=y.sum(); a.append((r[y==1].sum()-n1*(n1+1)/2)/(n1*(58-n1)))
    return np.mean(a)
b0=aucmat(R(v4)); print(f"steven_v4 alone {mac(v4):.4f}")
for k,x in X.items():
    row=f"{k:28s} alone {mac(x):.4f} |"
    for w in [.2,.35,.5]:
        S=(1-w)*R(v4)+w*R(x); d=aucmat(S)-b0
        row+=f" w{w:.2f} {mac(S):.4f} (d {mac(S)-mac(v4):+.4f}, P>v4 {np.mean(d>0):.2f})"
    print(row)
S=(R(v4)+R(pkv)+R(lxs)+R(X["jev_final2 (6-LLM text ens)"]))/4; d=aucmat(S)-b0
print(f"rank-mean v4+pilk+lixin+jev (4 readers)  {mac(S):.4f} (P>v4 {np.mean(d>0):.2f})")
