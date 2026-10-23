# Gold58 screen (n=58, research only): Raptor r384 multi-FOV TTA = mean(prob crop320 serve, prob full384) vs c4.
# Inputs read-only from session_2026-10-18/next/g58. Caveat: full384_k24 is the geometry nartaa used to RANK SWA epochs on gold58
# (CLAIMED 746792 / 750180), so any gain here is optimistic. Paired bootstrap over studies, ranks recomputed per resample.
import numpy as np, pandas as pd, json, sys
from scipy.stats import rankdata
D='C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
C4={"Medial OA":.5,"Lateral OA":.2,"PF OA":.5,"Synovitis":.8,"Baker's":.5,"Lateral Meniscus":.4,"Fracture":.65}
A224,WS6=.25,.2
g=pd.read_csv(D+'/gold74_eval_set.csv'); g=g[g.source=='original_gold58'].reset_index(drop=True)
ids=g.StudyInstanceUID.astype(str).tolist(); Y=g[L].values.astype(int); n=len(ids)
rd=lambda f: pd.read_csv(f,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
R6=rd(D+'/R6_gold58_serve.csv').loc[ids,L].values
R384=np.load(D+'/gold58_r384_kernelserve_cpu.npy'); R224=np.load(D+'/gold58_r224_kernelserve.npy')
C320=np.load(D+'/gold58_N_crop320_k24.npy'); F384=np.load(D+'/gold58_N_full384_k24.npy')
S6=rd(D+'/n42gpu_gold58.csv').loc[ids,L].values
pct=lambda v: rankdata(v)/len(v)
def auc(s,y):
    r=rankdata(s); p=y.sum(); q=len(y)-p
    return (r[y==1].sum()-p*(p+1)/2)/(p*q)
def c4(r384, idx=None):
    out=np.zeros((n,12))
    for j,c in enumerate(L):
        w=C4.get(c,.5); rr=(1-A224)*pct(r384[:,j])+A224*pct(R224[:,j]); b=w*rr+(1-w)*pct(R6[:,j])
        out[:,j]=(1-WS6)*pct(b)+WS6*pct(S6[:,j])
    return out
def macro(P,ix):
    return np.mean([auc(P[ix,j],Y[ix,j]) for j in range(12)])
def per(P): return [round(auc(P[:,j],Y[:,j]),4) for j in range(12)]
V={'c4':R384,'fov_mean_.5':.5*R384+.5*F384,'fov_mean_.33':(2/3)*R384+(1/3)*F384,'full384_only':F384}
base_ix=np.arange(n); res={}
B=np.random.default_rng(0)
boots=[B.integers(0,n,n) for _ in range(1500)]
boots=[b for b in boots if all(0<Y[b,j].sum()<len(b) for j in range(12))]
Pc4=c4(R384)
for k,r in V.items():
    P=c4(r); pt=macro(P,base_ix)-macro(Pc4,base_ix)
    d=np.array([macro(P,b)-macro(Pc4,b) for b in boots])
    res[k]={'macro':round(macro(P,base_ix),5),'delta_vs_c4':round(pt,5),'P(delta>0)':round(float((d>0).mean()),3),'sd':round(float(d.std()),5),'per_label':per(P)}
# standalone Raptor, like-for-like K=24
res['standalone']={'crop320_k24':round(macro(C320,base_ix),5),'full384_k24':round(macro(F384,base_ix),5),'mean_k24':round(macro(.5*C320+.5*F384,base_ix),5),'serve_r384':round(macro(R384,base_ix),5)}
res['n_boot']=len(boots)
json.dump(res,open(sys.argv[1],'w'),indent=1); print(json.dumps({k:(v if k in('standalone','n_boot') else {kk:vv for kk,vv in v.items() if kk!='per_label'}) for k,v in res.items()},indent=1))
