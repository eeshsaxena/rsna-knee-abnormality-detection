import numpy as np, pickle, pandas as pd, sys, io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
res=pickle.load(open("fp32.pkl","rb"))
idx=pd.read_csv("C:/Users/eeshs/Downloads/rsna-knee/_crop_shards/study_index.csv")
N=len(idx)
def z(v):
    v=v.reshape(-1).astype(np.float64); v=v-v.mean(); n=np.linalg.norm(v); return v/n if n>0 else v
combos=[(0,1),(1,1),(2,1),(0,0),(1,0),(2,0)]
S=np.zeros((N,N)); C=np.zeros((N,N))
for pl,fb in combos:
    M=np.zeros((N,1024)); has=np.zeros(N,bool)
    for si in range(N):
        k=(si,pl,fb)
        if k in res: M[si]=z(res[k][0]); has[si]=True
    G=M@M.T
    hh=np.outer(has,has)
    S+=np.where(hh,G,0); C+=hh
    gi=G[np.triu_indices(N,1)][hh[np.triu_indices(N,1)]]
    print((pl,fb),"n",has.sum(),"pair corr quantiles", np.round(np.quantile(gi,[.5,.9,.99,.999,.9999,1]),3))
avg=np.where(C>=2,S/np.maximum(C,1),np.nan)
np.fill_diagonal(avg,np.nan)
iu=np.triu_indices(N,1); a=avg[iu]; ok=~np.isnan(a)
print("avg-over-shared-combos corr quantiles", np.round(np.nanquantile(a,[.5,.9,.99,.999,.9999,1]),4))
order=np.argsort(-np.nan_to_num(a,nan=-9))
top=[(iu[0][o],iu[1][o],a[o],C[iu[0][o],iu[1][o]]) for o in order[:60]]
np.save("avgsim.npy",avg.astype(np.float32))
for i,j,s,c in top[:60]: print(i,j,round(s,4),int(c))
