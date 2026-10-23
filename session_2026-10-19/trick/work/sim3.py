import numpy as np, pickle, pandas as pd, sys, io, time
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
res=pickle.load(open("fp32.pkl","rb"))
N=4407
def zr(a):
    a=a.reshape(len(a),-1).astype(np.float32); a=a-a.mean(1,keepdims=True); n=np.linalg.norm(a,axis=1,keepdims=True); return a/np.maximum(n,1e-6)
def bestmatch(pl,fb):
    rows=[];own=[]
    for si in range(N):
        k=(si,pl,fb)
        if k in res:
            a=res[k][1]
            sd=a.reshape(len(a),-1).std(1); a=a[sd>20] # drop low-content slices
            if len(a)==0: continue
            rows.append(zr(a)); own+=[si]*len(a)
    X=np.concatenate(rows); own=np.array(own)
    B=np.full((N,N),np.nan,np.float32)
    starts=np.r_[0,np.flatnonzero(np.diff(own))+1]; cols=own[starts]
    for s in range(0,len(X),4096):
        G=X[s:s+4096]@X.T
        R=np.maximum.reduceat(G,starts,axis=1)
        o=own[s:s+4096]
        for r_i in range(len(o)):
            B[o[r_i],cols]=np.fmax(B[o[r_i],cols],R[r_i])
    B=np.fmax(B,B.T)
    np.fill_diagonal(B,np.nan)
    return B
combos=[(0,1),(1,1),(2,1),(0,0),(1,0)]
Z=np.zeros((N,N),np.float32); C=np.zeros((N,N),np.float32)
for c in combos:
    B=bestmatch(*c)
    # CSLS-like hub correction: subtract half of each study's mean top-10 similarity
    top10=np.sort(np.nan_to_num(B,nan=-1),axis=1)[:,-10:].mean(1)
    Bc=B-0.5*top10[:,None]-0.5*top10[None,:]
    mu=np.nanmean(Bc); sd=np.nanstd(Bc)
    Zc=(Bc-mu)/sd
    has=~np.isnan(Zc)
    Z+=np.where(has,Zc,0); C+=has
    print(c,"done")
S=np.where(C>=3,Z/np.maximum(C,1),np.nan)
np.fill_diagonal(S,np.nan)
np.save("csls_score.npy",S)
iu=np.triu_indices(N,1); v=S[iu]; ok=~np.isnan(v)
print("score quantiles",np.round(np.quantile(v[ok],[.5,.99,.999,.9999,.99999,1]),3))
o=np.argsort(-np.nan_to_num(v,nan=-99))[:40]
for k in o: print(iu[0][k],iu[1][k],round(float(v[k]),3),int(C[iu[0][k],iu[1][k]]))
# per-study best partner score and gap to second
Sn=np.nan_to_num(S,nan=-99); srt=np.sort(Sn,axis=1)
gap=srt[:,-1]-srt[:,-2]
print("studies with best-partner z >= 6:",int((srt[:,-1]>=6).sum()),">=5:",int((srt[:,-1]>=5).sum()),">=4:",int((srt[:,-1]>=4).sum()))
