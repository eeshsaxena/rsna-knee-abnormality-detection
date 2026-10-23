import numpy as np, pickle, pandas as pd, sys, io, time
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
res=pickle.load(open("fp32.pkl","rb"))
N=4407
def zr(a):
    a=a.reshape(len(a),-1).astype(np.float32); a=a-a.mean(1,keepdims=True); n=np.linalg.norm(a,axis=1,keepdims=True); return a/np.maximum(n,1e-6)
# 1) self-split baseline for mean-image descriptor (same series, disjoint random slices)
rng=np.random.default_rng(0); ss=[]; ssl=[]
for (si,pl,fb),(m,a) in res.items():
    if pl==0 and fb==1 and len(a)>=8:
        p=rng.permutation(len(a)); A=a[p[:len(a)//2]].mean(0); B=a[p[len(a)//2:]].mean(0)
        ss.append(float(zr(A[None])[0]@zr(B[None])[0]))
print("self-split mean-image corr (sag FS) quantiles", np.round(np.quantile(ss,[.01,.1,.5,.9]),3))
# 2) per-slice best-match descriptor, sagittal FS and coronal FS; all pairs, block-wise
def bestmatch(pl,fb):
    rows=[];own=[]
    for si in range(N):
        k=(si,pl,fb)
        if k in res:
            a=zr(res[k][1]); rows.append(a); own+= [si]*len(a)
    X=np.concatenate(rows); own=np.array(own)
    B=np.full((N,N),np.nan,np.float32)
    # within-study best non-identical slice match (self baseline: different random slices of same series)
    t=time.time()
    bs=4096
    for s in range(0,len(X),bs):
        G=X[s:s+bs]@X.T   # (bs, M)
        o=own[s:s+bs]
        # max per (row study, col study): reduce columns by study
        # use np.maximum.reduceat over sorted own (own is sorted ascending)
        starts=np.r_[0,np.flatnonzero(np.diff(own))+1]
        cols=own[starts]
        R=np.maximum.reduceat(G,starts,axis=1)  # (bs, nstudies_present)
        for r_i in range(len(o)):
            row=B[o[r_i],cols]
            B[o[r_i],cols]=np.fmax(row,R[r_i])
    print("block time",round(time.time()-t))
    return B
for pl,fb in [(0,1),(1,1)]:
    B=bestmatch(pl,fb)
    d=np.diag(B).copy()   # includes the trivial self-match = 1 (same slice)
    np.fill_diagonal(B,np.nan)
    iu=np.triu_indices(N,1); v=B[iu]; v=v[~np.isnan(v)]
    print((pl,fb),"cross-study best-slice corr quantiles",np.round(np.quantile(v,[.5,.9,.99,.999,.9999,1]),4), "n pairs",len(v))
    np.save(f"best_{pl}{fb}.npy",B)
