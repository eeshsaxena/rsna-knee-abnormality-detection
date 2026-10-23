import numpy as np, pickle, sys, io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
res=pickle.load(open("fp32.pkl","rb")); N=4407
def zr(a):
    a=a.reshape(len(a),-1).astype(np.float32); a=a-a.mean(1,keepdims=True); n=np.linalg.norm(a,axis=1,keepdims=True); return a/np.maximum(n,1e-6)
from collections import Counter
pairs=Counter()
for pl,fb in [(0,1),(1,1),(2,1),(0,0),(1,0),(2,0)]:
    rows=[];own=[]
    for si in range(N):
        k=(si,pl,fb)
        if k in res:
            a=res[k][1]; sd=a.reshape(len(a),-1).std(1); a=a[sd>10]
            if len(a): rows.append(zr(a)); own+=[si]*len(a)
    X=np.concatenate(rows); own=np.array(own); cnt=0
    for s in range(0,len(X),4096):
        G=X[s:s+4096]@X.T
        r,c=np.nonzero(G>=0.999)
        for a_,b_ in zip(own[s+r],own[c]):
            if a_<b_: pairs[(int(a_),int(b_))]+=1; cnt+=1
    print((pl,fb),"slices",len(X),"cross-study slice pairs with corr>=.999:",cnt)
print("study pairs with any identical slice:",len(pairs)); print(pairs.most_common(10))
