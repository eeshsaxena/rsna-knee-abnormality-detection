import numpy as np, pandas as pd, sys, io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
idx=pd.read_csv("C:/Users/eeshs/Downloads/rsna-knee/_crop_shards/study_index.csv")
N=4407
for f in ["best_01.npy","best_11.npy"]:
    B=np.load(f); iu=np.triu_indices(N,1); v=B[iu]
    for th in [.95,.97,.98,.99,.995,.999]:
        print(f, th, int(np.nansum(v>=th)))
B1=np.load("best_01.npy"); B2=np.load("best_11.npy")
M=np.fmax(np.nan_to_num(B1,nan=-1),np.nan_to_num(B2,nan=-1))
iu=np.triu_indices(N,1); v=M[iu]
o=np.argsort(-v)[:80]
rows=[(iu[0][k],iu[1][k],v[k],B1[iu[0][k],iu[1][k]],B2[iu[0][k],iu[1][k]]) for k in o]
for r in rows: print(r[0],r[1],np.round(r[2:],4))
