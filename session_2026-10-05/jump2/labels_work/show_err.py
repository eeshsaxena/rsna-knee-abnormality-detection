import pandas as pd, numpy as np, sys
ID="StudyInstanceUID"
E=pd.read_csv("v4_gold58_errors.csv",dtype={"uid6":str})
tr=pd.read_csv("/home/training/rsna_train/train.csv"); tr["uid6"]=tr[ID].str[-6:]
order=E.groupby("uid6").size().sort_values(ascending=False, kind="stable")
a,b=int(sys.argv[1]),int(sys.argv[2])
for u in order.index[a:b]:
    e=E[E.uid6==u]
    print("\n#####",u,e.lang.iloc[0],"| ERR:", "; ".join(f"{r.label} g{r.gold} v4={r.v4} pk={r.pilk} T={r.teacherP}" for r in e.itertuples()))
    print(str(tr.loc[tr.uid6==u,"Report"].iloc[0])[:int(sys.argv[3]) if len(sys.argv)>3 else 1500])
