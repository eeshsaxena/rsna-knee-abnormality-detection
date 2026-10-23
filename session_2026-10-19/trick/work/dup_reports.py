import pandas as pd, numpy as np, re, sys, io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
D="C:/Users/eeshs/Downloads/rsna-knee/_data"
t=pd.read_csv(D+"/train.csv")
lang=pd.read_csv(D+"/_reports_lang.csv",usecols=["StudyInstanceUID","lang"]).set_index("StudyInstanceUID")["lang"]
t["lang"]=t.StudyInstanceUID.map(lang)
print("n studies",len(t), "unique UIDs", t.StudyInstanceUID.nunique())
r=t.Report.fillna("")
print("exact dup reports: rows in a dup group", r.duplicated(keep=False).sum(), "groups", r[r.duplicated(keep=False)].nunique())
norm=r.str.lower().str.replace(r"\[date\]|\d+","",regex=True).str.replace(r"[^\w]+"," ",regex=True).str.strip()
dg=norm[norm.duplicated(keep=False)]
print("normalized dup reports: rows", len(dg), "groups", dg.nunique())
vc=dg.value_counts()
for txt,c in vc.head(15).items():
    print(c, "|", txt[:200])
print("report length quantiles", r.str.len().quantile([.01,.05,.25,.5,.75,.95]).to_dict())
print(t.lang.value_counts().head(12).to_dict())
