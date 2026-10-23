import pandas as pd, numpy as np, sys, io, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
D="C:/Users/eeshs/Downloads/rsna-knee/_data"
t=pd.read_csv(D+"/train.csv")
lang=pd.read_csv(D+"/_reports_lang.csv",usecols=["StudyInstanceUID","lang"]).set_index("StudyInstanceUID")["lang"]
t["lang"]=t.StudyInstanceUID.map(lang)
r=t.Report.fillna("").str.lower().str.replace(r"\[date\]","",regex=True)
L=r.str.len().values
v=TfidfVectorizer(analyzer="char_wb",ngram_range=(4,5),min_df=2,sublinear_tf=True)
X=v.fit_transform(r)
nn=NearestNeighbors(n_neighbors=2,metric="cosine").fit(X)
d,i=nn.kneighbors(X)
sim=1-d[:,1]; j=i[:,1]
t["nn_sim"]=sim; t["nn"]=j
long=L>=500
for th in [.8,.9,.95,.98]:
    print(f"long reports (>=500 chars, n={long.sum()}): nn sim >= {th}: {(sim[long]>=th).sum()}")
# show examples of high-sim long non-identical
idx=np.where(long & (sim>=0.9) & (r.values!=r.values[j]))[0]
print("long, sim>=.9, not identical:", len(idx))
seen=set()
for a in idx[:12]:
    b=j[a]
    if (b,a) in seen: continue
    seen.add((a,b))
    print("----",round(sim[a],3), t.lang[a], L[a], L[b])
    print("A:", t.Report[a][:600].replace("\n"," "))
    print("B:", t.Report[b][:600].replace("\n"," "))
# also: mention of prior exam
prior=r.str.contains(r"compar|prior|previous|vorunter|voraufnahme|vergel|vorig|previo|anterior estudio|estudio previo|önceki|precedente|préc|prethodn|предишн|προηγ|vergelijk",regex=True)
print("reports mentioning prior/comparison:", prior.sum(), "of", len(r))
print(t[prior].lang.value_counts().to_dict())
t[["StudyInstanceUID","lang","nn_sim"]].assign(nn_uid=t.StudyInstanceUID.values[j], L=L).to_csv("report_nn.csv",index=False)
