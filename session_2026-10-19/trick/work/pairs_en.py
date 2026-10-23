import pandas as pd, numpy as np, sys, io, re
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
D="C:/Users/eeshs/Downloads/rsna-knee/_data"
t=pd.read_csv(D+"/train.csv")
r=t.Report.fillna("")
tmpl=r.str.contains("Exam Type:",regex=False)
print("Exam Type template reports:", tmpl.sum())
def field(s,name,nxt):
    m=re.search(name+r"\s*:?\s*(.*?)\s*(?:"+nxt+r")",s,flags=re.S|re.I)
    return m.group(1).strip() if m else None
sub=t[tmpl].copy()
sub["exam"]=sub.Report.apply(lambda s: field(s,"Exam Type","Exam Date"))
sub["ind"]=sub.Report.apply(lambda s: field(s,"Indication","Comparison"))
sub["cmp"]=sub.Report.apply(lambda s: field(s,"Comparison","TECHNIQUE|Technique|FINDINGS"))
sub["side"]=sub.exam.fillna("").str.extract(r"(LEFT|RIGHT|BILAT)",expand=False)
print(sub.side.value_counts(dropna=False).to_dict())
print("comparison field top values:"); print(sub.cmp.fillna("NA").str[:60].value_counts().head(15).to_string())
mri_prior=sub.cmp.fillna("").str.contains(r"MR|MRI",regex=True)
print("comparison mentions MR/MRI:", mri_prior.sum())
for c in sub[mri_prior].cmp.head(15): print("  ", c[:150])
# identical non-generic indications
ind=sub.ind.fillna("").str.lower().str.replace(r"\[date\]|\[time\]","",regex=True).str.strip()
generic=ind.str.len()<25
g=sub[~generic].assign(indn=ind[~generic]).groupby("indn")
multi=[(k,v) for k,v in g if len(v)>1]
print("non-generic indications shared by >1 study:", len(multi), "studies", sum(len(v) for k,v in multi))
for k,v in multi[:25]:
    print(len(v), list(v.side), "|", k[:120])
sub[["StudyInstanceUID","exam","side","ind","cmp"]].to_csv("en_template_fields.csv",index=False)
