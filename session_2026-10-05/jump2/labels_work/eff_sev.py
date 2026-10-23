# A-priori multilingual effusion-severity lexicon (written before scoring; not tuned on gold58).
import pandas as pd, numpy as np, re
from sklearn.metrics import roc_auc_score
ID="StudyInstanceUID"; RT="/home/training/rsna_train"
tr=pd.read_csv(f"{RT}/train.csv").set_index(ID)
EFF=r"(effusion|derrame|erguss|izljev|izljeva|sıvı|sivi|vocht|hydrops|συλλογ\S* υγρού|υγρού|излив|épanchement|epanchement)"
MILD=r"(small|mild|minimal|minor|trace|tiny|some|slight|little|leve|pequeñ|mínim|minim|escas|gering|klein|wenig|discret|manj|minimaln|malo|blag|hafif|az miktar|minimaal|beperkt|licht|μικρ|ήπι|ελάχιστ|малък|лек|минимал|faible|petit)"
BIG=r"(moderate|large|significant|marked|abundant|extensive|massive|moderad|abundant|important|mäßig|maessig|deutlich|ausgepräg|umjeren|obiln|značaj|veća|belirgin|yaygın|orta |fazla|matig|uitgebreid|fors|μέτρι|ικαν|σημαντικ|άφθον|умерен|голям|значител|modéré|abondant)"
def sev(rep, pk):
    t=str(rep).lower()
    hits=[m.start() for m in re.finditer(EFF,t)]
    if not hits: return 1 if pk!="NO" else 0   # silent
    if pk=="NO": return 0
    mild=big=False
    for h in hits:
        w=t[max(0,h-60):h+60]
        if re.search(BIG,w): big=True
        if re.search(MILD,w): mild=True
    return 4 if big else (2 if mild else 3)
pk=pd.read_csv(f"{RT}/report_labels_v2.csv").set_index(ID)
v4=pd.read_csv(f"{RT}/llm_labels_v4_blend.csv").set_index(ID)
T=pd.read_csv("labels_in/T_raptor_proxy.csv").set_index(ID)
wd=pd.read_csv(f"{RT}/teacher_train_probs.csv").set_index(ID)
ids=v4.index
s=pd.Series({u:sev(tr.at[u,"Report"], pk["Effusion__verdict"].get(u,"UNK")) for u in ids})
gold=tr[tr["Effusion"].notna()]
g=gold.index
y=gold["Effusion"].values
print("gold58 Effusion AUC: v4 %.3f | sev-ordinal %.3f | v4 within sev (v4+sev) %.3f | teacher %.3f"%(
 roc_auc_score(y,v4.loc[g,"Effusion"]), roc_auc_score(y,s[g]), roc_auc_score(y,s[g]+v4.loc[g,"Effusion"]), roc_auc_score(y,T.loc[g,"Effusion"])))
print("gold58 by sev class: n / gold-pos rate"); print(pd.DataFrame({"s":s[g].values,"y":y}).groupby("s").y.agg(["size","mean"]).round(2).to_string())
ng=ids.difference(g)
print("non-gold 4349: mean teacher P(Effusion) by sev class (raptor proxy / widedense) and n:")
print(pd.DataFrame({"s":s[ng].values,"T":T.loc[ng,"Effusion"].values,"wd":wd.loc[ng,"Effusion"].values,"v4":v4.loc[ng,"Effusion"].values}).groupby("s").agg(n=("T","size"),T=("T","mean"),wd=("wd","mean"),v4=("v4","mean")).round(3).to_string())
