# lab2_effsize.py: does a SIZE-graded effusion read rank gold58 better than steven_v4? (CPU, read-only)
# Ordering fixed a priori: negated 0 < silent 1 < small 2 < unqualified 3 < moderate/large 4; v4 as within-level tiebreak.
# CAVEAT: the qualifier lists were written after reading ~18 gold58 reports -> in-sample, optimistic.
import re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ID="StudyInstanceUID"; RT="/home/training/rsna_train"
tr=pd.read_csv(f"{RT}/train.csv"); L=["Effusion","Synovitis"]
gd=tr[tr["ACL"].notna()].set_index(ID)
v4=pd.read_csv(f"{RT}/llm_labels_v4_blend.csv").set_index(ID).loc[gd.index]
mix=pd.read_csv(f"{RT}/labels_mix50_logit.csv").set_index(ID).loc[gd.index]
EFF=r"effusion|derrame|erguss|izljev|sıvı|συλλογ|излив|épanchement|epanchement|gewrichtsvocht|hydrops|hemarthros|hemartro|haemarthros|αιμάρθρ|joint fluid|intraarticular fluid|liquide articulaire|eklem içi sıvı"
SMALL=r"small|mild|minimal|minor|trace|slight|\bsome\b|little|physiolog|leve\b|peque|mínim|escaso|gering|klein|wenig|manj|\bmal[aio]\b|blag|hafif|\baz\b|μικρ|ελάχιστ|ήπι|незначит|малко|минимал|léger|leger|minime|faible|\bpeu\b|minimaal|beperkt|discreet|discret"
BIG=r"moderate|large|marked|significant|extensive|abundant|moderad|importante|mäßig|massig|deutlich|ausgepr|reichlich|umjeren|velik|obiln|izražen|\borta\b|belirgin|fazla|yoğun|μέτρια|ικαν|σημαντικ|άφθον|ευμεγέθ|умерен|значит|голям|обилен|modéré|modere|abondant|matig|uitgesproken|ruim|fors|groot|massive|tense|prominent"
NEG=r"\bno\b|without|\bnot\b|\bsin\b|no hay|\bkein|\bohne\b|\bnema\b|\bbez\b|\byok|izlenmedi|saptanmad|δεν|χωρίς|няма|\bбез\b|pas d|absence|\bgeen\b|zonder|\bnone\b|negative"
def code(rep):
    t=str(rep).replace("µ","μ").lower(); best=1
    for s in re.split(r"[.\n;]", t):
        if not re.search(EFF, s): continue
        if re.search(NEG, s) and not re.search(SMALL+"|"+BIG, s): c=0
        elif re.search(BIG, s): c=4
        elif re.search(SMALL, s): c=2
        else: c=3
        best=max(best, c) if best!=1 else c
    return best
c=gd.Report.map(code)
y=gd.Effusion.astype(int)
print("effusion size code x gold (n, gold+ rate, v4 mean):")
print(pd.DataFrame({"code":c,"y":y,"v4":v4.Effusion}).groupby("code").agg(n=("y","size"),gold=("y","mean"),v4=("v4","mean")).round(3).to_string())
for nm,s in [("v4",v4.Effusion),("code only",c.astype(float)),("code + v4 tiebreak",c+0.01*v4.Effusion.rank(pct=True)),("mix50",mix.Effusion)]:
    print(f"  Effusion AUC {nm:20s} {roc_auc_score(y,s):.4f}")
