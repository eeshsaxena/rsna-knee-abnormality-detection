# CPU-only: are report-YES cells with mild/hedged wording the gold false positives? (privileged-info severity test)
import pandas as pd, numpy as np, re, sys, io
from sklearn.metrics import roc_auc_score
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
S="C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
D="C:/Users/eeshs/Downloads/rsna-knee/_data"
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
rep=pd.read_csv(D+"/_reports_lang.csv").set_index("StudyInstanceUID")
g=pd.read_csv(S+"/kneelabels/jev/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].set_index("StudyInstanceUID")[LAB]
v4=pd.read_csv(S+"/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")[LAB].reindex(g.index)
P=pd.read_csv(S+"/teacher_v2_out/raptor_proxy_v5v8.csv").set_index("StudyInstanceUID")[LAB].reindex(g.index)
txt=rep["Report"].reindex(g.index).fillna("").str.lower()
# finding terms (en/es/tr/hr/el/bg/nl/de) and mild/hedge words; crude, pre-written before looking at gold
TERMS={"Effusion":r"effusion|derrame|efüzyon|sıvı|izljev|συλλογ|излив|hydrops|vocht|erguss|épanchement",
 "Contusion":r"contusion|bruis|marrow edema|marrow oedema|bone edema|contusi|edema óseo|edema oseo|ödem|kontüzyon|edem|οίδημα|оток|kneuzing|beenmergoedeem|knochenmarködem|kontusion|œdème",
 "Medial OA":r"osteoarthr|arthros|artrosis|gonartro|degenerat|chondro|condro|kondro|cartilag|cartílag|kıkırdak|hrskav|χόνδρ|хрущ|kraakbeen|knorpel",
 "Lateral OA":r"osteoarthr|arthros|artrosis|gonartro|degenerat|chondro|condro|kondro|cartilag|cartílag|kıkırdak|hrskav|χόνδρ|хрущ|kraakbeen|knorpel",
 "PF OA":r"patell|rótul|rotul|femoropat|patelofem|patellofem|retropatell|чашк|επιγονατ",
 "Synovitis":r"synovi|sinovi|sinovit|συνοβ|синов",
 "Baker's":r"baker|popl",
 "Fracture":r"fractur|fractura|kırık|fraktur|prijelom|κάταγμ|фрактур|breuk"}
MILD=r"mild|minimal|small|slight|subtle|trace|tiny|minor|early|possibl|probabl|suspect|questionab|cannot be excluded|leve|pequeñ|mínim|minim|discret|liger|incipient|posibl|probabl|hafif|küçük|az miktar|erken|olası|şüphe|blag|mal[aio] |početn|moguć|ήπι|μικρ|ελάχιστ|πιθαν|лек|малк|минимал|начал|gering|klein|beginnend|mogelijk|leicht|fraglich|diskret|möglich|léger|petit|minime|débutant"
rows=[]
for c,term in TERMS.items():
    flag=[]
    for sid,t in txt.items():
        f=0
        for m in re.finditer(term,t):
            w=t[max(0,m.start()-60):m.end()+60]
            if re.search(MILD,w): f=1;break
        flag.append(f)
    flag=np.array(flag); yes=(v4[c].values>=0.5); y=g[c].values
    n_yes=yes.sum(); n_m=(yes&(flag==1)).sum()
    pr_m=y[yes&(flag==1)].mean() if n_m else np.nan; pr_n=y[yes&(flag==0)].mean() if (n_yes-n_m) else np.nan
    a0=roc_auc_score(y,v4[c]); a1=roc_auc_score(y,v4[c]-0.25*(yes&(flag==1)))
    pm=P[c].rank(pct=True).values
    print(f"{c:10s} YES {n_yes:2d} | mild-flag {n_m:2d} gold+ {pr_m:.2f} | no-flag {n_yes-n_m:2d} gold+ {pr_n:.2f} | v4 AUC {a0:.3f} -> mild-down {a1:.3f} | proxy rank in mild-YES {np.nanmean(pm[yes&(flag==1)]) if n_m else np.nan:.2f} vs plain-YES {np.nanmean(pm[yes&(flag==0)]):.2f}")
