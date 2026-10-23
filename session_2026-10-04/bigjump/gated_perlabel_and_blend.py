# CPU-only: per-label view of report-gated fill (gold58, target level) + binormal blend value of candidate students.
import pandas as pd, numpy as np, sys, io
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
from scipy.stats import norm
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
S="C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
g=pd.read_csv(S+"/kneelabels/jev/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].set_index("StudyInstanceUID")[LAB]
v4=pd.read_csv(S+"/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")[LAB]
pk=pd.read_csv(S+"/kneelabels/rsna-knee-llm-labels/report_labels_v2.csv").set_index("StudyInstanceUID")
prox=pd.read_csv(S+"/teacher_v2_out/raptor_proxy_v5v8.csv").set_index("StudyInstanceUID")[LAB]
ids=v4.index.intersection(prox.index)
pkr=pk.reindex(ids)
sil=pd.DataFrame({c:(pkr[c+"__verdict"]=="UNK").values for c in LAB},index=ids)
V=v4.loc[ids]; R=prox.loc[ids].copy()
for c in LAB:  # logit mean/std match of teacher onto v4 scale, fitted on ALL train rows (gold58 included only as rows, no labels used)
    a=logit(np.clip(R[c].values,1e-4,1-1e-4)); b=logit(np.clip(V[c].values,1e-4,1-1e-4)); R[c]=expit((a-a.mean())/a.std()*b.std()+b.mean())
print("train-wide silent share:", {c[:8]:round(sil[c].mean(),2) for c in LAB})
G=g.index
def tab(ws,wa): W=np.where(sil.loc[G],ws,wa); return pd.DataFrame(V.loc[G].values*(1-W)+R.loc[G].values*W,index=G,columns=LAB)
base=[roc_auc_score(g[c],V.loc[G,c]) for c in LAB]
for nm,(ws,wa) in {"gated .75/0":(0.75,0),"gated 1.0/0":(1.0,0),"mixSA .75/.5":(0.75,0.5),"flat .5":(0.5,0.5)}.items():
    t=tab(ws,wa); a=[roc_auc_score(g[c],t[c]) for c in LAB]
    print(f"{nm:13s} macro {np.mean(a):.4f} (d {np.mean(a)-np.mean(base):+.4f}) | "+" ".join(f"{c[:5]}{x-y:+.3f}" for c,x,y in zip(LAB,a,base)))
# Binormal blend value (MODEL), parent 0.943 flat per label; student solo S, within-class rho r
def d(a): return np.sqrt(2)*norm.ppf(a)
def best(S,r,ap=0.943):
    out=[]
    for w in np.linspace(0,0.8,161):
        out.append(norm.cdf(((1-w)*d(ap)+w*d(S))/np.sqrt((1-w)**2+w**2+2*w*(1-w)*r)/np.sqrt(2)))
    i=int(np.argmax(out)); return out[i]-ap, np.linspace(0,0.8,161)[i]
print("\nMODEL: blend lift over 0.943 for candidate students (S0 = pure-v4 fleet solo, rho0 = .80)")
for S0 in [0.920,0.930]:
    for nm,dS,r in [("pure v4",0,0.80),("flat mix50 (Raptor-family)",0.010,0.86),("gated .75/0 (report-silent only)",0.005,0.81),("gated + reported .25",0.009,0.845)]:
        L,w=best(S0+dS,r); print(f"  S0 {S0:.3f} | {nm:34s} solo {S0+dS:.3f} rho {r:.3f} -> +{L:.4f} @W {w:.2f}")
