# CPU-only TARGET-level test on gold58: fill report-silent cells with an image reader.
# Compares an OWN-family reader (v4 crop ensemble, gold58-selected => optimistic) vs the parent-family Raptor proxy,
# and reports the table's within-class Spearman with the Raptor proxy (decorrelation property).
import pandas as pd, numpy as np, sys, io
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from scipy.special import logit, expit
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
S="C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
g=pd.read_csv(S+"/kneelabels/jev/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].set_index("StudyInstanceUID")[LAB]
v4=pd.read_csv(S+"/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")[LAB].reindex(g.index)
pk=pd.read_csv(S+"/kneelabels/rsna-knee-llm-labels/report_labels_v2.csv").set_index("StudyInstanceUID").reindex(g.index)
sil=pd.DataFrame({c:(pk[c+"__verdict"]=="UNK").values for c in LAB},index=g.index)
z=np.load(S+"/laneA/gold74_preds_laneA.npz",allow_pickle=True); ids=list(z["ids"])
def df(k): return pd.DataFrame(z[k],index=ids,columns=LAB).reindex(g.index)
own=(df("finetune_cropaug_f1_radimagenet")+df("finetune_cropaug_f2_radimagenet")+df("finetune_crop_f1_radimagenet"))/3
prox=pd.read_csv(S+"/teacher_v2_out/raptor_proxy_v5v8.csv").set_index("StudyInstanceUID")[LAB].reindex(g.index)
# map a reader onto v4's scale per label by logit mean/std match (monotone, keeps reader order)
def mapto(r,ref):
    out=r.copy()
    for c in LAB:
        a=logit(np.clip(r[c].values,1e-4,1-1e-4)); b=logit(np.clip(ref[c].values,1e-4,1-1e-4))
        out[c]=expit((a-a.mean())/a.std()*b.std()+b.mean())
    return out
OWN=mapto(own,v4); PRX=mapto(prox,v4)
def macro(t): return np.mean([roc_auc_score(g[c],t[c]) for c in LAB])
def wrho(t):
    r=[]
    for c in LAB:
        y=g[c].values
        for k in (0,1):
            m=y==k
            if m.sum()>4: r.append(spearmanr(t[c][m],prox[c][m])[0])
    return np.nanmean(r)
rng=np.random.default_rng(0)
def boot(a,b,n=1000):
    ids=np.array(g.index); out=[]
    for _ in range(n):
        s=ids[rng.integers(0,len(ids),len(ids))]
        try: out.append(np.mean([roc_auc_score(g.loc[s,c],b.loc[s,c])-roc_auc_score(g.loc[s,c],a.loc[s,c]) for c in LAB]))
        except ValueError: pass
    out=np.array(out); return out.mean(),out.std(),(out>0).mean()
print("silent share on gold58 cells:", round(sil.values.mean(),3), "| per label:", {c[:6]:round(sil[c].mean(),2) for c in LAB})
rows=[("v4 (LLM only)",v4)]
for nm,R in [("OWN crop",OWN),("RAPTOR proxy",PRX)]:
    for ws,wa in [(0.5,0.5),(0.75,0.0),(0.75,0.25),(1.0,0.0)]:
        t=v4*(1-np.where(sil,ws,wa))+R*np.where(sil,ws,wa)
        rows.append((f"{nm}: silent w{ws} / reported w{wa}",pd.DataFrame(t,index=g.index,columns=LAB)))
base=rows[0][1]
for nm,t in rows:
    mu,sd,p=boot(base,t) if nm!="v4 (LLM only)" else (0,0,0)
    print(f"{nm:42s} gold58 {macro(t):.4f}  d {mu:+.4f} sd {sd:.4f} P>0 {p:.2f}  within-class rho vs proxy {wrho(t):.3f}")
