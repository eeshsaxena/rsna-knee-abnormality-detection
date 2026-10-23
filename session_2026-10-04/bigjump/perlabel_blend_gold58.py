# CPU-only: per-label profile of the v4 crop ensemble vs the Raptor proxy (parent stand-in) on gold58.
import numpy as np, pandas as pd, sys, io
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
S="C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
g=pd.read_csv(S+"/kneelabels/jev/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].set_index("StudyInstanceUID")[LAB]
z=np.load(S+"/laneA/gold74_preds_laneA.npz",allow_pickle=True); ids=list(z["ids"])
def df(k): return pd.DataFrame(z[k],index=ids,columns=LAB).reindex(g.index)
rk=lambda d: d.rank(pct=True)
crop=(rk(df("finetune_cropaug_f1_radimagenet"))+rk(df("finetune_cropaug_f2_radimagenet"))+rk(df("finetune_crop_f1_radimagenet")))/3
P=pd.read_csv(S+"/teacher_v2_out/raptor_proxy_v5v8.csv").set_index("StudyInstanceUID")[LAB].reindex(g.index)
P=rk(P); C=rk(crop)
print(f"{'label':16s} proxy  crop   rhoW(pos/neg)  blend.25  d")
tot=[]
for c in LAB:
    y=g[c].values; ap=roc_auc_score(y,P[c]); ac=roc_auc_score(y,C[c]); b=roc_auc_score(y,0.75*P[c]+0.25*C[c])
    r1=spearmanr(P[c][y==1],C[c][y==1])[0]; r0=spearmanr(P[c][y==0],C[c][y==0])[0]
    tot.append(b-ap); print(f"{c:16s} {ap:.3f}  {ac:.3f}  {r1:+.2f}/{r0:+.2f}      {b:.3f}  {b-ap:+.3f}")
print("macro d", np.mean(tot).round(4))
