import numpy as np, pandas as pd, sys, io
from sklearn.metrics import roc_auc_score
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
R="C:/Users/eeshs/Downloads/rsna-knee"
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
idx=pd.read_csv(R+"/_crop_shards/study_index.csv"); uids=idx.StudyInstanceUID.values; N=len(uids)
S=np.load("csls_score.npy"); S=np.nan_to_num(S,nan=-99); np.fill_diagonal(S,-99)
oof=pd.read_csv(R+"/session_2026-10-18/next/g58/R7OOF.csv").set_index("StudyInstanceUID").reindex(uids)
P=oof[["p|"+l for l in LAB]].values
rep=pd.read_csv(R+"/_labels/rsna-knee-llm-labels/report_labels_v2.csv").set_index("StudyInstanceUID").reindex(uids)
Y=(rep[LAB].values>0.5).astype(float)
g=pd.read_csv(R+"/session_2026-10-18/next/g58/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].set_index("StudyInstanceUID")
gmask=np.isin(uids,g.index); G=g.reindex(uids)[LAB].values
lang=pd.read_csv(R+"/_data/_reports_lang.csv").set_index("StudyInstanceUID").lang.reindex(uids).values
pct=lambda v: pd.Series(v).rank(pct=True).values
Pp=np.column_stack([pct(P[:,j]) for j in range(12)])
def macro(pred,Yt,mask):
    a=[]
    for j in range(12):
        m=mask&~np.isnan(Yt[:,j])
        y=Yt[m,j]
        if y.min()==y.max(): continue
        a.append(roc_auc_score(y,pred[m,j]))
    return np.mean(a)
nong=~gmask & ~np.isnan(P).any(1)
base_oof=macro(Pp,Y,nong); base_g=macro(Pp,G,gmask)
print("base R7OOF: report-OOF macro %.5f  gold58 %.5f"%(base_oof,base_g))
for k in [5,20]:
    nb=np.argsort(-S,axis=1)[:,:k]
    NP=Pp[nb].mean(1)          # neighbours' predictions (test-time transductive smoothing)
    NY=np.nan_to_num(Y)[nb].mean(1)   # neighbours' train labels (kNN label transfer)
    for name,F in [("nbrPred",NP),("nbrLabel",NY)]:
        Fp=np.column_stack([pct(F[:,j]) for j in range(12)])
        solo_o=macro(Fp,Y,nong); solo_g=macro(Fp,G,gmask)
        line=f"k={k} {name}: solo report {solo_o:.4f} gold {solo_g:.4f} |"
        for a in [.05,.1,.2]:
            C=Pp+a*Fp
            line+=f" a={a}: dOOF {macro(C,Y,nong)-base_oof:+.5f} dG {macro(C,G,gmask)-base_g:+.5f};"
        print(line)
# same-language prior for reference
L=pd.Series(lang)
LP=np.column_stack([L.map(pd.Series(Pp[:,j]).groupby(L).mean()).values for j in range(12)])
LPp=np.column_stack([pct(LP[:,j]) for j in range(12)])
for a in [.05,.1,.2]:
    C=Pp+a*LPp; print(f"lang-mean-pred prior a={a}: dOOF {macro(C,Y,nong)-base_oof:+.5f} dG {macro(C,G,gmask)-base_g:+.5f}")
# label agreement of top image pairs vs same-lang random pairs
iu=np.triu_indices(N,1); v=S[iu]; o=np.argsort(-v)[:300]
a_,b_=iu[0][o],iu[1][o]
rng=np.random.default_rng(1); ra=rng.integers(0,N,20000); rb=rng.integers(0,N,20000); keep=(lang[ra]==lang[rb])&(ra!=rb); ra,rb=ra[keep][:3000],rb[keep][:3000]
Yn=np.nan_to_num(Y)
for j,l in enumerate(LAB):
    print(f"{l:17s} agree top300 {np.mean(Yn[a_,j]==Yn[b_,j]):.3f} same-lang random {np.mean(Yn[ra,j]==Yn[rb,j]):.3f}")
