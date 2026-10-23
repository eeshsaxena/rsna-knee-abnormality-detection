import pandas as pd, numpy as np, glob, os
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
H=os.path.expanduser("~")
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
g=pd.read_csv(H+"/rsna_train/gold74_eval_set.csv").set_index("StudyInstanceUID")
r7=pd.read_csv(H+"/r7t_labels/real/R7OOF.csv").set_index("StudyInstanceUID")
R7=r7.loc[g.index,["p|"+l for l in L]].values
def fam(tag):
    fs=sorted(glob.glob(H+"/r7t_labels/gold_preds/f*_"+tag+"__raw__A.csv"))
    a=[pd.read_csv(f).set_index("StudyInstanceUID").loc[g.index,L].values for f in fs]
    return np.mean(a,0), len(fs)
def auc(P, idx=None):
    idx = np.arange(len(g)) if idx is None else idx
    out=[]
    for j,l in enumerate(L):
        y=g[l].values[idx]; m=~np.isnan(y)
        if m.sum()>0 and len(set(y[m]))==2: out.append(roc_auc_score(y[m],P[idx][m,j]))
    return np.mean(out), len(out)
def rk(P): return np.column_stack([rankdata(P[:,j])/len(P) for j in range(P.shape[1])])
g58=np.where(g.source.values=="original_gold58")[0]
for name,P in [("R7",R7)]+[(t,fam(t)[0]) for t in ("s1047_ttr_r7tb","s1044_ttr_frac75")]:
    a74=auc(P); a58=auc(P,g58)
    print(name, "gold74 %.4f (%d) gold58 %.4f (%d)"%(a74[0],a74[1],a58[0],a58[1]))
for t in ("s1047_ttr_r7tb","s1044_ttr_frac75"):
    P=fam(t)[0]
    for w in (.2,.3):
        B=(1-w)*rk(R7)+w*rk(P)
        print("R7 + %.1f %s: gold74 %+.4f gold58 %+.4f"%(w,t,auc(B)[0]-auc(R7)[0],auc(B,g58)[0]-auc(R7,g58)[0]))
