import numpy as np, pandas as pd
from scipy.stats import rankdata
G='C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
g=pd.read_csv(G+"/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].reset_index(drop=True); ids=g.StudyInstanceUID.astype(str).tolist(); Y=g[L].values.astype(int)
M={'R6':pd.read_csv(G+"/R6_gold58_serve.csv").set_index("StudyInstanceUID").loc[ids,L].values,
   'S6fall':pd.read_csv(G+"/n42gpu_gold58.csv").set_index("StudyInstanceUID").loc[ids,L].values,
   'R7oof':pd.read_csv(G+"/R7OOF.csv").set_index("StudyInstanceUID").loc[ids,["p|"+l for l in L]].values,
   'r384':np.load(G+"/gold58_r384_kernelserve_cpu.npy"),'r224':np.load(G+"/gold58_r224_kernelserve.npy")}
def auc(s,y):
    p=y.sum(); q=len(y)-p; r=rankdata(s); return (r[y==1].sum()-p*(p+1)/2)/(p*q)
print('gold58 own / cross AUC (cross = this column scored against the other compartment gold); disc = discordant-pair ordering')
for k,X in M.items():
    row=[k]
    for a,b in [(2,3),(4,5)]:
        row.append('%s own %.3f cross %.3f | %s own %.3f cross %.3f' % (L[a][:6]+L[a].split()[1][:3],auc(X[:,a],Y[:,a]),auc(X[:,a],Y[:,b]),L[b][:6]+L[b].split()[1][:3],auc(X[:,b],Y[:,b]),auc(X[:,b],Y[:,a])))
        sel=Y[:,a]!=Y[:,b]; s=np.where(Y[sel,a]==1,1,-1)
        pa=rankdata(X[:,a])/58; pb=rankdata(X[:,b])/58
        row.append('disc n %d ok %.2f' % (sel.sum(), (np.sign(pa[sel]-pb[sel])==s).mean()))
    print(' || '.join(row))
