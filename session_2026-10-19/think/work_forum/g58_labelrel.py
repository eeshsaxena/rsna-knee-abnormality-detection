# gold58 label-level relations (Effusion vs Synovitis etc.) + R6 serve prediction cross-AUCs. CPU, read-only inputs.
import pandas as pd, numpy as np, json, sys
from sklearn.metrics import roc_auc_score
D='C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58/'
g=pd.read_csv(D+'gold74_eval_set.csv'); g=g[g.source=='original_gold58'].reset_index(drop=True)
L=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
out={'n':len(g)}
y=g[L].astype(int)
ct=pd.crosstab(y['Effusion'],y['Synovitis']); out['eff_x_syn']=ct.to_dict()
out['P(syn|eff=1)']=float(y.Synovitis[y.Effusion==1].mean()); out['P(syn|eff=0)']=float(y.Synovitis[y.Effusion==0].mean())
out['phi_eff_syn']=float(np.corrcoef(y.Effusion,y.Synovitis)[0,1])
# label-label pairwise phi top pairs
C=np.corrcoef(y.values.T); pairs=[]
for i in range(12):
    for j in range(i+1,12): pairs.append((round(float(C[i,j]),3),L[i],L[j]))
out['top_label_phi']=sorted(pairs,reverse=True)[:8]
r6=pd.read_csv(D+'R6_gold58_serve.csv')
m=g[['StudyInstanceUID']].merge(r6,on='StudyInstanceUID',how='left')
print('R6 cols',r6.columns.tolist()[:14], 'missing', int(m[L[0]].isna().sum()))
cross={}
for t in ['Synovitis','Effusion','Fracture','Contusion','Lateral OA','PF OA']:
    cross[t]={p:round(float(roc_auc_score(y[t],m[p])),3) for p in L}
out['R6_pred_col_AUC_vs_label']=cross
json.dump(out,open(sys.argv[1],'w'),indent=1)
print(json.dumps({k:v for k,v in out.items() if k!='R6_pred_col_AUC_vs_label'},indent=0))
for t,v in cross.items(): print(t, sorted(v.items(), key=lambda kv:-kv[1])[:4])
