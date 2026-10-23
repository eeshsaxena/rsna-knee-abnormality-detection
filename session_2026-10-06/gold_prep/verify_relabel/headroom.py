import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
DS='C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train'
rd=lambda p: pd.read_csv(p,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
g=rd(f'{DS}/gold74_eval_set.csv'); g58=g[g.source=='original_gold58']
m=rd(f'{DS}/labels_mix50_logit.csv'); v=rd(f'{DS}/llm_labels_v4_blend.csv'); f=rd(f'{DS}/folds5.csv')
print('gold rows', len(g), 'g58', len(g58), 'sources', g.source.value_counts().to_dict())
print('mix50 cols', list(m.columns)[:15], 'n', len(m), 'range', float(m.select_dtypes('number').min().min()), float(m.select_dtypes('number').max().max()))
print('v4 cols', list(v.columns)[:15], 'n', len(v))
print('folds cols', list(f.columns), 'n', len(f), 'gold74 in folds', int(g.index.isin(f.index).sum()), 'g58 in mix50', int(g58.index.isin(m.index).sum()))
T=['Effusion','Synovitis','Fracture','Contusion','PF OA','Lateral OA']
T12=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
print('gold58 label values', sorted(set(np.unique(g58[T12].to_numpy()))))
for t in T12:
    y=(g58[t]>=.5).astype(int); 
    print(f'{t:<17} n+={y.sum():>2} mix50 {roc_auc_score(y,m.loc[g58.index,t]):.3f} v4 {roc_auc_score(y,v.loc[g58.index,t]):.3f}')
# paired bootstrap SE of AUC(mix50)-AUC(v4) per target, as a proxy for paired-diff SE
rng=np.random.default_rng(0); idx=g58.index
for t in T:
    y=(g58[t]>=.5).to_numpy().astype(int); a=m.loc[idx,t].to_numpy(); b=v.loc[idx,t].to_numpy()
    d=[]; sa=[]
    for _ in range(2000):
        i=rng.integers(0,len(y),len(y))
        if 0<y[i].mean()<1: d.append(roc_auc_score(y[i],a[i])-roc_auc_score(y[i],b[i])); sa.append(roc_auc_score(y[i],b[i]))
    r=np.corrcoef(a,b)[0,1]
    print(f'{t:<11} SE(AUC v4) {np.std(sa):.3f}  SE(paired diff mix50-v4) {np.std(d):.3f}  corr(scores) {r:.2f}')
