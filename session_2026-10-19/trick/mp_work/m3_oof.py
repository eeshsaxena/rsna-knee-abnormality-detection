import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
B='C:/Users/eeshs/Downloads/rsna-knee/'
LAB=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
o=pd.read_csv(B+'session_2026-10-18/next/g58/R7OOF.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
fl=pd.read_csv('train_series_flags.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
rl=pd.read_csv(B+'_labels/rsna-knee-llm-labels/report_labels_v2.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
print('oof rows',len(o),'cols',len(o.columns)); print(o['fluid|status'].value_counts().head(8))
print(o[['r7_fold','precision']].describe().T)
J=o.join(fl,how='inner').join(rl,how='inner',rsuffix='_rl')
print('joined',len(J))
print('status x no_fs_sag'); print(pd.crosstab(J['fluid|status'].str[:30],J.no_fs_sag))
def auc(y,p):
    m=~np.isnan(y)
    y=y[m];p=p[m]
    if y.sum()<5 or (1-y).sum()<5: return np.nan
    return roc_auc_score(y,p)
def ylab(c,mode):
    v=J[c+'__verdict']
    if mode=='yn': return np.where(v=='YES',1.0,np.where(v=='NO',0.0,np.nan))
    return np.where(v=='YES',1.0,0.0)
groups={'all':np.ones(len(J),bool),'complete':~(J.no_fs_sag|J.no_fs_cor|J.no_nfs_sag).values,
        'no_fs_sag':J.no_fs_sag.values,'no_fs_cor':J.no_fs_cor.values,'no_nfs_sag':J.no_nfs_sag.values,
        'dup_fs_sag':J.dup_fs_sag.values,'n>=6':J['n>=6'].values,'de':(J.lang=='de').values,'de&no_fs_sag':((J.lang=='de')&J.no_fs_sag).values,
        'el':(J.lang=='el').values,'el&no_nfs_sag':((J.lang=='el')&J.no_nfs_sag).values,'en&no_fs_cor':((J.lang=='en')&J.no_fs_cor).values,'en':(J.lang=='en').values}
for mode in ['yn','yes_vs_rest']:
    rows=[]
    for g,m in groups.items():
        r={'group':g,'n':int(m.sum())}
        for c in LAB:
            y=ylab(c,mode)[m]; p=J['p|'+c].values[m]
            r[c]=auc(y,p)
        r['macro']=np.nanmean([r[c] for c in LAB])
        rows.append(r)
    T=pd.DataFrame(rows).set_index('group')
    print('\n=== R7 OOF per-label AUC vs report labels, mode',mode); print(T.round(3).to_string())
    T.to_csv(f'r7oof_auc_by_group_{mode}.csv')
J.to_pickle('J.pkl')
