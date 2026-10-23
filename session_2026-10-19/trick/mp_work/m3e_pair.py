import pandas as pd, numpy as np, glob, json
from sklearn.metrics import roc_auc_score
B='C:/Users/eeshs/Downloads/rsna-knee/'
LAB=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
fs=sorted(glob.glob(B+'session_2026-10-0*/kaggle_frac75*/out_*/f*_s1044_ttr_frac75/oof_f*_s1044_ema.csv'))
print([f.split('/')[-1] for f in fs])
T=pd.concat([pd.read_csv(f,dtype={'StudyInstanceUID':str}) for f in fs]).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
J=pd.read_pickle('J.pkl'); F=pd.read_csv('train_series_feats.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID'); J['nser']=F['n'].reindex(J.index)
J=J.join(T.add_prefix('t|'),how='inner'); print('paired studies',len(J))
lab_src={'v4blend':B+'_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv','jev_bin':B+'session_2026-10-05/jump2/labels_in/jev_hybrid_binary.csv'}
Y={k:(pd.read_csv(v,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').reindex(J.index)[LAB]>0.5).astype(float) for k,v in lab_src.items()}
pct=lambda x: pd.Series(x).rank(pct=True).values
groups={'complete':~(J.no_fs_sag|J.no_fs_cor|J.no_nfs_sag).values,'no_fs_sag':J.no_fs_sag.values,'no_fs_cor':J.no_fs_cor.values,'no_nfs_sag':J.no_nfs_sag.values,'dup_fs_sag':J.dup_fs_sag.values,'dup_sag':J.dup_sag.values,'n>=7':(J.nser>=7).values}
out=[]
for ln,y in Y.items():
    for c in LAB:
        yy=y[c].values; rt=pct(J['t|'+c].values); ro=pct(J['p|'+c].values); bl=0.5*rt+0.5*ro
        for g,m in groups.items():
            if yy[m].sum()<8 or (1-yy[m]).sum()<8: continue
            out.append(dict(labels=ln,label=c,group=g,n=int(m.sum()),auc_ttr=roc_auc_score(yy[m],rt[m]),auc_r7=roc_auc_score(yy[m],ro[m]),auc_blend=roc_auc_score(yy[m],bl[m])))
O=pd.DataFrame(out); O['r7_minus_ttr']=O.auc_r7-O.auc_ttr
P=O.groupby(['labels','group'])[['n','auc_ttr','auc_r7','auc_blend','r7_minus_ttr']].mean()
print(P.round(4).to_string())
O.to_csv('pair_ttr_r7_by_group.csv',index=False)
J.to_pickle('JT.pkl')
