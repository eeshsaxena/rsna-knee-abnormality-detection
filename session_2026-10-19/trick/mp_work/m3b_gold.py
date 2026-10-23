import pandas as pd, numpy as np
B='C:/Users/eeshs/Downloads/rsna-knee/'
f=pd.read_csv('train_series_feats.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
fl=pd.read_csv('train_series_flags.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
g=pd.read_csv(B+'session_2026-10-18/next/g58/gold74_eval_set.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
print(g.source.value_counts())
G=g.join(fl)
print('gold74 lang', G.lang.value_counts().to_dict())
for c in ['no_fs_sag','no_fs_cor','no_nfs_sag','dup_fs_sag','n>=6']:
    print(c, int(G[c].sum()), 'gold58:', int(G[G.source=='original_gold58'][c].sum()))
t=pd.read_csv(B+'_data/train.csv',dtype={'StudyInstanceUID':str})
lab=t.dropna(subset=['ACL']); print('train.csv labelled rows', len(lab))
