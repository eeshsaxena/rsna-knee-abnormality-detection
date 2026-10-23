import pandas as pd, numpy as np
D='C:/Users/eeshs/Downloads/rsna-knee/_data/'
s=pd.read_csv(D+'train_series.csv'); t=pd.read_csv(D+'train.csv'); lang=pd.read_csv(D+'_reports_lang.csv',usecols=['StudyInstanceUID','lang'])
ts=pd.read_csv(D+'test_series.csv')
print('train studies',t.StudyInstanceUID.nunique(),'series',len(s),'studies in series',s.StudyInstanceUID.nunique())
print(s.Anatomical_Plane.value_counts(dropna=False)); print(pd.crosstab(s.Fluid_Sensitive,s.Fat_Suppression))
def feats(s):
    g=s.groupby('StudyInstanceUID')
    f=pd.DataFrame({'n':g.size()})
    for p in ['Sagittal','Coronal','Axial']:
        f['has_'+p]=g.apply(lambda d:(d.Anatomical_Plane==p).sum())
        f['fs_'+p]=g.apply(lambda d:((d.Anatomical_Plane==p)&(d.Fluid_Sensitive==1)).sum())
        f['nfs_'+p]=g.apply(lambda d:((d.Anatomical_Plane==p)&(d.Fluid_Sensitive==0)).sum())
    f['any_fs']=g.Fluid_Sensitive.sum()
    f['any_fat']=g.Fat_Suppression.sum()
    return f
f=feats(s); ft=feats(ts)
print(ft)
f=f.join(lang.set_index('StudyInstanceUID'))
B=lambda c: (c>0)
out={}
out['lack_sag']=(f.has_Sagittal==0); out['lack_cor']=(f.has_Coronal==0); out['lack_ax']=(f.has_Axial==0)
out['lack_any_plane']=out['lack_sag']|out['lack_cor']|out['lack_ax']
out['no_fs']=f.any_fs==0
out['no_fs_sag']=f.fs_Sagittal==0; out['no_fs_cor']=f.fs_Coronal==0; out['no_fs_ax']=f.fs_Axial==0
out['no_nfs_sag']=f.nfs_Sagittal==0
out['n<=2']=f.n<=2; out['n<=3']=f.n<=3; out['n>=6']=f.n>=6
out['dup_fs_sag']=f.fs_Sagittal>=2; out['dup_sag']=f.has_Sagittal>=3
out['no_fat']=f.any_fat==0
o=pd.DataFrame(out)
print('\nshare of train studies:'); print(o.mean().round(4))
print('\nn series dist'); print(f.n.value_counts().sort_index())
o['lang']=f.lang
top=f.lang.value_counts(); print(top.head(12))
langs=top[top>=40].index
print('\nby lang (langs with >=40 studies):')
print(o[o.lang.isin(langs)].groupby('lang').mean().round(3).T.to_string())
o.to_csv('train_series_flags.csv')
f.to_csv('train_series_feats.csv')
# pattern of plane x fs
f['pattern']=f.apply(lambda r: f"S{int(r.fs_Sagittal)}/{int(r.nfs_Sagittal)} C{int(r.fs_Coronal)}/{int(r.nfs_Coronal)} A{int(r.fs_Axial)}/{int(r.nfs_Axial)}",axis=1)
print(f.pattern.value_counts().head(25))
ft['pattern']=ft.apply(lambda r: f"S{int(r.fs_Sagittal)}/{int(r.nfs_Sagittal)} C{int(r.fs_Coronal)}/{int(r.nfs_Coronal)} A{int(r.fs_Axial)}/{int(r.nfs_Axial)}",axis=1)
print(ft.pattern)
