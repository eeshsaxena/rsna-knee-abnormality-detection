# m1_sites.py (10-19 trick lens, Claude, research only, CPU): site proxies in train. Reads local copies only.
import numpy as np, pandas as pd
R='C:/Users/eeshs/Downloads/rsna-knee'; B=R+'/session_2026-10-19/trick/box'
lang=pd.read_csv(R+'/_data/_reports_lang.csv',usecols=['StudyInstanceUID','lang']).set_index('StudyInstanceUID').lang
dt=pd.read_csv(B+'/dicom_series_tags.csv',dtype=str)
dt['lang']=dt.StudyInstanceUID.map(lang)
st=dt.drop_duplicates('StudyInstanceUID')
print('DICOM subset studies',len(st)); print(st.lang.value_counts().to_dict())
st=st.assign(fs=pd.to_numeric(st.MagneticFieldStrength,errors='coerce'))
print('\nfield strength per lang (study-level, first series):')
print(pd.crosstab(st.lang,st.fs.round(1)))
print('\nmanufacturer per lang:')
print(pd.crosstab(st.lang,st.Manufacturer.str.split().str[0].str.upper()))
print('\nmodel names (top 25):'); print(st.ManufacturerModelName.value_counts().head(25).to_string())
print('\nInstitutionName non-empty:',st.InstitutionName.notna().sum(),' StationName non-empty:',st.StationName.notna().sum())
# transfer syntax not in tags; image size per lang
dt['rows']=pd.to_numeric(dt.Rows,errors='coerce')
print('\nRows median per lang:',dt.groupby('lang').rows.median().to_dict())
print('\nScanningSequence/ScanOptions FS method per lang (series-level, top):')
dt['fsopt']=dt.ScanOptions.fillna('').str.contains('FS').astype(int)
dt['ir']=dt.ScanningSequence.fillna('').str.contains('IR').astype(int)
dt['stir']=dt.SeriesDescription.fillna('').str.upper().str.contains('STIR|TIRM').astype(int)
print(dt.groupby('lang')[['fsopt','ir','stir']].mean().round(3))
# series composition
ts=pd.read_csv(R+'/_data/train_series.csv'); te=pd.read_csv(R+'/_data/test_series.csv')
def comp(s):
    s=s.copy(); s['k']=s.Anatomical_Plane.str[:3]+s.Fluid_Sensitive.astype(str)
    c=pd.crosstab(s.StudyInstanceUID,s.k); return c
ct=comp(ts); ce=comp(te)
for c in ['Sag1','Sag0','Cor1','Cor0','Axi1','Axi0']:
    if c not in ct: ct[c]=0
    if c not in ce: ce[c]=0
ct['lang']=ct.index.map(lang)
pres=(ct[['Sag1','Sag0','Cor1','Cor0','Axi1','Axi0']]>0)
pres['lang']=ct.lang
print('\nslot presence per lang (train):'); print(pres.groupby('lang').mean().round(3)); print('ALL',pres.drop(columns='lang').mean().round(3).to_dict())
print('\nvisible test studies:'); print(ce)
ct['nser']=ct[['Sag1','Sag0','Cor1','Cor0','Axi1','Axi0']].sum(1)
print('\nseries count per lang:',ct.groupby('lang').nser.describe().round(2).to_string())
miss_any=(~pres[['Sag1','Sag0','Cor1','Axi1']].astype(bool)).any(axis=1)
print('\nstudies missing >=1 of Sag-FS/Sag-nonFS/Cor-FS/Axi-FS:',int(miss_any.sum()),'of',len(pres))
print('no fluid-sensitive at all:',int(((ct.Sag1+ct.Cor1+ct.Axi1)==0).sum()),' no sagittal:',int(((ct.Sag1+ct.Sag0)==0).sum()))
ct.to_csv('series_comp_local.csv')
