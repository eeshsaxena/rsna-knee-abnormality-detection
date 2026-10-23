# G1: what do train series flagged (F,F) look like by DICOM header? (458 header studies; train only, no OAI)
import pandas as pd, numpy as np, re, sys
T = pd.read_csv(r'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/box/dicom_series_tags.csv', dtype=str)
S = pd.read_csv(r'C:/Users/eeshs/Downloads/rsna-knee/_data/train_series.csv', dtype=str)
m = T.merge(S, on=['StudyInstanceUID', 'SeriesInstanceUID'], how='inner')
print('tag rows', len(T), 'joined', len(m), 'studies', m.StudyInstanceUID.nunique())
f = lambda c: pd.to_numeric(m[c], errors='coerce')
m['TR'], m['TE'] = f('RepetitionTime'), f('EchoTime')
d = m['SeriesDescription'].fillna('').str.upper()
so = m['ScanOptions'].fillna('').str.upper()
ss = m['ScanningSequence'].fillna('').str.upper()
m['fs_dicom'] = (so.str.contains('FS') | so.str.contains('SPIR') | so.str.contains('SFS') | d.str.contains(r'FS|FATSAT|FAT SAT|SPAIR|SPIR|STIR|TIRM|\bFSE?\b|_FS|SPAIR|DIXON|WATER|\bW\b') | ss.str.contains('IR'))
m['stir'] = d.str.contains('STIR|TIRM') | (ss.str.contains('IR') & (m.TE < 80))
def wt(r):
    if r.TR < 1000 and r.TE < 30: return 'T1'
    if r.TR >= 1500 and r.TE < 50: return 'PD'
    if r.TR >= 1500 and r.TE >= 50: return 'T2'
    if r.TR < 1000 and r.TE >= 30: return 'GRE/T2*'
    return 'other'
m['w'] = m.apply(wt, axis=1)
m['flag'] = m.Fluid_Sensitive + m.Fat_Suppression
print('\nflag x weighting x fs_dicom')
print(pd.crosstab([m.flag, m.fs_dicom], m.w))
print('\ndescription samples, flag 00 and weighting PD/T2:')
x = m[(m.flag == '00') & m.w.isin(['PD', 'T2'])]
print(len(x), 'rows'); print(x.SeriesDescription.value_counts().head(25).to_string())
print('\nflag 00 fs_dicom True samples:')
x = m[(m.flag == '00') & m.fs_dicom]
print(len(x)); print(x[['SeriesDescription', 'ScanOptions', 'ScanningSequence', 'TR', 'TE']].head(20).to_string())
print('\nflag 11 fs_dicom False samples:')
x = m[(m.flag == '11') & ~m.fs_dicom]
print(len(x)); print(x[['SeriesDescription', 'ScanOptions', 'ScanningSequence', 'TR', 'TE', 'w']].head(30).to_string())
print('\nflag 11 T1 weighting:')
x = m[(m.flag == '11') & (m.w == 'T1')]
print(len(x)); print(x[['SeriesDescription', 'ScanOptions', 'TR', 'TE']].head(20).to_string())
m.to_csv(r'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/work/g1_tags_joined.csv', index=False)
