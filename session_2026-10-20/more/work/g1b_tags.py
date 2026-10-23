# G1b: refined fat-sat classifier from DICOM headers; train convention for the two flags (458 header studies, train only)
import pandas as pd, numpy as np, re
m = pd.read_csv(r'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/work/g1_tags_joined.csv', dtype={'Fluid_Sensitive': str, 'Fat_Suppression': str})
d = m['SeriesDescription'].fillna('').str.upper()
tok = m['ScanOptions'].fillna('').str.upper().str.split('|')
so_fs = tok.apply(lambda t: any(x in ('FS', 'SFS', 'WE', 'FSA_GEMS', 'FS_GEMS', 'CHEMSAT', 'SPAIR', 'SPIR') for x in t))
seq = m['ScanningSequence'].fillna('').str.upper()
ir = seq.str.contains('IR')
desc_fs = d.str.contains(r'(?:\bFS\b|_FS\b|_FS_|FS_|\bFS[A-Z]|FAT ?SAT|F/S|SPAIR|SPIR\b|SPIR |STIR|TIRM|\bWE\b|_WE_|DIXON|\bW\b|WATER|SMART FAT|FATSAT|\bFSE?_FS|IDEAL)', regex=True)
desc_fs = desc_fs | d.str.contains('_FS') | d.str.contains(' FS') | d.str.endswith('FS')
# GE: 'FSE' is fast spin echo, not fat sat: remove pure FSE hits
fse_only = d.str.contains('FSE') & ~d.str.replace('FSE', '').str.contains(r'FS|FAT|STIR|SPAIR|F/S')
desc_fs = desc_fs & ~fse_only
m['fsd'] = so_fs | desc_fs | ir
m['TR'] = pd.to_numeric(m['RepetitionTime'], errors='coerce'); m['TE'] = pd.to_numeric(m['EchoTime'], errors='coerce')
def wt(r):
    if 'GR' in str(r.ScanningSequence) and r.TR < 100: return 'GRE'
    if r.TR < 1000 and r.TE < 30: return 'T1'
    if r.TR >= 1000 and r.TE < 50: return 'PD'
    if r.TR >= 1000 and r.TE >= 50: return 'T2'
    return 'other'
m['w'] = m.apply(wt, axis=1)
m['flag'] = m.Fluid_Sensitive + m.Fat_Suppression
print(pd.crosstab([m.flag, m.fsd], m.w, margins=True))
print('\nagreement flag==fsd:', ((m.flag == '11') == m.fsd).mean().round(4), 'n', len(m))
print('\nT1-weighted SE with fat-sat by DICOM (would be (0,1) by the data-page definition):')
x = m[(m.w == 'T1') & m.fsd]; print(len(x)); print(x.groupby('flag').size().to_string())
print(x[['flag', 'SeriesDescription', 'ScanOptions', 'TR', 'TE', 'Anatomical_Plane']].head(25).to_string())
print('\nPD/T2 non-FS by DICOM (would be (1,0) by the definition):')
x = m[m.w.isin(['PD', 'T2']) & ~m.fsd]; print(len(x)); print(x.groupby('flag').size().to_string())
print(x[x.flag == '11'][['SeriesDescription', 'ScanOptions', 'ScanningSequence', 'TR', 'TE']].head(20).to_string())
print('\nflag 11 but fsd False:'); x = m[(m.flag == '11') & ~m.fsd]; print(len(x)); print(x.SeriesDescription.value_counts().head(15).to_string())
print('\nflag 00 but fsd True:'); x = m[(m.flag == '00') & m.fsd]; print(len(x)); print(x[['SeriesDescription', 'ScanOptions', 'ScanningSequence', 'TR', 'TE', 'w']].head(25).to_string())
st = m.groupby('StudyInstanceUID').apply(lambda g: (g.w.isin(['PD', 'T2']) & ~g.fsd).any())
print('\nstudies with >= 1 PD/T2 non-FS series (would carry a (1,0) row if hidden followed the definition): %d / %d = %.3f' % (st.sum(), len(st), st.mean()))
m.to_csv(r'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/work/g1b_classified.csv', index=False)
