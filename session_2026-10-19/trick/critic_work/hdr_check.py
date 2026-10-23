import pandas as pd, numpy as np, re
d = pd.read_csv('session_2026-10-19/trick/box/dicom_series_tags.csv', dtype=str)
print('rows', len(d), 'studies', d.StudyInstanceUID.nunique())
st = d.groupby('StudyInstanceUID').agg(sex=('PatientSex', lambda s: '|'.join(sorted(set(s.dropna())))),
                                       age=('PatientAge', lambda s: '|'.join(sorted(set(s.dropna())))),
                                       desc=('SeriesDescription', lambda s: ' || '.join(s.dropna())),
                                       body=('BodyPartExamined', lambda s: '|'.join(sorted(set(s.dropna())))),
                                       nser=('SeriesInstanceUID', 'nunique'))
print('sex values', st.sex.value_counts().to_dict())
print('age non-empty', (st.age != '').sum())
pat = re.compile(r'(\+C|\bGD\b|GADO|CONTRAST|\bKM\b|POST|\bPOS\b|DYN|SUB)', re.I)
hit = d[d.SeriesDescription.fillna('').str.contains(pat)]
print('contrast-like series', len(hit), 'studies', hit.StudyInstanceUID.nunique())
print(hit.SeriesDescription.value_counts().head(15).to_dict())
tr = pd.read_csv('_data/train.csv', dtype={'StudyInstanceUID': str})
lab = [c for c in tr.columns if c not in ('StudyInstanceUID', 'Report')]
g = tr.dropna(subset=lab)
print('gold58 in header sample', g.StudyInstanceUID.isin(st.index).sum())
st.to_csv(__import__('os').environ['TEMP'] + '/hdr_study.csv')
