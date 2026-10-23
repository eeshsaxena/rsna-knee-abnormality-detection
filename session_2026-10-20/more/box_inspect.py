# box_inspect.py (10-20 more, Claude). Read-only. Prints AGGREGATES ONLY: column names and per-column value counts
# (grade histograms). No id, row or per-knee value is printed. Piped via ssh stdin into python -I.
import os, pandas as pd, numpy as np
H = os.path.expanduser('~/oai_build')
T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str})
print('T123 cols:', [c for c in T.columns if c != 'StudyInstanceUID'])
for c in T.columns:
    if c == 'StudyInstanceUID': continue
    vc = T[c].round(3).value_counts(dropna=False).sort_index()
    print('  %-18s %s' % (c, dict(vc)))
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str})
print('sources n cols', S.shape[1])
print('sources cols:', list(S.columns))
for c in S.columns:
    if c in ('kid',) or S[c].dtype == object: continue
    vc = S[c].round(2).value_counts(dropna=False)
    if len(vc) <= 25: print('  %-14s %s' % (c, dict(vc.sort_index())))
    else: print('  %-14s nunique %d nonnull %d' % (c, S[c].nunique(), S[c].notna().sum()))
for f in ['/teacher/teacher_D5_mean_v00.csv', '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv', '/teacher/f0_s1043_ttr_frac75__raw__A__shard0of4.csv']:
    d = pd.read_csv(H + f, nrows=3)
    print(f, 'cols:', list(d.columns))
