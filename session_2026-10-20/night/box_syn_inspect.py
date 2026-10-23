# box_syn_inspect.py (10-20 night, Claude). Box, read-only, AGGREGATES ONLY: file existence, column NAMES, grade histograms, counts.
import os, glob
import pandas as pd
H = os.path.expanduser('~')
for p in ['/r5_oai/run_r6/worker0/predictions.csv', '/r5_oai/s6_r5/out/predictions.csv', '/oai_build/oai_label_sources.csv',
          '/oai_build/oai_labels_T123.csv', '/oai_build/teacher/teacher_D5_mean_v00.csv',
          '/oai_build/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv', '/oai_build/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv']:
    print(p, os.path.exists(H + p))
S = pd.read_csv(H + '/oai_build/oai_label_sources.csv', dtype={'kid': str})
print('sources n', len(S), '| cols with eff/sy/hof/pop:', [c for c in S.columns if any(k in c.lower() for k in ('eff', 'sy', 'hof', 'pop'))])
for c in [c for c in S.columns if any(k in c.lower() for k in ('eff', 'syic', 'hof'))]:
    print(c, 'hist', S[c].value_counts(dropna=False).sort_index().to_dict())
for p in ['/r5_oai/run_r6/worker0/predictions.csv', '/r5_oai/s6_r5/out/predictions.csv']:
    d = pd.read_csv(H + p, dtype={'StudyInstanceUID': str})
    print(p, 'rows', len(d), 'oai rows', int(d.StudyInstanceUID.str.startswith('oai_').sum()), 'cols', d.columns.tolist()[:14])
print('kid prefix oai_ share', float(S.kid.str.startswith('oai_').mean()))
