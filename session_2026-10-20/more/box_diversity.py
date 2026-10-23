# box_diversity.py (10-20 more, Claude): read-only, aggregates only. Mean between-member Spearman (per label, averaged over the
# 9 OAI-covered labels) for the OAI-free stand-in members, to compare member diversity with the report-OOF ruler's members.
# Piped: ssh training@192.168.10.30 '~/miniconda3/envs/rsna/bin/python -I - ' < box_diversity.py
import os, glob, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
H = os.path.expanduser('~/oai_build')
L9 = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"]
P = {}
for k in range(5):
    fs = sorted(glob.glob(H + '/teacher/f%d_s1043_ttr_frac75__raw__A__shard*of4.csv' % k))
    P['f%d' % k] = pd.concat([pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in fs]).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[L9]
for k, f in {'s1044': H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv', 's1045': H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv',
             'D5': H + '/teacher/teacher_D5_mean_v00.csv'}.items():
    P[k] = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[L9]
ids = None
for k in P:
    ix = P[k].dropna().index; ids = ix if ids is None else ids.intersection(ix)
print('n knees %d' % len(ids))
sp = lambda a, b: np.mean([spearmanr(P[a].loc[ids, c], P[b].loc[ids, c]).correlation for c in L9])
for a, b in [('f0', 'f1'), ('f0', 's1044'), ('f0', 's1045'), ('s1044', 's1045'), ('D5', 's1044'), ('D5', 's1045')]:
    print('mean per-label Spearman %-6s - %-6s %.3f' % (a, b, sp(a, b)))
fm = np.mean([sp('f%d' % i, 'f%d' % j) for i, j in itertools.combinations(range(5), 2)])
print('mean fold-fold Spearman %.3f' % fm)
