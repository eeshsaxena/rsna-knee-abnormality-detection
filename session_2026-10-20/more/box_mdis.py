# box_mdis.py (10-20 more, Claude). Read-only, AGGREGATES ONLY: mean across knees of the member-percentile sd per label (how far
# apart the OAI-free members are), plus mean pairwise Spearman between members. No id / row / per-knee value printed.
import os, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
H = os.path.expanduser('~/oai_build')
L9 = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"]
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[L9]
P = {'A': rd(H + '/teacher/teacher_D5_mean_v00.csv'), 'B': rd(H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv'),
     'C': rd(H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv')}
for k in range(5):
    fs = sorted(glob.glob(H + '/teacher/f%d_s1043_ttr_frac75__raw__A__shard*of4.csv' % k))
    P['f%d' % k] = pd.concat([pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in fs]).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[L9]
T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
common = T.index
for k in P: common = common.intersection(P[k].dropna().index)
pct = lambda v: rankdata(v) / len(v)
for sn, mem in (('3fam', ['A', 'B', 'C']), ('7mem', ['f0', 'f1', 'f2', 'f3', 'f4', 'B', 'C'])):
    out = []
    for c in L9:
        U = np.stack([pct(P[m].loc[common, c].values) for m in mem])
        rs = [spearmanr(U[i], U[j]).correlation for i in range(len(mem)) for j in range(i + 1, len(mem))]
        out.append('%s sd %.3f rho %.3f' % (c[:6], U.std(0).mean(), np.mean(rs)))
    print(sn, 'n', len(common), '|', ' ; '.join(out))
