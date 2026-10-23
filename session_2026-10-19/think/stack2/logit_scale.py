# logit_scale.py: why flat_logit moved on gold58: effective member weights under logit averaging = c4 weight x logit SD.
# Gold58 serve predictions (same files as stack_ops.py). Also R7 / s44 OOF logit SDs on the train ruler.
import numpy as np, pandas as pd
from load import *
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist()
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
print('label | logit SD r384 r224 R6 S6fall | c4 flat weight | effective logit weight (r384 r224 R6 S6) | R6 cells at |logit|>=15')
rows = []
for j, c in enumerate(L):
    w = C4.get(c, .5); e = np.array([.8 * w * .75, .8 * w * .25, .8 * (1 - w), .2])
    sd = np.array([lg(X[:, j]).std() for X in (R384, R224, R6, S6)])
    ef = e * sd / (e * sd).sum()
    sat = int((np.abs(lg(R6[:, j])) >= 15).sum())
    rows.append((c, sd, e, ef, sat))
    print('%-17s | %5.2f %5.2f %5.2f %5.2f | %s | %s | %d' % (c, *sd, np.round(e, 2), np.round(ef, 2), sat))
d7 = r7(); s44, _ = s6seed(1044); ii = [i for i in s44.index if i in d7.index]
print('OOF ruler logit SD (R7 p, R7 z, s44):')
for j, c in enumerate(L):
    print('  %-17s %5.2f %5.2f %5.2f' % (c, lg(d7.loc[ii, 'p|' + c].values).std(), d7.loc[ii, 'z|' + c].values.std(), lg(s44.loc[ii, c].values).std()))
