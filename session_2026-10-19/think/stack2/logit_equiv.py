# logit_equiv.py: is gold58 flat_logit (+.00088) just an implicit re-weighting? Compare it with a RANK blend that uses the
# effective logit weights (c4 flat weight x member logit SD, renormalised) per label. If the per-label deltas match, the operator adds nothing.
import numpy as np, pandas as pd
from load import *
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
M = (R384, R224, R6, S6)
c4 = np.zeros((n, 12)); fl = np.zeros((n, 12)); eq = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); e = np.array([.8 * w * .75, .8 * w * .25, .8 * (1 - w), .2])
    rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); c4[:, j] = .8 * pct(w * rr + (1 - w) * pct(R6[:, j])) + .2 * pct(S6[:, j])
    fl[:, j] = sum(e[k] * lg(M[k][:, j]) for k in range(4))
    sd = np.array([lg(M[k][:, j]).std() for k in range(4)]); ef = e * sd / (e * sd).sum()
    eq[:, j] = sum(ef[k] * pct(M[k][:, j]) for k in range(4))
dl = np.array([auc(fl[:, j], Y[:, j]) - auc(c4[:, j], Y[:, j]) for j in range(12)])
de = np.array([auc(eq[:, j], Y[:, j]) - auc(c4[:, j], Y[:, j]) for j in range(12)])
for j in range(12): print('%-17s flat_logit %+.4f | rank blend at effective logit weights %+.4f' % (L[j], dl[j], de[j]))
print('macro: flat_logit %+.5f | effective-weight rank blend %+.5f | corr of per-label deltas %.3f' % (dl.mean(), de.mean(), np.corrcoef(dl, de)[0, 1]))
