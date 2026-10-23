# pairscan.py: all 132 (target <- helper) borrowings s' = pct(c4_t) + a * pct(c4_h) at a = .1 on gold58 (null distribution for the
# Synovitis <- Effusion pick), and the same 132 on the OOF ruler (R7 + s44 .5/.5 rank blend vs v4 report labels) for the regime check.
import numpy as np, pandas as pd
from load import *
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
P = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j])
    P[:, j] = .8 * pct(w * rr + (1 - w) * pct(R6[:, j])) + .2 * pct(S6[:, j])
d7 = r7(); s44, _ = s6seed(1044); lab = labels('v4'); ii = [i for i in s44.index if i in d7.index]
Q = np.column_stack([.5 * pct(d7.loc[ii, 'p|' + c].values) + .5 * pct(s44.loc[ii, c].values) for c in L]); YQ = lab.loc[ii, L].values.astype(int)
rows = []
for t in range(12):
    b0 = auc(P[:, t], Y[:, t]); q0 = auc(Q[:, t], YQ[:, t])
    for h in range(12):
        if h == t: continue
        dg = auc(pct(P[:, t]) + .1 * pct(P[:, h]), Y[:, t]) - b0
        dq = auc(pct(Q[:, t]) + .1 * pct(Q[:, h]), YQ[:, t]) - q0
        rows.append((L[t], L[h], dg, dq))
df = pd.DataFrame(rows, columns=['target', 'helper', 'd_gold58', 'd_oof_v4'])
print('gold58 a=.1 deltas over 132 pairs: mean %+.4f, sd %.4f, frac>0 %.2f, frac<0 %.2f' % (df.d_gold58.mean(), df.d_gold58.std(), (df.d_gold58 > 0).mean(), (df.d_gold58 < 0).mean()))
print('OOF   a=.1 deltas over 132 pairs: mean %+.4f, sd %.4f, frac>0 %.2f' % (df.d_oof_v4.mean(), df.d_oof_v4.std(), (df.d_oof_v4 > 0).mean()))
print('corr(gold58 delta, OOF delta) over pairs: %.3f (Spearman %.3f)' % (df.d_gold58.corr(df.d_oof_v4), df.d_gold58.corr(df.d_oof_v4, method='spearman')))
df['rank_gold'] = df.d_gold58.rank(ascending=False)
print(df.sort_values('d_gold58', ascending=False).head(12).round(4).to_string(index=False))
print('Synovitis <- Effusion:', df[(df.target == 'Synovitis') & (df.helper == 'Effusion')].round(4).to_string(index=False, header=False))
print('OOF top 8:'); print(df.sort_values('d_oof_v4', ascending=False).head(8).round(4).to_string(index=False))
df.to_csv('pairscan.csv', index=False)
