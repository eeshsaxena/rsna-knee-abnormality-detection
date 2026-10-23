# member_g58.py (10-19 trick audit, Claude, research only): per-member per-label gold58 AUC (label-order sanity: a column
# permutation would show ~.5 on the swapped labels), duplicate / constant rows per member, and member-vs-author fidelity
# (S6fall: our n42gpu serve vs nartaa's published gold58 preds).
import sys; sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import numpy as np, pandas as pd
from load import *
from scipy.stats import spearmanr
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
r224ids = pd.read_csv(G + '/gold58_r224_ids.csv', dtype=str).StudyInstanceUID.tolist()
print('r224/r384 npy row order == gold58 order:', r224ids == ids)
M = {'r384': np.load(G + '/gold58_r384_kernelserve_cpu.npy'), 'r224': np.load(G + '/gold58_r224_kernelserve.npy'),
     'R6': rd('R6_gold58_serve.csv'), 'S6f_ours': rd('n42gpu_gold58.csv'), 'S6f_nartaa': rd('nartaa_gold58.csv')}
print('%-17s' % 'label' + ''.join('%11s' % k for k in M))
for j, c in enumerate(L):
    print('%-17s' % c + ''.join('%11.4f' % auc(M[k][:, j], Y[:, j]) for k in M))
print('%-17s' % 'macro' + ''.join('%11.4f' % np.mean([auc(M[k][:, j], Y[:, j]) for j in range(12)]) for k in M))
for k, v in M.items():
    print(k, 'dup rows', len(v) - len(np.unique(np.round(v, 6), axis=0)), '| rows all 0.5', int((v == .5).all(1).sum()), '| min/max', v.min().round(6), v.max().round(6))
a, b = M['S6f_ours'], M['S6f_nartaa']
print('S6fall fidelity: max |ours - nartaa| %.2e, min per-label spearman %.5f' % (np.abs(a - b).max(), min(spearmanr(a[:, j], b[:, j])[0] for j in range(12))))
