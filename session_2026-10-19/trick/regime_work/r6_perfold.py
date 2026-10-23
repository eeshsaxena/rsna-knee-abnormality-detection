# r6_perfold.py: R6 single-fold vs 4-fold-mean gold58 macro AUC from the box serve dump (run_b, all 58), to put the CLAIMED
# B-leg single-fold gold numbers (.911 / .927) on the same scale.
import sys, json, numpy as np
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, gold, G
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int)
d = json.load(open(G + '/r6g58_box/run_b/worker0/dump.json'))
Z = np.array([d[u]['z'] for u in ids])  # (58, 4, 12)
Zp = np.array([d[u]['z_plain'] for u in ids])
print('dump z shape', Z.shape)
mac = lambda S: np.mean([auc(S[:, j], Y[:, j]) for j in range(12)])
for f in range(4):
    print('fold %d flipTTA z macro %.4f | plain %.4f' % (f, mac(Z[:, f]), mac(Zp[:, f])))
print('4-fold mean z macro %.4f (final R6 output with fluid / Baker stackers: see R6_gold58_serve receipt .9366)' % mac(Z.mean(1)))
print('mean of single-fold macros %.4f' % np.mean([mac(Z[:, f]) for f in range(4)]))
