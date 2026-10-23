"""Probe: is the 1-rank stage1 mismatch a float tie-break from the member accumulation order?"""
import numpy as np, itertools
from inject_common import *
D = load_all()
ref = D['st1']
# exact-arithmetic version: integer rank sums (ranks*2 are integers with average ties) -> no float noise
R2 = np.stack([pd.DataFrame(m.astype(np.float64)).rank(method='average').to_numpy() * 2 for m in D['dino']])
acc_int = R2.sum(0)
exact = rank_pct(acc_int)
print('exact-integer-tie version vs stored: max|diff|', np.abs(exact - ref).max())
print('manifest-order float version vs stored:', np.abs(stage1_dino(D['dino']) - ref).max())
rng = np.random.default_rng(0); best = 9
for k in range(300):
    o = rng.permutation(20); d = np.abs(stage1_dino(D['dino'], o) - ref).max(); best = min(best, d)
    if d == 0: print('exact match with order', o.tolist()); break
print('best over 300 random member orders', best)
# where do ties sit?
n_ties = sum(len(acc_int[:, j]) - len(np.unique(acc_int[:, j])) for j in range(12))
print('exact rank-sum ties across labels:', n_ties)
