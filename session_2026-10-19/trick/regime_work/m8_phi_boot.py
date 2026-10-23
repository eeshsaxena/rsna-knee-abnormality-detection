# m8_phi_boot.py: paired bootstrap (same 58 studies) of phi_report - phi_expert for every label pair; where does LatOA~PF OA sit?
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, gold, R
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(ids)[L].fillna(0).values >= .5
def phim(X):
    X = X.astype(float); C = np.corrcoef(X.T); return np.nan_to_num(C)
G0 = phim(rep) - phim(Y)
rng = np.random.default_rng(5); B = []
for _ in range(3000):
    ix = rng.integers(0, n, n); B.append(phim(rep[ix]) - phim(Y[ix]))
B = np.array(B)
rows = []
for a in range(12):
    for b in range(a + 1, 12):
        v = B[:, a, b]; rows.append((G0[a, b], v.std(), (v > 0).mean(), L[a], L[b]))
rows.sort(key=lambda r: -abs(r[0]))
print('pair | phi_report(sol56) - phi_expert on the same 58 | boot sd | P(gap>0)')
for r in rows[:12]: print('  %-16s ~ %-16s %+.3f  sd %.3f  P>0 %.3f' % (r[3], r[4], r[0], r[1], r[2]))
k = [r for r in rows if {r[3], r[4]} == {'Lateral OA', 'PF OA'}][0]
print('LatOA~PF OA rank by |gap| %d of %d: gap %+.3f sd %.3f P>0 %.4f' % (rows.index(k) + 1, len(rows), k[0], k[1], k[2]))
print('fraction of 66 pairs with |gap| > .2: %.2f' % np.mean([abs(r[0]) > .2 for r in rows]))
