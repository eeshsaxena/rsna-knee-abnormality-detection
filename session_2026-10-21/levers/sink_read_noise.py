# sink_read_noise.py (10-21, Claude, SINK track). LOCAL CPU, NO OAI. Paired read noise of CVX (convex E-OA p 2) vs its base at public
# (390) and private (910) size, method of stack/kernels/read_noise.py (R7 OOF non-gold, report v2 labels, 1,000 subsets without
# replacement, the SAME subsets for every component -> cross-component read-error correlations). Aggregates only.
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
from census_lib import auc, pct, L
import load as LD
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/sink_read_noise.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n')
rng = np.random.default_rng(21022)
J = {c: i for i, c in enumerate(L)}; OA = ['Medial OA', 'Lateral OA', 'PF OA']
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
def pw(F, a, p=1.):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}; S = {c: U[c] ** p for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * S[o[0]] + w[1] * S[o[1]]))
    return G
def line(F, t, o, a):
    G = F.copy(); G[:, J[t]] = pct(pct(F[:, J[t]]) - a * pct(F[:, J[o]])); return G
r7 = LD.r7(); g_, _ = LD.gold(); gold = set(g_.StudyInstanceUID); lab = LD.labels('v2')[L]
ids = [u for u in r7.index if u not in gold and u in lab.index]
X = r7.loc[ids, ['p|' + c for c in L]].values.astype(float); Y = (lab.loc[ids].fillna(0).values >= .5).astype(int)
P('== labels v2: n %d non-gold R7 OOF studies' % len(ids))
K = ['pw25', 'lomm05', 'cvx25', 'cvx30', 'cvx25_on_spl']
for size in (390, 910):
    D = {k: [] for k in K}
    for _ in range(1000):
        ix = rng.choice(len(ids), size, replace=False); y = Y[ix]
        mac = lambda G, cols: sum(auc(G[:, J[c]], y[:, J[c]]) for c in cols) / 12
        F30 = pw(X[ix], .3); F25 = pw(X[ix], .25); C25 = pw(X[ix], .25, 2.); C30 = pw(X[ix], .3, 2.)
        b30 = mac(F30, OA); b25 = mac(F25, OA)
        D['pw25'].append((b25 - b30) * 1e4)
        D['lomm05'].append((mac(line(F30, 'Lateral OA', 'Medial Meniscus', .05), ['Lateral OA']) - mac(F30, ['Lateral OA'])) * 1e4)
        D['cvx25'].append((mac(C25, OA) - b25) * 1e4); D['cvx30'].append((mac(C30, OA) - b30) * 1e4)
        S25 = line(F25, 'Lateral OA', 'Medial Meniscus', .05); SC = line(C25, 'Lateral OA', 'Medial Meniscus', .05)
        D['cvx25_on_spl'].append((mac(SC, OA) - mac(S25, OA)) * 1e4)
    A = np.array([D[k] for k in K]); C = np.corrcoef(A)
    P('  n %3d | sd e-4: %s | mean (report labels, INFO): %s' % (size, ' '.join('%s %.2f' % (k, s) for k, s in zip(K, A.std(1))), ' '.join('%s %+.2f' % (k, m) for k, m in zip(K, A.mean(1)))))
    P('          corr: ' + ' | '.join('%s~%s %+.2f' % (K[i], K[j], C[i, j]) for i in range(len(K)) for j in range(i + 1, len(K))))
P('SINK_READ_NOISE DONE')
OUT.close()
