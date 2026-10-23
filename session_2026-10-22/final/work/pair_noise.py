# pair_noise.py (10-22, Claude, PUBLIC-PUSH track). LOCAL CPU, NO OAI, NO GPU. Paired PUBLIC read noise (n 390, 1,000 subsets without
# replacement, the SAME subsets for every component) of the candidate changes ON THE SWIDE BASE (pw25 + LOA<-MM .05 + MCL<-MM .1),
# method of levers/sink_read_noise.py (R7 OOF non-gold, report v2 labels). Gives the cross-candidate read-error correlations the pair
# choice needs. Means are report-label INFO only (not expert, not a ruler). Aggregates only.
import sys
import numpy as np
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
from census_lib import auc, pct, L
import load as LD
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-22/final/work/pair_noise.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n')
rng = np.random.default_rng(22101)
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
def stack(F, al, am):
    return line(line(F, 'Lateral OA', 'Medial Meniscus', al), 'MCL', 'Medial Meniscus', am)
def moacf(F, a):
    return line(line(F, 'Medial OA', 'Contusion', a), 'Medial OA', 'Fracture', a)
r7 = LD.r7(); g_, _ = LD.gold(); gold = set(g_.StudyInstanceUID); lab = LD.labels('v2')[L]
ids = [u for u in r7.index if u not in gold and u in lab.index]
X = r7.loc[ids, ['p|' + c for c in L]].values.astype(float); Y = (lab.loc[ids].fillna(0).values >= .5).astype(int)
P('== labels v2: n %d non-gold R7 OOF studies; base = swide (pw25 + lomm05 + MCLmm .1)' % len(ids))
K = ['Q05', 'Q10', 'L08', 'M15', 'CVX', 'L08M15', 'L08Q05', 'M15Q05', 'pw25_vs_pw30']
for size in (390,):
    D = {k: [] for k in K}
    for _ in range(1000):
        ix = rng.choice(len(ids), size, replace=False); y = Y[ix]
        mac = lambda G: sum(auc(G[:, j], y[:, j]) for j in range(12)) / 12
        F25 = pw(X[ix], .25); SW = stack(F25, .05, .1); b = mac(SW)
        d = {}
        d['Q05'] = mac(moacf(SW, .05)) - b
        d['Q10'] = mac(moacf(SW, .10)) - b
        d['L08'] = mac(stack(F25, .08, .1)) - b
        d['M15'] = mac(stack(F25, .05, .15)) - b
        d['CVX'] = mac(stack(pw(X[ix], .25, 2.), .05, .1)) - b
        d['L08M15'] = mac(stack(F25, .08, .15)) - b
        d['L08Q05'] = mac(moacf(stack(F25, .08, .1), .05)) - b
        d['M15Q05'] = mac(moacf(stack(F25, .05, .15), .05)) - b
        d['pw25_vs_pw30'] = mac(F25) - mac(pw(X[ix], .3))
        for k in K: D[k].append(d[k] * 1e4)
    A = np.array([D[k] for k in K]); C = np.corrcoef(A)
    P('  n %3d | paired sd e-4: %s' % (size, ' '.join('%s %.2f' % (k, s) for k, s in zip(K, A.std(1)))))
    P('          mean (report labels, INFO only): %s' % ' '.join('%s %+.2f' % (k, m) for k, m in zip(K, A.mean(1))))
    P('          corr matrix (rows/cols %s):' % ' '.join(K))
    for i, k in enumerate(K):
        P('    %-12s %s' % (k, ' '.join('%+.2f' % C[i, j] for j in range(len(K)))))
P('PAIR_NOISE DONE')
OUT.close()
