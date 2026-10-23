# skep_c12p_rho.py (10-20 night, Claude, SKEPTIC C12'): is c4's gold58 rho_ns(LOA, MM) on the pw30 base representative? Local only.
import sys
import numpy as np
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import pct, L
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/skep_c12p_rho.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}; OA = ['Medial OA', 'Lateral OA', 'PF OA']
W = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
def pw(F, a=.3):
    G = F.copy(); U = {c: pct(F[:, J[c]]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]; w = [W[tuple(sorted((c, k), key=OA.index))] for k in o]
        G[:, J[c]] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G
T, O = J['Lateral OA'], J['Medial Meniscus']
rng = np.random.default_rng(7)
def rho(F):
    m = ~np.isnan(F[:, [T, O] + [J[c] for c in OA]]).any(1); F = F[m]
    r0 = np.corrcoef(ns(F[:, T]), ns(F[:, O]))[0, 1]; G = pw(F); r1 = np.corrcoef(ns(G[:, T]), ns(G[:, O]))[0, 1]
    bs = []
    for _ in range(500):
        ii = rng.integers(0, len(F), len(F)); Gi = pw(F[ii]); bs.append(np.corrcoef(ns(Gi[:, T]), ns(Gi[:, O]))[0, 1])
    return m.sum(), r0, r1, np.std(bs)
D = CL.data()
for nm, F in (('c4 gold58', D['C4']), ('R7 gold58', D['R7_58']), ('R7 gold74', D['R7_74']), ('R7 train', D['F7']), ('M3 train', D['M3']),
              ('R6 gold58', D['members']['R6']), ('S6 gold58', D['members']['S6']), ('R384 gold58', D['members']['R384'])):
    n, r0, r1, sd = rho(np.asarray(F, float))
    P('%-12s n %4d | rho_ns raw %.3f | on pw30 %.3f (boot sd %.3f)' % (nm, n, r0, r1, sd))
