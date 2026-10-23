# skep_c12p_rho2.py (10-20 night, Claude, SKEPTIC C12'): is gold58 a low co-firing sample for LOA-MM only, or for all degenerative pairs?
# And how correlated are c4's and R7's gold58 sampling deviations? Local only.
import sys
import numpy as np
from scipy.stats import norm, rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import pct, L
import load as LD
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/skep_c12p_rho2.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
J = {c: i for i, c in enumerate(L)}
D = CL.data()
r = lambda F, a, b: np.corrcoef(ns(F[:, J[a]]), ns(F[:, J[b]]))[0, 1]
PAIRS = [('Lateral OA', 'Medial Meniscus'), ('Lateral OA', 'Lateral Meniscus'), ('Medial OA', 'Medial Meniscus'), ('Medial OA', 'Lateral OA'),
         ('Lateral OA', 'PF OA'), ('Medial Meniscus', 'Lateral Meniscus'), ('Medial OA', 'Lateral Meniscus'), ('Lateral OA', 'Effusion')]
F7 = D['F7']; F7 = F7[~np.isnan(F7).any(1)]; R58 = D['R7_58']; C4 = D['C4']
s44, _ = LD.s6seed(1044); ids = D['ids']
P('raw rho_ns (no pw)       | R7 train | R7 gold58 | c4 gold58 | gold58 R7 - train')
for a, b in PAIRS:
    P('%-17s <-> %-17s | %.3f | %.3f | %.3f | %+.3f' % (a, b, r(F7, a, b), r(R58, a, b), r(C4, a, b), r(R58, a, b) - r(F7, a, b)))
P('s1044 OOF has gold58 knees: %d of 58' % sum(u in s44.index for u in ids))
rng = np.random.default_rng(11); n = len(ids); bc, br = [], []
for _ in range(2000):
    ii = rng.integers(0, n, n); bc.append(r(C4[ii], 'Lateral OA', 'Medial Meniscus')); br.append(r(R58[ii], 'Lateral OA', 'Medial Meniscus'))
bc, br = np.array(bc), np.array(br)
P('gold58 boot (raw LOA-MM): c4 sd %.3f, R7 sd %.3f, corr %.2f; implied E[c4 pop] = c4_g58 - corr*sd_c4/sd_R7*(R7_g58 - R7_train) = %.3f' % (
    bc.std(), br.std(), np.corrcoef(bc, br)[0, 1], r(C4, 'Lateral OA', 'Medial Meniscus') - np.corrcoef(bc, br)[0, 1] * bc.std() / br.std() * (r(R58, 'Lateral OA', 'Medial Meniscus') - r(F7, 'Lateral OA', 'Medial Meniscus'))))
# prevalence context: report v4 on gold58 knees vs train
lv4 = LD.labels('v4'); g = set(D['ids'])
tr = [u for u in lv4.index if u not in g]
P('report v4 prevalence gold58 vs train: ' + ' '.join('%s %.2f/%.2f' % (CL.AB[c], np.nanmean(lv4.reindex(D['ids'])[c].values), np.nanmean(lv4.loc[tr, c].values)) for c in L))
P('R7 mean p (raw prob) gold58 vs train: ' + ' '.join('%s %.2f/%.2f' % (CL.AB[c], np.nanmean(R58[:, J[c]]), np.nanmean(F7[:, J[c]])) for c in L))
