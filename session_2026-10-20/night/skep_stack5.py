# skep_stack5.py (10-20 night, Claude, SKEPTIC): gold58-only vs extra16 couplings for the probe pairs (local, no OAI). Prints only.
import sys
import numpy as np
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import L, tet
D = CL.data(); J = {c: i for i, c in enumerate(L)}
Y58 = D['Y58'].astype(float); Y74 = D['Y74']
for a, b in (('Lateral OA', 'Medial Meniscus'), ('Medial Meniscus', 'Lateral Meniscus'), ('Medial OA', 'Lateral OA'), ('Lateral OA', 'PF OA')):
    for nm, Y in (('gold58', Y58), ('gold74', Y74)):
        m = ~np.isnan(Y[:, J[a]]) & ~np.isnan(Y[:, J[b]]); x = Y[m, J[a]].astype(int); y = Y[m, J[b]].astype(int)
        c = [((x == 0) & (y == 0)).sum(), ((x == 0) & (y == 1)).sum(), ((x == 1) & (y == 0)).sum(), ((x == 1) & (y == 1)).sum()]
        cs = np.array(c) + .5; lor = np.log(cs[3] * cs[0] / (cs[1] * cs[2])); sd = np.sqrt((1 / cs).sum())
        print('%-34s %s n %d cells %s tet %+.3f log OR %+.2f (sd %.2f)' % (a + ' x ' + b, nm, m.sum(), c, tet(x, y), lor, sd))
