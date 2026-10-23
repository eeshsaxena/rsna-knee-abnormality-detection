# skep_lomm05_g58.py (10-21 levers, Claude, lomm05 SKEPTIC). Local CPU, NO OAI. Aggregates only: gold58 / gold74 expert LOA x MM 2x2 and a
# bootstrap of the expert tetrachoric, to show how much the face-value regimes (G74 / RAW) lean on a few knees.
import numpy as np
from rg_lib import L, J, tet_xy_fast, LD
g, g58 = LD.gold(); i, j = J['Lateral OA'], J['Medial Meniscus']
for nm, G in (('gold58', g58), ('gold74', g)):
    E = G[L].values.astype(float); a, b = E[:, i], E[:, j]
    T = [[int(((a == x) & (b == y)).sum()) for y in (0, 1)] for x in (0, 1)]
    rr = np.random.default_rng(5); bs = []
    for _ in range(2000):
        k = rr.integers(0, len(E), len(E)); bs.append(tet_xy_fast(a[k], b[k]))
    bs = np.array(bs)
    print('%s expert LOA x MM 2x2 [[LOA-MM-, LOA-MM+], [LOA+MM-, LOA+MM+]] = %s | tet %.3f | boot p10 / p50 / p90 %.3f / %.3f / %.3f | P(tet > .265 EB target) %.2f | P(tet > .30) %.2f' % (
        nm, T, tet_xy_fast(a, b), *np.percentile(bs, [10, 50, 90]), (bs > .265).mean(), (bs > .30).mean()))
