# lingauss_shape.py (val SKEPTIC, local, no data): linear-Gaussian E-OA corr-gain curve
# g(a) = (1 - a q) / sqrt(1 - 2 a rho + a^2 (1 + rho) / 2) - 1  (sym family, per-label corr with own label; standardized U)
# Q: at a matched optimum (or matched public shape G20/G10 and S30/S20), does a higher prediction coupling rho (c4 ~.82 vs
# c4D5 .66 / D5 .78) change what lies beyond pw30 (S35, S40 relative to S30)?
import numpy as np
from scipy.optimize import brentq
A = np.round(np.arange(0, .7001, .025), 3)
def g(a, r, q): return (1 - a * q) / np.sqrt(1 - 2 * a * r + a * a * (1 + r) / 2) - 1
def astar(r, q): return (r - q) / ((1 + r) / 2 - r * q)
print('formula a* at fitted regimes: c4D5 P_FT MAP (rho .663 q .520) %.3f [OAI AUC argmax: sym .175 pw .225]; c4D5 P_AREA MAP (rho .668 q .543) %.3f [AUC sym .25 pw .25]' % (astar(.663, .52), astar(.668, .543)))
print('gap-matched transfer: D5 (rho .776, gap .128) %.3f -> c4 (rho .816, gap .128) %.3f ; c4D5 (rho .663, gap .151) %.3f -> c4 (rho .816, gap .151) %.3f' % (
    astar(.776, .776 - .128), astar(.816, .816 - .128), astar(.663, .663 - .151), astar(.816, .816 - .151)))
print('\nshape at matched a* (q solved per rho): values relative to g(.30)')
print(' a*    rho  |  q     | g(.1)/g(.2) | g(.2)/g(.3) g(.25)/g(.3) g(.35)/g(.3) g(.4)/g(.3) g(.45)/g(.3)')
for at in (.25, .30, .35):
    for r in (.60, .66, .78, .82, .90):
        q = brentq(lambda q: astar(r, q) - at, -0.5, r)
        G = lambda a: g(a, r, q)
        print(' %.2f  %.2f | %.3f  |   %.3f     | %.3f  %.3f  %.3f  %.3f  %.3f' % (at, r, q, G(.1) / G(.2), G(.2) / G(.3), G(.25) / G(.3), G(.35) / G(.3), G(.4) / G(.3), G(.45) / G(.3)))
