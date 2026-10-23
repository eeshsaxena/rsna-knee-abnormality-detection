# sink_cvx_matched.py (10-21, Claude, SINK track). LOCAL, NO OAI. Two extra CVX skeptic checks on RAK-PUB (T32 target, s +0.23) and G58:
# (a) matched decorrelation: linear pw E-OA at the dose a_m whose mean label-free trio co-firing (normal-score rho) equals p2 @ .25,
#     then p2 @ .25 minus linear @ a_m (the levers 'RSNA INFO' comparison, CLAIMED negative on R7 x gold74 and c4 x gold58);
# (b) placebo direction: concave p .5 @ .25 (must not gain if the convex mechanism is real). Aggregates only. Writes sink_cvx_matched.log.
import numpy as np
from scipy.stats import norm, rankdata
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/sink_local.py', encoding='utf-8').read()
src = src.split('SUMM = {}')[0].replace("log = open(OUT + '.log', 'w')", "log = open(OUT + '_mtmp.log', 'w')")
ns = {}; exec(src, ns)
pipeline, wauc, mk, weights, L, J, OA, X, YF, NESTS, M58, Y58 = (ns[k] for k in ('pipeline', 'wauc', 'mk', 'weights', 'L', 'J', 'OA', 'X', 'YF', 'NESTS', 'M58', 'Y58'))
LOG = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/sink_cvx_matched.log', 'w')
def P(s):
    print(s, flush=True); LOG.write(s + '\n')
nsc = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
def cof(o):
    Z = [nsc(o[:, J[c]]) for c in OA]; return np.mean([np.corrcoef(Z[i], Z[j])[0, 1] for i in range(3) for j in range(i + 1, 3)])
WT = weights((0.23, 0.)); mac = lambda o, base: sum(wauc(o[WT[c][0], J[c]], YF[WT[c][0], J[c]], WT[c][1]) - wauc(base[WT[c][0], J[c]], YF[WT[c][0], J[c]], WT[c][1]) for c in OA) / 12 * 1e4
GRID = np.round(np.arange(.20, .36, .01), 3)
for nn, (ra, r6, s6) in list(NESTS.items()) + [('G58', (None, None, None))]:
    if nn == 'G58':
        M = M58; Yv = Y58; w = None
        f = lambda o, b: sum(wauc(o[:, J[c]], Yv[:, J[c]], np.ones(len(Yv))) - wauc(b[:, J[c]], Yv[:, J[c]], np.ones(len(Yv))) for c in OA) / 12 * 1e4
    else:
        M = dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]); f = mac
    b25 = pipeline(M, mk('pw25')); c25 = pipeline(M, dict(mk('pw25'), p=2.)); v25 = pipeline(M, dict(mk('pw25'), p=.5))
    target = cof(c25); lin = {a: pipeline(M, dict(mk('pw25'), a=float(a))) for a in GRID}; co = {a: cof(o) for a, o in lin.items()}
    am = min(GRID, key=lambda a: abs(co[a] - target))
    P('%-4s co-firing: lin .25 %.4f | p2 .25 %.4f | lin a_m %.2f %.4f | p.5 .25 %.4f || RAK-PUB/G58 vs pw25: p2 .25 %+.2f | lin a_m %+.2f | p2 - lin(a_m) %+.2f | concave p.5 %+.2f' % (
        nn, cof(b25), target, am, co[am], cof(v25), f(c25, b25), f(lin[am], b25), f(c25, b25) - f(lin[am], b25), f(v25, b25)))
LOG.close()
