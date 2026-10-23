# public_pair_mc.py (10-22 final, Claude). MODEL only, no data read. Which 2 public-only entries maximise the public display gain
# over swide 57464326? Display = best public of all our subs, so a losing entry costs nothing on public: value = E[max(0, d1, d2)]
# where d = (entry public - swide public) in 1e-4, times P(clean rerun). Inputs (CLAIMED):
#   MOACF(swide): population N(+.35, .46) (lastday/surface.md row), public paired read noise sd 1.82 (levers/moacf_noise.log)
#   CONS(swide) = cvxswide: population N(+.30, 1.1) pw25 family (LEVERS_1021 row 2), haircut for the lomm05 overlap -> +.15;
#                 skeptic variant -1.0 (gold58 -3.2..-5.3 public-equivalent); noise sd 2.66 (levers/sink_read_noise.log cvx25_on_spl)
#   swide08 (lomm .05 -> .08): N(+.11, .886) total (surface.md); MCLmm .15: N(-.02, .83) total (surface.md)
#   swide08+MOACF = MOACF + L8 increments (additive; surface sd 2.08 matches sqrt(1.87^2 + .886^2))
# Read-error correlation MOACF~CVX unknown: rho 0 / .3 / .6. P(clean) per entry .55 [.35-.80] (CLAIMED CANARY_1021), copula rho_c .3.
# Rank: ~6 teams per 1e-4 inside the .960 band (GUESS surface.md 5-7; 51 teams in band, 48 above us, MEASURED lb_bands.out).
import numpy as np
rng = np.random.default_rng(1022)
N = 400000
def run(pc, rho, mu_cvx, rho_c=.3):
    z = rng.standard_normal((N, 6))
    M = .35 + .46 * z[:, 0] + 1.82 * z[:, 1]
    nc = rho * z[:, 1] + np.sqrt(1 - rho ** 2) * z[:, 2]
    X = mu_cvx + 1.1 * z[:, 3] + 2.66 * nc
    L8 = .11 + .886 * z[:, 4]
    C15 = -.02 + .83 * z[:, 5]
    ML = M + L8
    # correlated clean indicators
    u = rng.standard_normal((N, 2)); u[:, 1] = rho_c * u[:, 0] + np.sqrt(1 - rho_c ** 2) * u[:, 1]
    from math import erf, sqrt
    thr = np.quantile(rng.standard_normal(200000), pc)
    c1, c2 = u[:, 0] < thr, u[:, 1] < thr
    E = {'MOACF': M, 'CONS': X, 'swide08': L8, 'MCLmm15': C15, 'swide08+MOACF': ML, 'MCLmm15+MOACF': M + C15}
    pairs = [('MOACF', 'CONS'), ('MOACF', 'swide08'), ('MOACF', 'swide08+MOACF'), ('swide08+MOACF', 'CONS'),
             ('MOACF', 'MCLmm15'), ('swide08+MOACF', 'MCLmm15+MOACF'), ('MOACF', None), ('CONS', None)]
    out = []
    for a, b in pairs:
        da = np.where(c1, E[a], -99); db = np.where(c2, E[b], -99) if b else np.full(N, -99.)
        g = np.maximum(0, np.maximum(da, db))
        out.append((a, b, (g > 0).mean(), g.mean(), np.minimum(48, 6 * g).mean(), np.quantile(np.minimum(48, 6 * g), .9)))
    return out
for pc in (.55, .8):
    for rho in (0, .3, .6):
        for mu in (.15, -1.0):
            print('P(clean) %.2f  rho(MOACF,CVX noise) %.1f  CONS pop mean %+.2f' % (pc, rho, mu))
            for a, b, p, e, r, r90 in run(pc, rho, mu):
                print('   %-28s P(public gain) %.3f  E gain %.2fe-4  E places %.1f  p90 places %.0f' % (a + (' + ' + b if b else ''), p, e, r, r90))
