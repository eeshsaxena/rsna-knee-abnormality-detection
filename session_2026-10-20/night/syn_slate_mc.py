# syn_slate_mc.py (10-20 night, Claude). MODEL only, no data. E[private gain vs pw30] of 10-21 A-probe slates under R1
# (A = max public among pw30 and the probes). Inputs (macro AUC units):
#   pw25 - pw30: true ~ N(+0.3e-4, 0.8e-4) (population range -0.27 to +0.90e-4, CLAIMED val/implied2.md sec 8), public paired sd 1.4e-4 (CLAIMED implied2 l.257)
#   E2 (Syn <- Eff on pw30): mixture, p_real; real ~ N(m_real, .6 m_real + .4e-4); null ~ N(m_null, same sd)
#     a .1: m_real = .0036 x .4 / 12 = +1.2e-4, m_null = -.0005 x 2 / 12 = -0.8e-4, public paired sd 1.8e-4 (MEASURED syn_more.log)
#     a .2: m_real = .0084 x .4 / 12 = +2.8e-4, m_null = -.0012 x 2 / 12 = -2.0e-4, public paired sd 2.9e-4 (MEASURED syn_more.log)
#   x .4 = the gold58 -> public-implied ratio of the scored E-OA family (gold58 pw30 +2.2e-4 vs implied +0.8 to 1.2e-4, CLAIMED VAL / implied2)
#   probes' public noises independent (different columns).
import numpy as np
rng = np.random.default_rng(7); N = 1000000
def run(p_real, a):
    m_real, m_null, sp = {.1: (1.2e-4, -0.8e-4, 1.8e-4), .2: (2.8e-4, -2.0e-4, 2.9e-4)}[a]
    s = .6 * m_real + .4e-4
    t25 = rng.normal(.3e-4, .8e-4, N); p25 = t25 + rng.normal(0, 1.4e-4, N)
    real = rng.random(N) < p_real
    tE = np.where(real, rng.normal(m_real, s, N), rng.normal(m_null, s, N)); pE = tE + rng.normal(0, sp, N)
    z = np.zeros(N)
    only25 = np.where(p25 > 0, t25, z)
    onlyE = np.where(pE > 0, tE, z)
    both = np.where((pE > p25) & (pE > 0), tE, np.where(p25 > 0, t25, z))
    harmE = ((pE > 0) & (tE < 0)).mean()
    return only25.mean(), onlyE.mean(), both.mean(), (pE > 0).mean(), (tE > 0).mean(), tE.mean(), harmE
for a in (.1, .2):
    for p in (.35, .5, .65):
        r = run(p, a)
        print('E2 a %.1f p_real %.2f | E[gain] pw25 alone %+.2fe-4 | E2 alone %+.2fe-4 | pw25 + E2 %+.2fe-4 | E2: P(move) %.2f P(true>0) %.2f E[true] %+.2fe-4 P(adopted & harmful) %.2f' % (
            a, p, r[0] * 1e4, r[1] * 1e4, r[2] * 1e4, r[3], r[4], r[5] * 1e4, r[6]))
