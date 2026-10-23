# recal_table.py (10-20 night, Claude, TRACK 1). Arithmetic only (no data). Writes recal_table.log.
# Turns each candidate's calibrated population effect delta ~ N(mu, tau^2) (macro e-4 vs pw30; chosen in recal.md sec 4 from the
# MEASURED / CLAIMED rulers) and its paired read noise (MEASURED recal_transfer.log / recal_oa_targets.log, census_d.py method) into
#   P(best-id move vs pw30) = P(delta + e_pub > 0), e_pub ~ N(0, s_pub^2)      (R1: best-id = max unrounded public)
#   R1 value              = E[delta * 1(delta + e_pub > 0)]                       (private gain of probing then following R1)
#   P(private delta > 0)  = P(delta + e_priv > 0)
# Verdict rule fixed before reading the numbers: PROBE if mu >= +1.0 and P(mu > 0) >= .80; MAYBE if mu > 0 and P(mu > 0) >= .60; else NO.
import numpy as np
from scipy.stats import norm
C = [  # name, mu, tau, s_pub, s_priv
    ('C6 menisci MM<->LM a .1 (as killed)', -.7, 1.0, 2.25, 1.52),
    ("C6' LM<-MM a .05 only (re-dosed)", .4, .6, 1.0, .65),
    ('C7 fluid FL3 a .1', -.5, 1.0, 3.03, 1.88),
    ('C8 fluid ES2 Eff<->Syn a .1', -.5, .8, 2.04, 1.17),
    ('C9 OA-meniscus 4 dirs a .1', -.8, .4, 3.61, 2.12),
    ('C10 ACL <- LM / MM / Eff a .1', -.5, .5, 2.0, 1.25),
    ('C11 global G12 a .1', 0., 2.0, 3.97, 2.56),
    ('C12 LOA<-MM a .1 on pw30', .1, .8, 1.93, 1.29),
    ("C12' LOA<-MM a .05 on pw30", .3, .6, 1.0, .65),
    ('C13 BOR MOA+=MM, LOA+=LM a -.1', .2, 1.2, 2.33, 1.37),
    ('C14 LOA+=LM a -.1 on pw30', .3, 1.0, 1.73, 1.14),
    ('MCL<-MM a .1', -.2, 1.0, 2.66, 1.44),
    ('Frac<-MOA a .1', -.3, 1.0, 3.60, 2.05),
    ('MCL<-mean11 a .1', 0., .6, 1.47, .88),
    ('LOA<-Eff borrow a -.1', -.3, 1.0, 1.84, 1.15),
    ('MCL<-Syn/Cont/Frac/MOA a .1', -.3, 1.0, 2.7, 1.55),
    ('LM<-MM a .1 (census KILL)', -.1, .8, 1.93, 1.16),
    ('MM<-LOA a .1 (census KILL)', -1.0, .5, 1.48, .91),
    ('MCL+Frac minus mean(DEG) a .1', -.2, 1.0, 3.53, 2.11),
    ('Cont<->Frac a .1', -.1, .8, 2.99, 1.85),
    ('E2 Syn+=Eff a -.1', -.1, 1.0, 1.57, .89),
    ('E2 Syn+=Eff a -.2', -.5, 1.5, 2.66, 1.62),
    ('convex E-OA twin (pw25p2 / pw20p2)', .2, 1.0, 2.2, 1.4),
    ('SUB_HIGH E-OA variant', .2, 1.0, 2.2, 1.4),
    ("stack LM<-MM .05 + LOA<-MM .05 on pw30", .6, .9, 1.4, .9),
]
log = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/recal_table.log', 'w')
def P(s): print(s); log.write(s + '\n')
P('%-44s | %-22s | P(mu>0) | s_pub | P(move) | R1 value | P(priv>0) | verdict' % ('candidate (vs pw30, macro e-4)', 'mu [p10, p90]'))
for nm, mu, tau, sp, sv in C:
    S = np.hypot(tau, sp); z = mu / S
    pm = norm.cdf(z); r1 = mu * pm + tau ** 2 / S * norm.pdf(z); pp = norm.cdf(mu / np.hypot(tau, sv)); p0 = norm.cdf(mu / tau)
    v = 'PROBE' if (mu >= 1. and p0 >= .8) else ('MAYBE' if (mu > 0 and p0 >= .6) else 'NO')
    P('%-44s | %+.1f [%+.1f, %+.1f]%s | %.2f | %.2f | %.2f | %+.2f | %.2f | %s' % (nm, mu, mu - 1.2816 * tau, mu + 1.2816 * tau, ' ' * 3, p0, sp, pm, r1, pp, v))
