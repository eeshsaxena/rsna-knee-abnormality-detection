# e1_option.py (10-20 push synthesis, Claude): MODEL only. Generic R1 option value of ONE probe whose hidden effect d ~ N(mu, tau)
# is read once on public with paired noise s (independent of private): A switches iff the read r = d + e > 0.
# E[private gain] = E[d 1(r > 0)]; also P(switch) and P(d < 0 | switch). s for E1-on-eoa20 = gold58 bootstrap sd .00097 (MEASURED,
# e1/gate_e1_offline.log) scaled to n 390 (x sqrt(58/390)) = 3.7e-4; s for a shape fill-in is ~1.8e-4 but its d is ~99% explained by
# the A21 reads (skeptic_abc.log), which this formula cannot see, so it is NOT applied to shape fill-ins.
import numpy as np
from scipy.stats import norm
s = .00097 * np.sqrt(58 / 390)
print('paired public read sd for E1 (scaled) %.2fe-4' % (s * 1e4))
for tau in (1e-4, 2e-4, 3e-4):
    for mu in (-1e-4, -.5e-4, 0, .5e-4):
        sr = np.hypot(tau, s); z = mu / sr
        g = mu * norm.cdf(z) + tau ** 2 / sr * norm.pdf(z)
        rng = np.random.default_rng(0); d = rng.normal(mu, tau, 400000); r = d + rng.normal(0, s, 400000)
        print('tau %.1fe-4 mu %+.1fe-4 | P(switch) %.2f | E[private gain] %+.2fe-4 | P(d<0 | switch) %.2f' % (
            tau * 1e4, mu * 1e4, (r > 0).mean(), g * 1e4, (d[r > 0] < 0).mean()))
