# skep_moacf_r1.py (10-21 levers, Claude, SKEPTIC of MOACF part 3). MODEL only, no data. Writes skep_moacf_r1.log.
# Corrected prior for one MOACF entry on the A22-best base, read paired vs that base under R1 (pick the higher public).
# Mixture (GUESS weights): transfers (lognormal median .55e-4, sd log .6) w .65 | c4 has no such bias (perturbation cost N(-.1,.05)) w .25 |
# sign flip on expert hidden labels N(-.5,.3) w .10. Public paired read sd 1.5 (rulers 1.8 MEASURED moacf_noise.log; c4 tighter, GUESS);
# private paired sd 1.0 (rulers 1.1-1.2 MEASURED). Output: E theta, P(theta>0), P(selected), E private gain of the R1 pick vs base,
# P(R1 pick worse than base on private).
import numpy as np
rng = np.random.default_rng(7); N = 2_000_000
out = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/skep_moacf_r1.log', 'w')
def P(s): print(s); out.write(s + '\n')
for tag, w, med, sdp in (('base', (.65, .25, .10), .55, 1.5), ('optim', (.75, .18, .07), .8, 1.5), ('pess', (.5, .35, .15), .4, 1.5), ('base sd1.8', (.65, .25, .10), .55, 1.8)):
    k = rng.choice(3, N, p=w)
    th = np.where(k == 0, med * np.exp(.6 * rng.standard_normal(N)), np.where(k == 1, -.1 + .05 * rng.standard_normal(N), -.5 + .3 * rng.standard_normal(N)))
    pub = th + sdp * rng.standard_normal(N); prv = th + 1.0 * rng.standard_normal(N); sel = pub > 0
    P('%-10s E theta %+.2f e-4 | P(theta>0) %.2f | q05/q95 %+.2f/%+.2f | P(sel) %.2f | E private gain of R1 pick %+.2f e-4 | P(R1 pick worse on private) %.2f' % (
        tag, th.mean(), (th > 0).mean(), np.quantile(th, .05), np.quantile(th, .95), sel.mean(), (prv * sel).mean(), (sel & (prv < 0)).mean()))
