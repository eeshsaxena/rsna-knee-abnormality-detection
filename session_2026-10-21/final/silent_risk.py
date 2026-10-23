#!/usr/bin/env python
# silent_risk.py (10-21 final red-team, Claude). MODEL only, local CPU, no data beyond fallback_g58.json (gold58 replay costs).
# (1) swide: posterior probability that its hidden run lost part of the R6 leg silently (worker crash: rows take Raptor rank, no
#     canary), given that it displays .960 above pw25 (.959). Bayes on the public gap D = swide - pw25 (e-4):
#       no crash: D = T, T ~ N(m0, sT^2) (stack prior predictive, public);   crash: D = T - c(f), f ~ U(0, .5), c(f) = h x gold58 cost.
#     The read is summarised as D_obs ~ N(Dhat, .5^2) (surface 1.39 / bpick 2.03). Private bias of the no-crash model given a crash:
#     -(1 - s) c(f), s = population share of var(T) (the part of a larger clean T the model would have credited to private).
# (2) r6pw20 resubmit: a silently degraded copy (no canary) passes the >= .958 display gate when its clean public margin over .9580,
#     m ~ N(9.7, 3.9) e-4 (bpick posterior: P(.957 / .958 / .959) .006 / .522 / .472), exceeds c(f). Expected B degradation that
#     slips through the gate, per resubmit, and the silver cost via the clean-vs-R6 linearisation (+0.67 step / +1.27 smooth pts per
#     +8.1e-4 of B, bpick base).
import json, numpy as np
from math import erf, sqrt
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/final'
J = json.load(open(HERE + '/fallback_g58.json'))
fs = np.array([0.0] + sorted(float(k) for k in J['r6_fail']))
cost = {nm: np.array([0.0] + [-J['r6_fail'][k][nm][0] for k in sorted(J['r6_fail'], key=float)]) for nm in ('swide', 'r6pw20')}
c_of = lambda nm, f, h: h * np.interp(f, fs, cost[nm])
npdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * sqrt(2 * np.pi))
Phi = lambda z: 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))
fgrid = np.linspace(0, .5, 2001)
out = []


def say(s):
    print(s, flush=True); out.append(s)


say('(1) swide: P(silent R6-leg crash | read) and the private bias it implies (e-4)')
say('  prior pi = P(crash in one hidden run), GUESS; f ~ U(0, .5); h = gold58 -> LB cost haircut; sT = public sd of the stack prior')
for Dhat in (1.39, 2.03):
    for sT in (2.5, 3.0, 3.5):
        for h in (.5, 1.0):
            s = 2.1 / sT ** 2                    # population variance share (~2.1 of var(T), skeptic_wc priors, GUESS-level)
            sd = sqrt(sT ** 2 + .25)
            L0 = npdf(Dhat, .24, sd)
            cf = c_of('swide', fgrid, h)
            Lc = npdf(Dhat + cf, .24, sd)          # the clean gap would have had to be Dhat + c(f)
            LR = Lc.mean() / L0
            Ec = (Lc * cf).mean() / Lc.mean()      # E[c(f) | crash, read]
            row = []
            for pi in (.02, .05, .10, .20):
                post = pi * LR / (pi * LR + 1 - pi)
                row.append('pi %.2f -> %.4f (bias %+.3f)' % (pi, post, -post * (1 - s) * Ec))
            say('  Dhat %.2f sT %.1f h %.1f | LR %.3f, E[cost | crash, read] %.2f | ' % (Dhat, sT, h, LR, Ec) + ' ; '.join(row))
say('\n(2) r6pw20 resubmit: degraded-but-passing copies (per copy, given a silent crash)')
for h in (.5, 1.0):
    cf = c_of('r6pw20', fgrid, h)
    ppass = Phi((9.7 - cf) / 3.9)                 # P(m >= c(f))
    P_pass = ppass.mean(); E_slip = (ppass * cf).mean()
    worse = (ppass * (cf > 8.1)).mean()           # passes and is privately below R6 in expectation (clean edge +8.1)
    say('  h %.1f | P(pass .958 | crash) %.3f | E[cost x 1(pass)] %.2f e-4 | P(pass and below R6) %.3f' % (h, P_pass, E_slip, worse))
    for pi in (.02, .05, .10, .20):
        e = pi * E_slip
        say('     pi %.2f: expected B shift %.3f e-4 -> silver %.3f step / %.3f smooth pts (linearised)' % (pi, -e, -e * .67 / 8.1, -e * 1.27 / 8.1))
open(HERE + '/silent_risk.log', 'w').write('\n'.join(out) + '\n')
