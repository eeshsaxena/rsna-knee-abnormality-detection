# synth_convex_r1.py (10-20, Claude, LEVERS_1020 synthesis). Local arithmetic only, no data.
# R1 value of ONE convex twin of the A21 winner: A switches to the twin iff its public read beats the winner.
# theta = population macro gain of twin over winner (e-4); public read = theta + e, e ~ N(0, s_pub); private = theta + noise.
# R1 value = E[theta * 1(read > 0)]; wrong-sign share = P(theta < 0 | read > 0).
from math import erf, exp, pi, sqrt
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2))); phi = lambda z: exp(-z * z / 2) / sqrt(2 * pi)
import random
random.seed(5)
s_pub = 2.2   # paired public sd of a same-a twin, e-4 (MEASURED box_F5.log / box_F5_R3plo.log, space.md 4.2)
priors = [("space lens (held-out a only)", 1.0, 0.5),
          ("skeptic: fixed-p2 held-out, transfer-shaded", 0.5, 0.5),
          ("skeptic: whole-search held-out (selection-corrected)", 0.15, 0.5),
          ("RSNA INFO shaded (gold58 / gold74 lean negative)", -0.2, 0.6)]
for tag, mu, tau in priors:
    sr = sqrt(tau ** 2 + s_pub ** 2)
    pmove = Phi(mu / sr)
    val = mu * pmove + tau ** 2 * phi(mu / sr) / sr
    n = 200000; ws = 0; mv = 0
    for _ in range(n):
        th = random.gauss(mu, tau); r = th + random.gauss(0, s_pub)
        if r > 0:
            mv += 1; ws += th < 0
    print(f"{tag:55s} prior N({mu:+.2f}, {tau}) | P(move) {pmove:.2f} | R1 value {val:+.2f}e-4 | P(wrong sign | move) {ws / max(mv, 1):.2f}")
