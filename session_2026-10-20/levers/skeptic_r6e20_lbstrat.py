# skeptic_r6e20_lbstrat.py (10-20, Claude, lbstrat skeptic of the B = R6e20 slot proposal). Local arithmetic only, no data.
from math import comb, erf, sqrt
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
def p_le(k, n): return sum(comb(n, i) for i in range(k + 1)) / 2 ** n
print("K ledger sign test, 2 positives:")
for n, why in ((11, "as claimed (fr100, pf10, c2, c2max, c3max counted separately)"),
               (9, "c2 / c2max dropped: exact sums of the fr100 + pf10 atoms (sample signs implied)"),
               (8, "also c3max dropped (only adds the unread lat10 atom)")):
    print(f"  n {n:2d}: P(<=2 of n | coin) = {p_le(2, n):.3f}   {why}")
print("P(eoa20 private < c4 private) under alternative priors on the population gain d20 (e-4):")
pub, s_pub, s_priv = 10.8, 4.81, 3.15     # public read (bpick magnitude), gold-regime public / private paired sd (lbstrat_noise.log)
for mu0, sd0, tag in ((0.0, 9.1, "bpick-like wide"), (0.0, 5.0, "null, sd 5"), (-1.0, 3.0, "OAI r6_oai shaded (eoa20 ~ -1e-4)")):
    v = 1 / (1 / sd0 ** 2 + 1 / s_pub ** 2); m = v * (mu0 / sd0 ** 2 + pub / s_pub ** 2)
    print(f"  prior N({mu0:+.1f}, {sd0}) [{tag}]: post {m:+.2f} sd {sqrt(v):.2f}; P(private delta < 0) = {Phi(-m / sqrt(v + s_priv ** 2)):.3f}")
