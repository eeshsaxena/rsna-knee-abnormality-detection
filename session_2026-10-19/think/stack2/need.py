# need.py: what a true macro effect mu buys (MODEL). Public paired sd and private paired sd are the MEASURED subsample sds of an
# operator change on the OOF ruler (subsample_sd.log: pow2 .00041 / .00025; invgeo .00048 / .00032). Gap to display .959 ~ N(3.4e-4, .94e-4)
# (density.md p50 3.4e-4, p10-p90 2.2-4.6e-4). Prior on the real effect of a stacking change: mu itself, or mu ~ N(m, s) for shrinkage.
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(0)
for sdp, sdv, nm in ((.00041, .00025, 'pow2-like'), (.00048, .00032, 'invgeo-like'), (.00030, .00017, 'weight tweak')):
    print('noise class %s (public sd %.5f, private sd %.5f)' % (nm, sdp, sdv))
    for mu in (0, 1e-4, 2e-4, 3e-4, 4e-4, 6e-4):
        gap = rng.normal(3.4e-4, .94e-4, 200000); pub = mu + rng.normal(0, sdp, 200000); prv = mu + rng.normal(0, sdv, 200000)
        print('  mu %+.4f: P(public delta > 0, best-id move) %.2f | P(shows .959) %.2f | P(private delta > 0) %.2f | P(priv>0 | pub>0) %.2f' % (
            mu, (pub > 0).mean(), (pub > gap).mean(), (prv > 0).mean(), (prv[pub > 0] > 0).mean()))
# pair flips per label for +.0041 single-label AUC (= +3.4e-4 macro) at the public size 390, two prevalence regimes
for lab, pg, pr in (('Synovitis', .47, .12), ('Fracture', .31, .07), ('Lateral OA', .19, .27), ('PF OA', .36, .46), ('ACL', .41, .20)):
    for reg, p in (('gold58-like', pg), ('report-like', pr)):
        npos = round(390 * p); pairs = npos * (390 - npos)
        print('%-11s %-11s prev %.2f: %3d pos, %6d pairs, +.0041 AUC = %4d pair flips' % (lab, reg, p, npos, pairs, round(.0041 * pairs)))
print('--- Bayesian: mu ~ N(m, s) prior (MODEL), public read informs private ---')
for m, s, sdp, sdv, nm in ((2e-4, 3e-4, .00041, .00025, 'pow2 nested (prior m 2e-4, s 3e-4)'), (1e-4, 3e-4, .00041, .00025, 'Syn<-Eff a.1 (prior m 1e-4, s 3e-4)'),
                           (0, 3e-4, .00041, .00025, 'zero-mean change')):
    mu = rng.normal(m, s, 400000); pub = mu + rng.normal(0, sdp, 400000); prv = mu + rng.normal(0, sdv, 400000); gap = rng.normal(3.4e-4, .94e-4, 400000)
    print('%-38s P(pub>0) %.2f P(priv>0) %.2f | P(priv>0 | pub>0) %.2f | P(priv>0 | pub>gap i.e. shows .959) %.2f | E[priv delta | pub>0] %+.5f' % (
        nm, (pub > 0).mean(), (prv > 0).mean(), (prv[pub > 0] > 0).mean(), (prv[pub > gap] > 0).mean(), prv[pub > 0].mean()))
