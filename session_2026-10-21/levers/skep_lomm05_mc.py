# skep_lomm05_mc.py (10-21 levers, Claude, lomm05 SKEPTIC). MODEL only: lomm05 population effect (macro e-4, public = private mean) as the
# stack round's scenario mixture (CLAIMED stack/inventory.md sec 6: S1 train-like, S2 c4 co-firing carries signal, S3 gold-like; x .85 search
# haircut; scenario sds 1.23 / 1.23 / 1.65 CLAIMED night skep_c12p_post.log) with S1 replaced by the label-source-calibrated EB reads of
# skep_lomm05.log x skep_lomm05_cal.log (nest mean x own public / E-OA factor: v2 .88, v4 .62, sol56 .55) and S3 by -0.26..0.
import numpy as np
rr = np.random.default_rng(21); N = 400000
def run(tag, S1, S3, w=(.40, .35, .25), S2=-.48, h=.85):
    k = rr.choice(3, N, p=w); mu = np.where(k == 0, S1, np.where(k == 1, S2, S3)); sd = np.where(k == 2, 1.65, 1.23)
    x = h * rr.normal(mu, sd)
    print('%-58s E %+.2f [p10 %+.2f, p90 %+.2f] P(>0) %.2f' % (tag, x.mean(), *np.percentile(x, [10, 90]), (x > 0).mean()))
run('stack MODEL as built (S1 1.10, S3 -0.26)  [parity +0.17]', 1.10, -.26)
run('skeptic central (S1 .68 = mean of 3 sources, S3 -0.13)', .68, -.13)
run('skeptic low (S1 .55 sol56/v4, S3 -0.26)', .55, -.26)
run('skeptic high (S1 .88 v2 only, S3 0, NIGHT weights)', .88, 0., w=(.45, .30, .25))
run('v2 only, stack weights (S1 .88, S3 -0.26)', .88, -.26)
